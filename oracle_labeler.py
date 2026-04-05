"""Oracle labeling script for training the Learned Complexity Router (LCR).

Supports **multi-method composite sensitivity scoring**: the oracle measures
PPL degradation under multiple pruning methods (e.g., attention head pruning
AND layer skipping) and produces a single aggregated sensitivity label.

Creates a CSV with columns (multi-method mode):
    Prompt, ppl_dense, loss_dense,
    ppl_sparse_{method}, loss_sparse_{method}, gap_{method} (PPL gap), loss_gap_{method},
    raw_composite (+ raw_composite_ppl/raw_composite_loss), normalized_sensitivity, sensitivity_score

Single-method mode (legacy, backward-compatible):
    Prompt, ppl_dense, loss_dense, ppl_sparse, loss_sparse, raw_gap (PPL gap), loss_gap, raw_metric,
    normalized_sensitivity, sensitivity_score

Dense pass:  base Llama-3.2-1B perplexity (no pruning)
Sparse pass: one pass per pruning config in --sparse-configs

Usage examples:
  # Multi-method (recommended for generalized LCR):
  python oracle_labeler.py \\
    --input lcr_mixture_5k.csv \\
    --output oracle_lcr_5k_composite.csv \\
    --sparse-configs "attention_heads:0.30,transformer_layers:0.25" \\
    --device gpu --samples 5000

  # Legacy single-method (backward compatible):
  python oracle_labeler.py \\
    --input lcr_mixture.csv \\
    --output oracle_lcr.csv \\
    --sparse-target attention_heads --sparse-intensity 0.50 \\
    --device gpu --samples 500

Notes:
- This script uses model loss -> perplexity (teacher-forcing).
- For a more stable oracle target, prefer `--gap-metric loss_gap` (equivalently log-PPL ratio).
- Sparse passes use this repo's pruning engine (no Wanda, no ONNX).
- Cross-method Spearman correlation is computed and logged when multi-method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from model_loader import RealModelEngine

try:
    from scipy.stats import spearmanr as _spearmanr, pearsonr as _pearsonr

    _SCIPY = True
except ImportError:
    _spearmanr = None
    _pearsonr = None
    _SCIPY = False


@dataclass
class FixedPruningAction:
    target: str
    intensity: float


@dataclass(frozen=True)
class PromptRow:
    prompt: str
    meta: Dict[str, str]


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_prompt_loss_and_ppl(model_engine: RealModelEngine, prompt: str, max_length: int = 128) -> Tuple[float, float]:
    """Compute prompt loss and perplexity using the model's teacher-forcing loss."""

    tok = model_engine.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
    )
    tok = {k: v.to(model_engine.model.device) for k, v in tok.items()}

    # Standard HF CausalLM: loss is cross-entropy over shifted tokens.
    outputs = model_engine.model(**tok, labels=tok["input_ids"], use_cache=False, return_dict=True)
    loss = float(outputs.loss.detach().cpu().item())
    # Cap exp to avoid inf on pathological prompts; keep raw loss for stable labeling.
    ppl = math.exp(min(20.0, loss))
    return float(loss), float(ppl)


@torch.no_grad()
def compute_prompt_ppl(model_engine: RealModelEngine, prompt: str, max_length: int = 128) -> float:
    """Backward-compatible wrapper returning only PPL."""

    _, ppl = compute_prompt_loss_and_ppl(model_engine, prompt, max_length=int(max_length))
    return float(ppl)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _parse_csv_list(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (0..100)."""

    if not values:
        return 0.0
    if p <= 0.0:
        return float(min(values))
    if p >= 100.0:
        return float(max(values))

    xs = sorted(float(x) for x in values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def _parse_sparse_configs(s: str) -> List[FixedPruningAction]:
    """Parse 'target:intensity,target:intensity,...' into a list of actions."""
    configs: List[FixedPruningAction] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid sparse config '{part}'. Expected 'target:intensity'.")
        target, intensity_str = part.split(":", 1)
        target = target.strip()
        intensity = float(intensity_str.strip())
        configs.append(FixedPruningAction(target=target, intensity=intensity))
    return configs


def _normalize_pclip_minmax(
    values: Sequence[float], p_low: float, p_high: float
) -> Tuple[List[float], Dict[str, float]]:
    """Percentile-clipped min-max normalization to [0, 1]."""
    clip_low = _percentile(values, p_low)
    clip_high = _percentile(values, p_high)
    if clip_high <= clip_low:
        clip_high = clip_low + 1e-8

    denom = max(1e-8, clip_high - clip_low)
    labels = []
    for v in values:
        vc = min(max(float(v), clip_low), clip_high)
        labels.append(clamp01((vc - clip_low) / denom))

    meta = {
        "method": "pclip_minmax",
        "p_low": p_low,
        "p_high": p_high,
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
    }
    return labels, meta


def _normalize_minmax(values: Sequence[float]) -> Tuple[List[float], Dict[str, float]]:
    min_g = float(min(values))
    max_g = float(max(values))
    denom = max(1e-8, max_g - min_g)
    labels = [clamp01((float(v) - min_g) / denom) for v in values]
    return labels, {"method": "minmax", "min_gap": min_g, "max_gap": max_g}


def _normalize_ratio(
    values: Sequence[float], dense_ppls: Sequence[float]
) -> Tuple[List[float], Dict[str, float]]:
    labels = []
    for g, pd in zip(values, dense_ppls):
        labels.append(clamp01(float(g) / max(1e-8, float(pd))))
    return labels, {"method": "ratio", "eps": 1e-8}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_prompt_rows_from_csv(
    path: str,
    text_column: str,
    max_samples: int,
    carry_columns: Sequence[str],
) -> Tuple[List[PromptRow], List[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if text_column not in fieldnames:
            raise ValueError(f"CSV column '{text_column}' not found. Columns: {fieldnames}")

        carry = [c for c in carry_columns if c in fieldnames and c != text_column]

        rows: List[PromptRow] = []
        for row in reader:
            prompt = (row.get(text_column) or "").strip()
            if not prompt:
                continue
            meta = {c: (row.get(c) or "") for c in carry}
            rows.append(PromptRow(prompt=prompt, meta=meta))
            if int(max_samples) > 0 and len(rows) >= int(max_samples):
                break

    return rows, carry


# ---------------------------------------------------------------------------
# Cross-method correlation analysis
# ---------------------------------------------------------------------------

def _compute_cross_method_correlation(
    gaps: Dict[str, List[float]],
) -> Dict[str, float]:
    """Compute pairwise Spearman, Pearson, and R² correlations between pruning methods."""
    result: Dict[str, float] = {}
    keys = sorted(gaps.keys())
    if len(keys) < 2 or not _SCIPY:
        return result

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            g1, g2 = gaps[k1], gaps[k2]
            n = min(len(g1), len(g2))
            if n < 3:
                continue
            try:
                sp = _spearmanr(g1[:n], g2[:n])
                result[f"spearman_{k1}_vs_{k2}"] = float(sp.correlation) if sp.correlation == sp.correlation else 0.0
                result[f"spearman_pvalue_{k1}_vs_{k2}"] = float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0
            except Exception:
                pass
            try:
                pr = _pearsonr(g1[:n], g2[:n])
                r_val = float(pr[0]) if pr[0] == pr[0] else 0.0
                result[f"pearson_{k1}_vs_{k2}"] = r_val
                result[f"r2_{k1}_vs_{k2}"] = r_val ** 2
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle labeler for LCR training — supports multi-method composite sensitivity."
    )
    parser.add_argument("--input", required=True, help="Input prompt CSV")
    parser.add_argument("--output", required=True, help="Output oracle CSV")
    parser.add_argument("--text-column", default="Prompt", help="CSV column containing the prompt text")
    parser.add_argument("--samples", type=int, default=5000, help="Max prompts to label")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--backbone-model",
        default="meta-llama/Llama-3.2-1B",
        help="HF model id for oracle PPL labeling (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load from local HF cache (no network).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max tokens used for oracle loss/PPL computation (recommend matching MiniBERT max_length)",
    )

    parser.add_argument(
        "--gap-metric",
        choices=["loss_gap", "ppl_gap"],
        default="loss_gap",
        help=(
            "Raw oracle gap metric to aggregate/normalize. "
            "'loss_gap' uses (loss_sparse - loss_dense) (numerically stable; equals log-PPL ratio). "
            "'ppl_gap' uses (ppl_sparse - ppl_dense) for backward compatibility."
        ),
    )
    parser.add_argument(
        "--allow-negative-gaps",
        action="store_true",
        help="Allow negative gaps (pruning improves) to contribute to the composite; default clamps gaps at 0 for stable sensitivity labels.",
    )

    # --- Multi-method config (new, recommended) ---
    parser.add_argument(
        "--sparse-configs",
        default="",
        help=(
            "Comma-separated list of pruning configs as 'target:intensity'. "
            "Example: 'attention_heads:0.30,transformer_layers:0.25'. "
            "When provided, computes composite sensitivity across ALL methods. "
            "Overrides --sparse-target and --sparse-intensity."
        ),
    )
    parser.add_argument(
        "--composite-weights",
        default="",
        help=(
            "Optional comma-separated weights for each sparse config (must match count). "
            "Default: equal weight. Example: '0.5,0.5' or '0.6,0.4'."
        ),
    )

    # --- Legacy single-method config (backward compatible) ---
    parser.add_argument(
        "--sparse-target",
        choices=["attention_heads", "transformer_layers"],
        default="attention_heads",
        help="Fixed pruning target for single-method sparse pass (legacy)",
    )
    parser.add_argument("--sparse-intensity", type=float, default=0.50,
                        help="Fixed pruning intensity for single-method sparse pass (legacy)")

    parser.add_argument(
        "--normalization",
        choices=["ratio", "minmax", "pclip_minmax"],
        default="pclip_minmax",
        help=(
            "How to normalize the oracle label into [0,1]. "
            "'pclip_minmax' clips raw values to percentiles then min-max (recommended). "
            "'ratio' computes (gap / ppl_dense) and is only meaningful when --gap-metric=ppl_gap."
        ),
    )
    parser.add_argument(
        "--clip-percentiles",
        default="5,95",
        help="For pclip_minmax: low,high percentiles (e.g. '5,95').",
    )
    parser.add_argument(
        "--carry-columns",
        default="Category,Subject,Context Dependency,SourceDataset,SourceSplit,SourceId,Choices,AnswerIndex,AnswerLetter,AnswerText,Gsm8kAnswer,MbppTests,Split",
        help="Comma-separated columns to carry from input -> output (if present).",
    )
    parser.add_argument(
        "--meta-json",
        default="",
        help="Optional path to write a JSON sidecar with oracle + normalization metadata.",
    )
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N samples")
    args = parser.parse_args()

    t_start = time.time()

    # ---- Determine pruning configs ----
    sparse_configs_str = str(args.sparse_configs).strip()
    if sparse_configs_str:
        sparse_configs = _parse_sparse_configs(sparse_configs_str)
        multi_method = len(sparse_configs) > 1
    else:
        # Legacy single-method mode
        sparse_configs = [FixedPruningAction(target=args.sparse_target, intensity=float(args.sparse_intensity))]
        multi_method = False

    if not sparse_configs:
        raise SystemExit("No sparse configs provided.")

    # Parse composite weights
    weights: List[float] = []
    if str(args.composite_weights).strip():
        weights = [float(w.strip()) for w in str(args.composite_weights).split(",") if w.strip()]
        if len(weights) != len(sparse_configs):
            raise SystemExit(
                f"--composite-weights count ({len(weights)}) must match --sparse-configs count ({len(sparse_configs)})"
            )
    else:
        # Equal weights
        weights = [1.0 / len(sparse_configs)] * len(sparse_configs)

    # Normalize weights to sum to 1
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    print(f"[Oracle] Mode: {'multi-method composite' if multi_method else 'single-method (legacy)'}")
    for i, (cfg, w) in enumerate(zip(sparse_configs, weights)):
        print(f"[Oracle]   Config {i+1}: {cfg.target} @ {cfg.intensity:.0%} (weight={w:.2f})")

    # ---- Load backbone ----
    device = "cuda" if (args.device == "gpu" and torch.cuda.is_available()) else "cpu"
    print(f"[Oracle] Loading backbone on {device}...")
    engine = RealModelEngine(
        device=device,
        model_name=str(args.backbone_model),
        local_files_only=bool(args.local_files_only),
    )

    # ---- Load prompts ----
    carry_columns = _parse_csv_list(str(args.carry_columns))
    rows, carried = load_prompt_rows_from_csv(
        args.input,
        text_column=str(args.text_column),
        max_samples=int(args.samples),
        carry_columns=carry_columns,
    )
    if not rows:
        raise SystemExit("No prompts found in input CSV.")

    prompts = [r.prompt for r in rows]
    n_prompts = len(prompts)
    total_passes = 1 + len(sparse_configs)  # 1 dense + N sparse
    print(f"[Oracle] Loaded {n_prompts} prompts. Will do {total_passes} passes ({total_passes * n_prompts} forward calls).")

    # ---- Dense pass ----
    print(f"\n[Oracle] === Dense Pass (1/{total_passes}) ===")
    dense_ppl: List[float] = []
    dense_loss: List[float] = []
    engine.restore_model()
    t_dense_start = time.time()
    for idx, prompt in enumerate(prompts, start=1):
        if int(args.progress_every) > 0 and (idx % int(args.progress_every) == 0):
            elapsed = time.time() - t_dense_start
            eta = (elapsed / idx) * (n_prompts - idx)
            print(f"[Oracle] Dense {idx}/{n_prompts} (ETA: {eta:.0f}s)")
        loss_d, ppl_d = compute_prompt_loss_and_ppl(engine, prompt, max_length=int(args.max_length))
        dense_loss.append(float(loss_d))
        dense_ppl.append(float(ppl_d))
    t_dense = time.time() - t_dense_start
    print(f"[Oracle] Dense pass done in {t_dense:.1f}s ({t_dense/n_prompts*1000:.0f}ms/sample)")

    # ---- Sparse passes ----
    # gaps_by_method[config_key] = [gap_for_prompt_0, gap_for_prompt_1, ...]
    gaps_by_method: Dict[str, List[float]] = {}
    loss_gaps_by_method: Dict[str, List[float]] = {}
    sparse_ppls_by_method: Dict[str, List[float]] = {}
    sparse_losses_by_method: Dict[str, List[float]] = {}

    for pass_idx, cfg in enumerate(sparse_configs, start=2):
        config_key = f"{cfg.target}_{cfg.intensity:.2f}".replace(".", "p")
        print(f"\n[Oracle] === Sparse Pass {pass_idx}/{total_passes}: {cfg.target} @ {cfg.intensity:.0%} ===")

        engine.restore_model()
        engine.apply_pruning(cfg)
        sparse_ppl: List[float] = []
        sparse_loss: List[float] = []
        t_sparse_start = time.time()

        for idx, prompt in enumerate(prompts, start=1):
            if int(args.progress_every) > 0 and (idx % int(args.progress_every) == 0):
                elapsed = time.time() - t_sparse_start
                eta = (elapsed / idx) * (n_prompts - idx)
                print(f"[Oracle] Sparse[{cfg.target[:4]}] {idx}/{n_prompts} (ETA: {eta:.0f}s)")
            loss_s, ppl_s = compute_prompt_loss_and_ppl(engine, prompt, max_length=int(args.max_length))
            sparse_loss.append(float(loss_s))
            sparse_ppl.append(float(ppl_s))

        t_sparse = time.time() - t_sparse_start
        print(f"[Oracle] Sparse pass [{cfg.target}] done in {t_sparse:.1f}s ({t_sparse/n_prompts*1000:.0f}ms/sample)")

        gaps = [float(ps - pd) for pd, ps in zip(dense_ppl, sparse_ppl)]
        loss_gaps = [float(ls - ld) for ld, ls in zip(dense_loss, sparse_loss)]
        if not bool(args.allow_negative_gaps):
            gaps = [max(0.0, float(g)) for g in gaps]
            loss_gaps = [max(0.0, float(g)) for g in loss_gaps]
        gaps_by_method[config_key] = gaps
        loss_gaps_by_method[config_key] = loss_gaps
        sparse_ppls_by_method[config_key] = sparse_ppl
        sparse_losses_by_method[config_key] = sparse_loss

    # Always restore to keep engine usable after labeling.
    engine.restore_model()

    # ---- Cross-method correlation analysis ----
    gap_metric = str(args.gap_metric).strip() or "loss_gap"
    cross_corr: Dict[str, float] = {}
    if multi_method:
        gaps_for_corr = loss_gaps_by_method if gap_metric == "loss_gap" else gaps_by_method
        cross_corr = _compute_cross_method_correlation(gaps_for_corr)
        print(f"\n[Oracle] === Cross-Method Correlation Analysis ===")
        for k, v in sorted(cross_corr.items()):
            print(f"[Oracle]   {k}: {v:.4f}")

    # ---- Compute composite raw values ----
    method_keys = list(gaps_by_method.keys())
    # Composite = weighted average of per-method gaps
    composite_raw_ppl: List[float] = []
    composite_raw_loss: List[float] = []
    for i in range(n_prompts):
        v_ppl = 0.0
        v_loss = 0.0
        for mk, w in zip(method_keys, weights):
            v_ppl += w * gaps_by_method[mk][i]
            v_loss += w * loss_gaps_by_method[mk][i]
        composite_raw_ppl.append(v_ppl)
        composite_raw_loss.append(v_loss)

    composite_raw: List[float] = composite_raw_loss if gap_metric == "loss_gap" else composite_raw_ppl

    # ---- Normalize ----
    parts = _parse_csv_list(str(args.clip_percentiles))
    if len(parts) != 2:
        raise SystemExit("--clip-percentiles must be like '5,95'")
    p_low = float(parts[0])
    p_high = float(parts[1])

    norm_method = str(args.normalization)
    if norm_method == "pclip_minmax":
        labels, norm_meta = _normalize_pclip_minmax(composite_raw, p_low, p_high)
    elif norm_method == "minmax":
        labels, norm_meta = _normalize_minmax(composite_raw)
    else:  # ratio
        if gap_metric != "ppl_gap":
            raise SystemExit("--normalization ratio is only supported when --gap-metric=ppl_gap")
        labels, norm_meta = _normalize_ratio(composite_raw, dense_ppl)

    norm_meta.update({"gap_metric": gap_metric, "allow_negative_gaps": bool(args.allow_negative_gaps)})

    # ---- Write CSV ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    out_fieldnames: List[str] = [str(args.text_column)]
    out_fieldnames.extend(carried)

    if multi_method:
        out_fieldnames.extend(["ppl_dense", "loss_dense"])
        for mk in method_keys:
            out_fieldnames.append(f"ppl_sparse_{mk}")
            out_fieldnames.append(f"loss_sparse_{mk}")
            # Back-compat: gap_* remains the PPL gap.
            out_fieldnames.append(f"gap_{mk}")
            out_fieldnames.append(f"loss_gap_{mk}")
        out_fieldnames.extend([
            "raw_composite",
            "raw_composite_ppl",
            "raw_composite_loss",
            "normalized_sensitivity",
            "sensitivity_score",
        ])
    else:
        # Legacy single-method output
        out_fieldnames.extend([
            "ppl_dense",
            "loss_dense",
            "ppl_sparse",
            "loss_sparse",
            "raw_gap",
            "loss_gap",
            "raw_metric",
            "normalized_sensitivity",
            "sensitivity_score",
        ])

    with open(args.output, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()

        for i, (row, ppl_d, loss_d, y) in enumerate(zip(rows, dense_ppl, dense_loss, labels)):
            out_row: Dict[str, str] = {
                str(args.text_column): row.prompt,
                "ppl_dense": f"{float(ppl_d):.6f}",
                "loss_dense": f"{float(loss_d):.6f}",
                "normalized_sensitivity": f"{float(y):.6f}",
                "sensitivity_score": f"{float(y):.6f}",
            }

            if multi_method:
                out_row["raw_composite"] = f"{composite_raw[i]:.6f}"
                out_row["raw_composite_ppl"] = f"{composite_raw_ppl[i]:.6f}"
                out_row["raw_composite_loss"] = f"{composite_raw_loss[i]:.6f}"
                for mk in method_keys:
                    out_row[f"ppl_sparse_{mk}"] = f"{sparse_ppls_by_method[mk][i]:.6f}"
                    out_row[f"loss_sparse_{mk}"] = f"{sparse_losses_by_method[mk][i]:.6f}"
                    out_row[f"gap_{mk}"] = f"{gaps_by_method[mk][i]:.6f}"
                    out_row[f"loss_gap_{mk}"] = f"{loss_gaps_by_method[mk][i]:.6f}"
            else:
                mk0 = method_keys[0]
                out_row["ppl_sparse"] = f"{sparse_ppls_by_method[mk0][i]:.6f}"
                out_row["raw_gap"] = f"{gaps_by_method[mk0][i]:.6f}"
                out_row["loss_sparse"] = f"{sparse_losses_by_method[mk0][i]:.6f}"
                out_row["loss_gap"] = f"{loss_gaps_by_method[mk0][i]:.6f}"
                # The raw value actually normalized into the [0,1] label.
                out_row["raw_metric"] = f"{composite_raw[i]:.6f}"

            for k, v in row.meta.items():
                out_row[k] = v
            writer.writerow(out_row)

    # ---- Write JSON metadata ----
    elapsed_total = time.time() - t_start
    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": n_prompts,
        "text_column": str(args.text_column),
        "carry_columns": list(carried),
        "backbone_model": str(args.backbone_model),
        "multi_method": multi_method,
        "gap_metric": gap_metric,
        "allow_negative_gaps": bool(args.allow_negative_gaps),
        "sparse_configs": [
            {"target": c.target, "intensity": c.intensity, "weight": w}
            for c, w in zip(sparse_configs, weights)
        ],
        "max_length": int(args.max_length),
        "normalization": norm_meta,
        "cross_method_correlation": cross_corr,
        "elapsed_seconds": round(elapsed_total, 1),
        "samples_per_second": round(n_prompts / max(1, elapsed_total), 2),
    }

    # Always write meta json (auto-generate path if not provided)
    meta_json_path = str(args.meta_json).strip()
    if not meta_json_path:
        meta_json_path = str(args.output).rsplit(".", 1)[0] + ".meta.json"

    os.makedirs(os.path.dirname(meta_json_path) or ".", exist_ok=True)
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[Oracle] Wrote: {args.output}")
    print(f"[Oracle] Meta:  {meta_json_path}")
    print(f"[Oracle] Normalization: {norm_meta}")
    if cross_corr:
        print(f"[Oracle] Cross-method correlations: {cross_corr}")
    print(f"[Oracle] Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")


if __name__ == "__main__":
    main()
