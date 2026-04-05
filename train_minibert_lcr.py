"""Fine-tune BERT-mini for the Learned Complexity Router (LCR) — Multi-Output v2.

This follows the thesis pipeline:
- Text inputs + auxiliary statistical features
- Multi-output labels: sensitivity_head, sensitivity_layer (independently normalized [0,1])
- BERT-mini encoder (4 layers, 256 hidden) + aux projector + 3-layer dual-output head
- Differential learning rates: head lr=3e-5, backbone lr=factor*lr
- Linear warmup (10%) + cosine decay scheduler
- Mean pooling, source-balanced oversampling, label smoothing, gradient clipping
- Per-output metrics: R², Spearman for head and layer independently

Backward compatible: supports single-output mode with --label-columns "normalized_sensitivity"

Usage (dual-output):
  python train_minibert_lcr.py \\
    --data oracle_lcr_10k_dual.csv \\
    --label-columns "sensitivity_head,sensitivity_layer" \\
    --aux-features "log_token_count,compression_ratio,avg_word_length,special_char_ratio,unique_token_ratio,has_code_markers" \\
    --epochs 30 --patience 10 --dropout 0.20 --backbone-lr-factor 0.08 \\
    --label-smooth 0.015 --balance-sources

Usage (raw dataset + separate oracle labels file):
    python train_minibert_lcr.py \
        --data Oracle_dataset.csv \
        --labels-file oracle_lcr_labels.csv \
        --label-columns "normalized_sensitivity" \
        --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch

from lcr_minibert import (
    MiniBertLcrConfig, MiniBertLcrScorer,
    _RegressorHead, _AuxProjector, _ScalarMix, _AttentionStatsExtractor,
    compute_aux_features, AUX_FEATURE_DIM, AUX_FEATURE_NAMES,
)

try:
    from scipy.stats import spearmanr

    _SCIPY_AVAILABLE = True
except Exception:
    spearmanr = None
    _SCIPY_AVAILABLE = False


@dataclass
class Sample:
    text: str
    y: List[float]  # [y_head, y_layer] or [y_composite] for single-output
    source: str
    aux_features: Optional[List[float]] = None  # precomputed auxiliary features
    dataset_split: Optional[str] = None  # "train" or "test" from Split column


@dataclass(frozen=True)
class SplitResult:
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]
    by_source: Dict[str, Dict[str, int]]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def oversample_to_balance(samples: List[Sample], seed: int) -> List[Sample]:
    """Oversample minority sources so every source has as many training samples
    as the majority source."""
    if not samples:
        return samples
    by_src: Dict[str, List[Sample]] = {}
    for s in samples:
        by_src.setdefault(s.source, []).append(s)
    max_count = max(len(v) for v in by_src.values())
    rng = random.Random(seed)
    balanced: List[Sample] = []
    for src, group in by_src.items():
        if len(group) >= max_count:
            balanced.extend(group)
        else:
            extras_needed = max_count - len(group)
            extras = [rng.choice(group) for _ in range(extras_needed)]
            balanced.extend(group)
            balanced.extend(extras)
    rng.shuffle(balanced)
    return balanced


def _label_smooth(y: float, amount: float, rng: random.Random) -> float:
    if amount <= 0:
        return y
    noise = rng.uniform(-amount, amount)
    return _clamp01(y + noise)


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _next_run_dir(base_dir: str, prefix: str) -> str:
    _safe_makedirs(base_dir)
    max_idx = 0
    for name in os.listdir(base_dir):
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):].strip()
        try:
            idx = int(rest)
        except Exception:
            continue
        max_idx = max(max_idx, idx)
    run_dir = os.path.join(base_dir, f"{prefix} {max_idx + 1}")
    _safe_makedirs(run_dir)
    return run_dir


def _resolve_existing_column(fieldnames: Sequence[str], preferred: str, fallbacks: Sequence[str]) -> str:
    if preferred in fieldnames:
        return preferred
    for c in fallbacks:
        if c in fieldnames:
            return c
    raise ValueError(f"None of the columns exist: preferred='{preferred}', fallbacks={list(fallbacks)}")


def _read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        return list(reader), fieldnames


def _build_join_key(row: Dict[str, str], join_columns: Sequence[str]) -> Tuple[str, ...]:
    return tuple((row.get(c) or "").strip() for c in join_columns)


def _format_join_key(join_columns: Sequence[str], key: Sequence[str]) -> str:
    return ", ".join(f"{col}={val!r}" for col, val in zip(join_columns, key))


def _resolve_join_columns(
    data_fields: Sequence[str],
    label_fields: Sequence[str],
    join_columns: Sequence[str],
    text_column: str,
) -> List[str]:
    requested = [c.strip() for c in join_columns if c and c.strip()]
    shared = [c for c in requested if c in data_fields and c in label_fields]
    if shared:
        return shared
    if text_column in data_fields and text_column in label_fields:
        return [text_column]
    raise ValueError(
        "Could not resolve shared join columns between data CSV and labels CSV. "
        f"Requested={requested}, data_fields={list(data_fields)}, label_fields={list(label_fields)}"
    )


def load_oracle_csv(
    path: str,
    text_column: str,
    label_columns: List[str],
    source_column: str,
    aux_feature_columns: List[str],
    split_column: str = "Split",
    labels_path: str = "",
    join_columns: Optional[List[str]] = None,
) -> Tuple[List[Sample], Dict[str, str]]:
    """Load samples with potentially multiple label columns and auxiliary features."""
    out: List[Sample] = []
    data_rows, fieldnames = _read_csv_rows(path)
    if text_column not in fieldnames:
        raise ValueError(f"CSV column '{text_column}' not found. Columns: {fieldnames}")

    label_rows = data_rows
    label_fieldnames = fieldnames
    resolved_join_columns: List[str] = []
    labels_map: Dict[Tuple[str, ...], Dict[str, str]] = {}

    if labels_path:
        label_rows, label_fieldnames = _read_csv_rows(labels_path)
        resolved_join_columns = _resolve_join_columns(
            fieldnames,
            label_fieldnames,
            join_columns or [text_column],
            text_column,
        )
        duplicate_keys: List[Tuple[str, ...]] = []
        for label_row in label_rows:
            key = _build_join_key(label_row, resolved_join_columns)
            if key in labels_map:
                duplicate_keys.append(key)
                continue
            labels_map[key] = label_row
        if duplicate_keys:
            print(
                f"[LCR-v2] WARNING: {len(duplicate_keys)} duplicate keys in labels CSV (kept first occurrence). "
                f"Example: {_format_join_key(resolved_join_columns, duplicate_keys[0])}"
            )

    # Resolve label columns from the source that actually contains them.
    resolved_labels: List[str] = []
    for lc in label_columns:
        resolved = _resolve_existing_column(
            label_fieldnames, preferred=lc,
            fallbacks=["normalized_sensitivity", "sensitivity_score"],
        )
        resolved_labels.append(resolved)

    used_source: Optional[str] = None
    if source_column in fieldnames:
        used_source = source_column
    elif labels_path and source_column in label_fieldnames:
        used_source = source_column
    elif "SourceDataset" in fieldnames:
        used_source = "SourceDataset"
    elif labels_path and "SourceDataset" in label_fieldnames:
        used_source = "SourceDataset"

    has_split_col = split_column in fieldnames or (labels_path and split_column in label_fieldnames)

    resolved_aux: List[str] = []
    for c in aux_feature_columns:
        if c in fieldnames or (labels_path and c in label_fieldnames):
            resolved_aux.append(c)
    has_aux = len(resolved_aux) > 0

    missing_label_keys: List[Tuple[str, ...]] = []
    for row in data_rows:
        t = (row.get(text_column) or "").strip()
        if not t:
            continue

        label_row = row
        if labels_path:
            key = _build_join_key(row, resolved_join_columns)
            label_row = labels_map.get(key)
            if label_row is None:
                missing_label_keys.append(key)
                continue

        y_vals: List[float] = []
        valid = True
        for lc in resolved_labels:
            try:
                y_vals.append(_clamp01(float(label_row.get(lc))))
            except Exception:
                valid = False
                break
        if not valid:
            continue

        src = None
        if used_source:
            if used_source in fieldnames:
                src = row.get(used_source)
            elif used_source in label_fieldnames:
                src = label_row.get(used_source)
        src_s = (src or "unknown").strip() or "unknown"

        aux: Optional[List[float]] = None
        if has_aux:
            try:
                csv_aux: List[float] = []
                for c in resolved_aux:
                    if c in fieldnames:
                        csv_aux.append(float(row.get(c, 0.0)))
                    else:
                        csv_aux.append(float(label_row.get(c, 0.0)))
                if len(csv_aux) == len(aux_feature_columns):
                    aux = csv_aux
                else:
                    aux = compute_aux_features(t)
            except Exception:
                aux = compute_aux_features(t)
        else:
            aux = compute_aux_features(t) if aux_feature_columns else None

        ds_split = None
        if has_split_col:
            if split_column in fieldnames:
                ds_split = (row.get(split_column) or "").strip().lower()
            elif split_column in label_fieldnames:
                ds_split = (label_row.get(split_column) or "").strip().lower()
        out.append(Sample(text=t, y=y_vals, source=src_s, aux_features=aux, dataset_split=ds_split))

    if missing_label_keys:
        raise ValueError(
            f"Failed to match {len(missing_label_keys)} data rows to labels in '{labels_path}'. "
            f"Join columns={resolved_join_columns}. Example missing key: "
            f"{_format_join_key(resolved_join_columns, missing_label_keys[0])}"
        )

    meta = {
        "used_label_columns": resolved_labels,
        "labels_source": labels_path or path,
        "used_source_column": used_source or "(none)",
        "used_aux_features": resolved_aux if resolved_aux else ("computed" if aux_feature_columns else "none"),
        "used_join_columns": resolved_join_columns if labels_path else [text_column],
        "n_outputs": len(resolved_labels),
        "has_split_column": has_split_col,
    }
    return out, meta


def stratified_split(
    samples: List[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitResult:
    if not samples:
        return SplitResult(train=[], val=[], test=[], by_source={})
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(int(seed))
    by_src: Dict[str, List[Sample]] = {}
    for s in samples:
        by_src.setdefault(s.source, []).append(s)

    train_all, val_all, test_all = [], [], []
    by_source_stats: Dict[str, Dict[str, int]] = {}

    for src, group in sorted(by_src.items()):
        rng.shuffle(group)
        n_total = len(group)
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = max(1, int(round(n_total * test_ratio)))
        n_train = n_total - n_val - n_test
        if n_train < 1:
            n_train, n_val = 1, max(1, n_total - 2)
            n_test = n_total - n_train - n_val
        train_g = group[:n_train]
        val_g = group[n_train: n_train + n_val]
        test_g = group[n_train + n_val:]
        train_all.extend(train_g)
        val_all.extend(val_g)
        test_all.extend(test_g)
        by_source_stats[src] = {
            "train": len(train_g), "val": len(val_g),
            "test": len(test_g), "total": n_total,
        }

    rng.shuffle(train_all)
    return SplitResult(train=train_all, val=val_all, test=test_all, by_source=by_source_stats)


@torch.no_grad()
def _predict(
    backbone, head, tok, aux_proj, samples, batch_size, device, max_length, pooling, n_outputs, use_aux,
    scalar_mix=None, attn_extractor=None,
) -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """Predict labels for a list of samples. Returns per-output true/pred lists."""
    if not samples:
        return [], [], []

    head.eval()
    backbone.eval()
    if aux_proj is not None:
        aux_proj.eval()
    if scalar_mix is not None:
        scalar_mix.eval()

    y_true: List[List[float]] = [[] for _ in range(n_outputs)]
    y_pred: List[List[float]] = [[] for _ in range(n_outputs)]
    sources: List[str] = []

    for i in range(0, len(samples), int(batch_size)):
        batch = samples[i: i + int(batch_size)]
        texts = [s.text for s in batch]

        enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = backbone(**enc, return_dict=True, output_hidden_states=True, output_attentions=True)

        # Multi-layer pooling via ScalarMix (or fall back to last_hidden_state)
        if scalar_mix is not None:
            mixed_hidden = scalar_mix(out.hidden_states)
        else:
            mixed_hidden = out.last_hidden_state

        attention_mask = enc.get("attention_mask")
        if pooling == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(mixed_hidden).float()
                pooled = (mixed_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = mixed_hidden.mean(dim=1)
        else:
            pooled = mixed_hidden[:, 0, :]

        # Attention entropy features + text aux features → projector
        if aux_proj is not None:
            attn_stats = attn_extractor(out.attentions, attention_mask) if attn_extractor is not None else None
            if use_aux:
                aux_feats = []
                for s in batch:
                    af = s.aux_features if s.aux_features else compute_aux_features(s.text)
                    aux_feats.append(af)
                aux_t = torch.tensor(aux_feats, dtype=torch.float32, device=device)
                if attn_stats is not None:
                    combined_aux = torch.cat([aux_t, attn_stats], dim=-1)
                else:
                    combined_aux = aux_t
            else:
                combined_aux = attn_stats if attn_stats is not None else torch.zeros(len(batch), 0, device=device)
            aux_emb = aux_proj(combined_aux)
            pooled = torch.cat([pooled, aux_emb], dim=-1)

        pred = head(pooled)  # [B, n_outputs]
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)

        for out_idx in range(n_outputs):
            for j, s in enumerate(batch):
                y_true[out_idx].append(float(s.y[out_idx]) if out_idx < len(s.y) else float(s.y[0]))
                y_pred[out_idx].append(float(pred[j, out_idx].item()) if out_idx < pred.shape[1] else float(pred[j, 0].item()))

        sources.extend([s.source for s in batch])

    return y_true, y_pred, sources


def _compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0, "spearman": 0.0, "bin3_acc": 0.0}
    yt = [float(x) for x in y_true[:n]]
    yp = [float(x) for x in y_pred[:n]]

    mse = sum((a - b) ** 2 for a, b in zip(yt, yp)) / float(n)
    mae = sum(abs(a - b) for a, b in zip(yt, yp)) / float(n)

    mean_y = sum(yt) / float(n)
    ss_res = sum((a - b) ** 2 for a, b in zip(yt, yp))
    ss_tot = sum((a - mean_y) ** 2 for a in yt)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    sp = 0.0
    if _SCIPY_AVAILABLE and spearmanr is not None and n >= 2:
        try:
            sp_val = spearmanr(yt, yp).correlation
            if sp_val is not None and sp_val == sp_val:
                sp = float(sp_val)
        except Exception:
            sp = 0.0

    # Binned accuracy: bucket into Low [0, 0.33), Mid [0.33, 0.67), High [0.67, 1.0]
    # This shows the model's ability to correctly classify pruning risk tier
    correct = 0
    for a, b in zip(yt, yp):
        true_bin = 0 if a < 0.33 else (1 if a < 0.67 else 2)
        pred_bin = 0 if b < 0.33 else (1 if b < 0.67 else 2)
        if true_bin == pred_bin:
            correct += 1
    bin3_acc = float(correct) / float(n)

    return {"mse": float(mse), "mae": float(mae), "r2": float(r2),
            "spearman": float(sp), "bin3_acc": float(bin3_acc)}


def _metrics_by_source(
    y_true: Sequence[float], y_pred: Sequence[float], sources: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    idxs: Dict[str, List[int]] = {}
    for i, s in enumerate(sources):
        idxs.setdefault(s, []).append(i)
    out: Dict[str, Dict[str, float]] = {}
    for src, ii in idxs.items():
        yt = [float(y_true[i]) for i in ii]
        yp = [float(y_pred[i]) for i in ii]
        out[src] = _compute_metrics(yt, yp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Oracle CSV (from prepare_dual_labels.py)")
    parser.add_argument("--labels-file", default="",
                        help="Optional separate CSV containing oracle labels to join onto --data.")
    parser.add_argument("--text-column", default="Prompt")
    # Multi-output: comma-separated label columns
    parser.add_argument("--label-columns", default="sensitivity_head,sensitivity_layer",
                        help="Comma-separated label columns. 2 columns = dual-output, 1 = single-output.")
    parser.add_argument("--source-column", default="SourceDataset")
    parser.add_argument(
        "--join-columns",
        default="SourceDataset,SourceSplit,SourceId,Category,Subject,Split,Prompt",
        help="Comma-separated columns used to join --data with --labels-file.",
    )
    # Auxiliary features
    parser.add_argument("--aux-features", default=",".join(AUX_FEATURE_NAMES),
                        help="Comma-separated aux feature column names (or empty to disable).")
    parser.add_argument("--no-aux-features", dest="aux_features", action="store_const", const="")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.08,
                        help="Backbone LR = lr * this factor.")
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model-name", default="prajjwal1/bert-mini")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--warmup-ratio", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--balance-sources", action="store_true", default=True)
    parser.add_argument("--no-balance-sources", dest="balance_sources", action="store_false")
    parser.add_argument("--label-smooth", type=float, default=0.01)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse",
                        help="Loss function. 'huber' (SmoothL1) is more robust to label noise.")
    parser.add_argument("--huber-delta", type=float, default=0.15,
                        help="Delta for Huber/SmoothL1 loss (only used when --loss huber).")
    parser.add_argument("--rank-lambda", type=float, default=0.0,
                        help="Weight for pairwise ranking loss (0 = disabled). Adds a margin-based "
                             "ranking loss that directly optimizes for Spearman correlation.")

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--split-column", default="Split",
                        help="CSV column marking train/test split. When present, uses dataset split "
                             "instead of random splitting. Set to '' to disable.")

    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default="",
                        help="Convenience alias to export runtime artifacts into this directory.")
    parser.add_argument("--export-head", default=os.path.join("checkpoints", "minibert_lcr_head.pt"))
    parser.add_argument("--export-backbone-dir", default=os.path.join("checkpoints", "minibert_lcr_backbone"))
    args = parser.parse_args()

    t0 = time.time()

    # Parse label columns
    label_columns = [c.strip() for c in str(args.label_columns).split(",") if c.strip()]
    n_outputs = len(label_columns)
    output_names = label_columns if n_outputs > 1 else ["composite"]
    print(f"[LCR-v2] Mode: {'dual-output' if n_outputs >= 2 else 'single-output'} ({output_names})")

    # Parse aux feature columns
    aux_feature_cols = [c.strip() for c in str(args.aux_features).split(",") if c.strip()]
    use_aux = len(aux_feature_cols) > 0
    print(f"[LCR-v2] Auxiliary features: {'ON' if use_aux else 'OFF'} ({len(aux_feature_cols)} features)")

    join_columns = [c.strip() for c in str(args.join_columns).split(",") if c.strip()]
    labels_path = str(args.labels_file).strip()

    output_dir = str(args.output_dir).strip()
    if output_dir:
        args.export_head = os.path.join(output_dir, "minibert_lcr_head.pt")
        args.export_backbone_dir = os.path.join(output_dir, "minibert_lcr_backbone")

    # Load data
    split_col = str(args.split_column).strip() if hasattr(args, 'split_column') else "Split"
    samples, load_meta = load_oracle_csv(
        args.data,
        text_column=str(args.text_column),
        label_columns=label_columns,
        source_column=str(args.source_column),
        aux_feature_columns=aux_feature_cols,
        split_column=split_col,
        labels_path=labels_path,
        join_columns=join_columns,
    )
    if not samples:
        raise SystemExit("No samples loaded from oracle CSV.")

    print(f"[LCR-v2] Loaded {len(samples)} samples from {args.data}")
    if labels_path:
        print(f"[LCR-v2] Joined labels from {labels_path} using {load_meta.get('used_join_columns')}")

    # Splitting: use dataset Split column if present, otherwise random stratified split
    use_dataset_split = bool(load_meta.get("has_split_column")) and split_col and any(
        s.dataset_split in ("train", "test") for s in samples
    )

    if use_dataset_split:
        print(f"[LCR-v2] Using dataset '{split_col}' column for train/test assignment")
        ds_train = [s for s in samples if s.dataset_split == "train"]
        ds_test = [s for s in samples if s.dataset_split == "test"]
        # Further split ds_train into train/val (90/10) for early stopping
        rng_split = random.Random(int(args.seed))
        by_src_train: Dict[str, List[Sample]] = {}
        for s in ds_train:
            by_src_train.setdefault(s.source, []).append(s)
        train_final, val_final = [], []
        by_source_stats: Dict[str, Dict[str, int]] = {}
        for src, group in sorted(by_src_train.items()):
            rng_split.shuffle(group)
            n_val = max(1, int(round(len(group) * 0.1)))
            n_tr = len(group) - n_val
            train_final.extend(group[:n_tr])
            val_final.extend(group[n_tr:])
            by_source_stats[src] = {
                "train": n_tr, "val": n_val, "test": len([s for s in ds_test if s.source == src]),
                "total": len(group) + len([s for s in ds_test if s.source == src]),
            }
        rng_split.shuffle(train_final)
        split = SplitResult(train=train_final, val=val_final, test=ds_test, by_source=by_source_stats)
        print(f"[LCR-v2] Dataset split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    else:
        # Stratified split (legacy behavior)
        split = stratified_split(
            samples, train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio),
            seed=int(args.seed),
        )
        print(f"[LCR-v2] Random split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    # Source balancing
    if bool(args.balance_sources):
        original_train_len = len(split.train)
        balanced_train = oversample_to_balance(split.train, seed=int(args.seed))
        src_counts_before: Dict[str, int] = {}
        src_counts_after: Dict[str, int] = {}
        for s in split.train:
            src_counts_before[s.source] = src_counts_before.get(s.source, 0) + 1
        for s in balanced_train:
            src_counts_after[s.source] = src_counts_after.get(s.source, 0) + 1
        print(f"[LCR-v2] Source balancing: {original_train_len} → {len(balanced_train)} train samples")
        for src in sorted(src_counts_before.keys()):
            print(f"  {src}: {src_counts_before[src]} → {src_counts_after.get(src, 0)}")
        split = SplitResult(train=balanced_train, val=split.val, test=split.test, by_source=split.by_source)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    run_dir = str(args.run_dir).strip() or _next_run_dir(os.path.join("Training Report"), "MiniBERT Train")

    # Initialize model components
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_name), use_fast=True)
    try:
        backbone = AutoModel.from_pretrained(str(args.model_name), use_safetensors=True)
    except Exception:
        backbone = AutoModel.from_pretrained(str(args.model_name))
    backbone.to(device)

    hidden_size = int(getattr(backbone.config, "hidden_size", 256))
    n_bert_layers = int(getattr(backbone.config, "num_hidden_layers", 4))
    n_bert_heads = int(getattr(backbone.config, "num_attention_heads", 4))

    # Multi-layer pooling: learned weighted sum over all hidden layers
    scalar_mix = _ScalarMix(n_layers=n_bert_layers + 1)  # +1 for embedding layer
    scalar_mix.to(device)

    # Attention entropy features: model-internal complexity signals
    attn_extractor = _AttentionStatsExtractor(n_layers=n_bert_layers, n_heads=n_bert_heads)
    attn_extractor.to(device)
    attn_stats_dim = attn_extractor.out_dim  # n_layers * n_heads = 16 for bert-mini

    # Auxiliary projector (text features + attention entropy features)
    aux_proj = None
    total_aux_dim = 0
    if use_aux:
        total_aux_dim += AUX_FEATURE_DIM
    total_aux_dim += attn_stats_dim  # always use attention stats
    aux_proj = _AuxProjector(total_aux_dim, out_dim=48)
    aux_proj.to(device)
    input_dim = hidden_size + 48

    # Regression head
    head = _RegressorHead(
        input_dim=input_dim,
        dropout=float(args.dropout),
        n_outputs=n_outputs,
    )
    head.to(device)

    if bool(args.freeze_backbone):
        for p in backbone.parameters():
            p.requires_grad = False

    # Parameter groups with differential LR
    head_params = list(head.parameters()) + list(scalar_mix.parameters()) + list(attn_extractor.parameters())
    if aux_proj is not None:
        head_params += list(aux_proj.parameters())

    backbone_lr = float(args.lr) * float(args.backbone_lr_factor)
    if bool(args.freeze_backbone):
        param_groups = [{"params": head_params, "lr": float(args.lr)}]
    else:
        param_groups = [
            {"params": head_params, "lr": float(args.lr)},
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": backbone_lr},
        ]
    opt = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

    trainable = [p for pg in param_groups for p in pg["params"] if p.requires_grad]

    # Scheduler
    steps_per_epoch = max(1, (len(split.train) + int(args.batch_size) - 1) // int(args.batch_size))
    total_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = int(total_steps * float(args.warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    # Loss function selection
    if str(args.loss) == "huber":
        loss_fn = torch.nn.SmoothL1Loss(beta=float(args.huber_delta))
        print(f"[LCR-v2] Loss: Huber (SmoothL1, delta={args.huber_delta})")
    else:
        loss_fn = torch.nn.MSELoss()
        print(f"[LCR-v2] Loss: MSE")

    rank_lambda = float(args.rank_lambda)
    if rank_lambda > 0:
        print(f"[LCR-v2] Pairwise ranking loss: λ={rank_lambda}")

    print(f"[LCR-v2] head_lr={args.lr:.1e} backbone_lr={backbone_lr:.1e} warmup={warmup_steps} total={total_steps}")
    print(f"[LCR-v2] pooling={args.pooling} dropout={args.dropout} grad_clip={args.max_grad_norm}")
    print(f"[LCR-v2] hidden_size={hidden_size} input_dim={input_dim} n_outputs={n_outputs}")
    print(f"[LCR-v2] multi-layer ScalarMix over {n_bert_layers+1} hidden states")
    print(f"[LCR-v2] attention entropy: {attn_stats_dim} features ({n_bert_layers} layers × {n_bert_heads} heads)")

    best = {
        "objective": -1e18, "epoch": 0, "val": {},
        "head_state": None, "backbone_state": None, "aux_state": None, "scalar_mix_state": None,
    }
    patience_left = int(args.patience)
    history: List[Dict[str, float]] = []

    # ---- Training loop ----
    for epoch in range(1, int(args.epochs) + 1):
        rng = random.Random(int(args.seed) + epoch)
        rng.shuffle(split.train)

        head.train()
        backbone.train(mode=not bool(args.freeze_backbone))
        scalar_mix.train()
        if aux_proj is not None:
            aux_proj.train()

        total_loss = 0.0
        seen = 0
        label_smooth_amt = float(args.label_smooth)

        for i in range(0, len(split.train), int(args.batch_size)):
            batch = split.train[i: i + int(args.batch_size)]
            texts = [s.text for s in batch]

            # Build target tensor [B, n_outputs]
            y_list = []
            for s in batch:
                y_vals = [_label_smooth(s.y[oi], label_smooth_amt, rng) for oi in range(n_outputs)]
                y_list.append(y_vals)
            y = torch.tensor(y_list, dtype=torch.float32, device=device)  # [B, n_outputs]

            enc = tok(texts, return_tensors="pt", truncation=True, max_length=int(args.max_length), padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}

            out = backbone(**enc, return_dict=True, output_hidden_states=True, output_attentions=True)

            # Multi-layer pooling via ScalarMix
            mixed_hidden = scalar_mix(out.hidden_states)  # [B, seq, H]
            attention_mask = enc.get("attention_mask")
            if str(args.pooling) == "mean":
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(mixed_hidden).float()
                    pooled = (mixed_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                else:
                    pooled = mixed_hidden.mean(dim=1)
            else:
                pooled = mixed_hidden[:, 0, :]

            # Attention entropy features + text aux features → projector
            attn_stats = attn_extractor(out.attentions, attention_mask)  # [B, 16]
            if use_aux:
                aux_feats = []
                for s in batch:
                    af = s.aux_features if s.aux_features else compute_aux_features(s.text)
                    aux_feats.append(af)
                aux_t = torch.tensor(aux_feats, dtype=torch.float32, device=device)
                combined_aux = torch.cat([aux_t, attn_stats], dim=-1)  # [B, 9+16=25]
            else:
                combined_aux = attn_stats  # [B, 16]
            aux_emb = aux_proj(combined_aux)  # [B, 48]
            pooled = torch.cat([pooled, aux_emb], dim=-1)  # [B, H+48]

            pred = head(pooled)  # [B, n_outputs]
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            loss = loss_fn(pred, y)

            # Pairwise ranking loss: for each pair (i,j) where y_i > y_j,
            # encourage pred_i > pred_j with a margin proportional to label gap.
            if rank_lambda > 0 and pred.shape[0] >= 4:
                p_flat = pred[:, 0]  # [B]
                y_flat = y[:, 0]     # [B]
                # Sample random pairs (efficient: B//2 pairs)
                n_pairs = pred.shape[0] // 2
                idx_a = torch.arange(0, n_pairs * 2, 2, device=device)
                idx_b = torch.arange(1, n_pairs * 2, 2, device=device)
                # Margin = |y_a - y_b|, sign = sign(y_a - y_b)
                y_diff = y_flat[idx_a] - y_flat[idx_b]
                p_diff = p_flat[idx_a] - p_flat[idx_b]
                target_sign = torch.sign(y_diff)
                margin = torch.abs(y_diff).clamp(min=0.02)
                # MarginRankingLoss: max(0, -target_sign * p_diff + margin)
                rank_loss = torch.clamp(-target_sign * p_diff + margin, min=0.0).mean()
                loss = loss + rank_lambda * rank_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.max_grad_norm))
            opt.step()
            scheduler.step()

            total_loss += float(loss.detach().cpu().item()) * len(batch)
            seen += len(batch)

        train_mse = total_loss / max(1, seen)

        # ---- Validation ----
        yv_true, yv_pred, yv_src = _predict(
            backbone, head, tok, aux_proj, split.val,
            batch_size=int(args.batch_size), device=device,
            max_length=int(args.max_length), pooling=str(args.pooling),
            n_outputs=n_outputs, use_aux=use_aux,
            scalar_mix=scalar_mix, attn_extractor=attn_extractor,
        )

        # Compute per-output metrics
        per_output_metrics: List[Dict[str, float]] = []
        per_output_by_src: List[Dict[str, Dict[str, float]]] = []
        for oi in range(n_outputs):
            m = _compute_metrics(yv_true[oi], yv_pred[oi])
            per_output_metrics.append(m)
            per_output_by_src.append(_metrics_by_source(yv_true[oi], yv_pred[oi], yv_src))

        # Combined metrics (average across outputs)
        avg_mse = sum(m["mse"] for m in per_output_metrics) / n_outputs
        avg_spearman = sum(m["spearman"] for m in per_output_metrics) / n_outputs
        avg_r2 = sum(m["r2"] for m in per_output_metrics) / n_outputs
        avg_bin3 = sum(m["bin3_acc"] for m in per_output_metrics) / n_outputs
        # Objective: prioritize ranking (Spearman) and explained variance (R²)
        # over raw MSE. Spearman matters most for RL integration (ranking prompts).
        objective = 0.4 * avg_r2 + 0.5 * avg_spearman + 0.1 * avg_bin3

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_mse": float(train_mse),
            "val_mse_avg": float(avg_mse),
            "val_r2_avg": float(avg_r2),
            "val_spearman_avg": float(avg_spearman),
            "val_bin3_avg": float(avg_bin3),
            "objective": float(objective),
        }
        for oi, name in enumerate(output_names):
            row[f"val_mse_{name}"] = float(per_output_metrics[oi]["mse"])
            row[f"val_r2_{name}"] = float(per_output_metrics[oi]["r2"])
            row[f"val_spearman_{name}"] = float(per_output_metrics[oi]["spearman"])
            row[f"val_bin3_{name}"] = float(per_output_metrics[oi]["bin3_acc"])
        history.append(row)

        per_out_str = " | ".join(
            f"{name}: r2={per_output_metrics[oi]['r2']:.4f} ρ={per_output_metrics[oi]['spearman']:.4f} bin3={per_output_metrics[oi]['bin3_acc']:.3f}"
            for oi, name in enumerate(output_names)
        )
        print(
            f"[LCR-v2] epoch={epoch} train_mse={train_mse:.6f} "
            f"val_mse={avg_mse:.6f} val_r2={avg_r2:.4f} val_ρ={avg_spearman:.4f} | {per_out_str}"
        )

        improved = objective > float(best["objective"]) + 1e-9
        if improved:
            best["objective"] = float(objective)
            best["epoch"] = int(epoch)
            best["val"] = {
                "avg": {"mse": avg_mse, "r2": avg_r2, "spearman": avg_spearman},
                "per_output": [
                    {"name": output_names[oi], **per_output_metrics[oi], "by_source": per_output_by_src[oi]}
                    for oi in range(n_outputs)
                ],
            }
            best["head_state"] = copy.deepcopy(head.state_dict())
            best["backbone_state"] = copy.deepcopy(backbone.state_dict())
            best["aux_state"] = copy.deepcopy(aux_proj.state_dict()) if aux_proj is not None else None
            best["scalar_mix_state"] = copy.deepcopy(scalar_mix.state_dict())
            patience_left = int(args.patience)
            print(f"[LCR-v2] ★ new best @ epoch {epoch} (objective={objective:.6f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[LCR-v2] early stopping")
                break

    # ---- Restore best and evaluate on test ----
    if best["head_state"] is None or best["backbone_state"] is None:
        raise SystemExit("Training did not produce a best checkpoint.")
    head.load_state_dict(best["head_state"], strict=True)
    backbone.load_state_dict(best["backbone_state"], strict=True)
    if aux_proj is not None and best.get("aux_state") is not None:
        aux_proj.load_state_dict(best["aux_state"], strict=True)
    if best.get("scalar_mix_state") is not None:
        scalar_mix.load_state_dict(best["scalar_mix_state"], strict=True)

    yt_true, yt_pred, yt_src = _predict(
        backbone, head, tok, aux_proj, split.test,
        batch_size=int(args.batch_size), device=device,
        max_length=int(args.max_length), pooling=str(args.pooling),
        n_outputs=n_outputs, use_aux=use_aux,
        scalar_mix=scalar_mix, attn_extractor=attn_extractor,
    )

    test_per_output = []
    for oi in range(n_outputs):
        m = _compute_metrics(yt_true[oi], yt_pred[oi])
        bs = _metrics_by_source(yt_true[oi], yt_pred[oi], yt_src)
        # Bootstrap 95% CI for Spearman (1000 resamples)
        sp_ci = {"low": 0.0, "high": 0.0}
        if _SCIPY_AVAILABLE and spearmanr is not None and len(yt_true[oi]) >= 20:
            rng_boot = random.Random(42)
            boot_sp = []
            n_test = len(yt_true[oi])
            for _ in range(1000):
                idx = [rng_boot.randint(0, n_test - 1) for _ in range(n_test)]
                bt = [yt_true[oi][j] for j in idx]
                bp = [yt_pred[oi][j] for j in idx]
                try:
                    sv = spearmanr(bt, bp).correlation
                    if sv is not None and sv == sv:
                        boot_sp.append(float(sv))
                except Exception:
                    pass
            if boot_sp:
                boot_sp.sort()
                sp_ci["low"] = float(boot_sp[int(len(boot_sp) * 0.025)])
                sp_ci["high"] = float(boot_sp[int(len(boot_sp) * 0.975)])
        m["spearman_ci_low"] = sp_ci["low"]
        m["spearman_ci_high"] = sp_ci["high"]
        test_per_output.append({"name": output_names[oi], **m, "by_source": bs})

    test_avg_mse = sum(t["mse"] for t in test_per_output) / n_outputs
    test_avg_r2 = sum(t["r2"] for t in test_per_output) / n_outputs
    test_avg_spearman = sum(t["spearman"] for t in test_per_output) / n_outputs
    test_avg_bin3 = sum(t["bin3_acc"] for t in test_per_output) / n_outputs

    # ---- Save artifacts ----
    run_head_path = os.path.join(run_dir, "minibert_lcr_head.pt")
    run_backbone_dir = os.path.join(run_dir, "minibert_lcr_backbone")
    _safe_makedirs(run_dir)

    # Save head + aux_proj + scalar_mix + attn_extractor together in a single checkpoint
    save_state = {}
    for k, v in head.state_dict().items():
        save_state[f"head.{k}"] = v
    if aux_proj is not None:
        for k, v in aux_proj.state_dict().items():
            save_state[f"aux_proj.{k}"] = v
    for k, v in scalar_mix.state_dict().items():
        save_state[f"scalar_mix.{k}"] = v
    for k, v in attn_extractor.state_dict().items():
        save_state[f"attn_extractor.{k}"] = v
    torch.save(save_state, run_head_path)

    if os.path.exists(run_backbone_dir):
        shutil.rmtree(run_backbone_dir)
    backbone.save_pretrained(run_backbone_dir)
    tok.save_pretrained(run_backbone_dir)

    metrics_out = {
        "run_dir": run_dir,
        "data": str(args.data),
        "labels_file": labels_path or None,
        "load": load_meta,
        "splits": {
            "train": len(split.train), "val": len(split.val), "test": len(split.test),
            "by_source": split.by_source,
        },
        "config": {
            "model_name": str(args.model_name),
            "max_length": int(args.max_length),
            "dropout": float(args.dropout),
            "n_outputs": n_outputs,
            "output_names": output_names,
            "use_aux_features": use_aux,
            "hidden_size": hidden_size,
            "input_dim": input_dim,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "backbone_lr_factor": float(args.backbone_lr_factor),
            "weight_decay": float(args.weight_decay),
            "freeze_backbone": bool(args.freeze_backbone),
            "balance_sources": bool(args.balance_sources),
            "label_smooth": float(args.label_smooth),
            "seed": int(args.seed),
            "patience": int(args.patience),
            "objective": "0.4*r2 + 0.5*spearman + 0.1*bin3_acc",
            "loss": str(args.loss),
            "huber_delta": float(args.huber_delta) if str(args.loss) == "huber" else None,
        },
        "best_epoch": int(best["epoch"]),
        "best_val": best["val"],
        "test": {
            "avg": {"mse": test_avg_mse, "r2": test_avg_r2, "spearman": test_avg_spearman, "bin3_acc": test_avg_bin3},
            "per_output": test_per_output,
        },
        "history": history,
        "seconds": float(time.time() - t0),
    }

    with open(os.path.join(run_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write("MiniBERT LCR v2 — Multi-Output Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {args.data}\n")
        if labels_path:
            f.write(f"Labels file: {labels_path}\n")
            f.write(f"Join columns: {', '.join(load_meta.get('used_join_columns', []))}\n")
        f.write(f"Label columns: {', '.join(label_columns)}\n")
        f.write(f"Outputs: {n_outputs} ({', '.join(output_names)})\n")
        f.write(f"Aux features: {'ON' if use_aux else 'OFF'}\n\n")
        f.write(f"Train/Val/Test: {len(split.train)}/{len(split.val)}/{len(split.test)}\n")
        f.write("By source:\n")
        for src, c in split.by_source.items():
            f.write(f"  - {src}: {c}\n")
        f.write(f"\nBest epoch: {best['epoch']}\n\n")
        f.write("=== VALIDATION (best epoch) ===\n")
        f.write(f"  Average: mse={best['val']['avg']['mse']:.6f} r2={best['val']['avg']['r2']:.4f} spearman={best['val']['avg']['spearman']:.4f}\n")
        for po in best['val']['per_output']:
            f.write(f"  {po['name']}: mse={po['mse']:.6f} r2={po['r2']:.4f} spearman={po['spearman']:.4f} bin3={po['bin3_acc']:.4f}\n")
        f.write(f"\n=== TEST ===\n")
        f.write(f"  Average: mse={test_avg_mse:.6f} r2={test_avg_r2:.4f} spearman={test_avg_spearman:.4f} bin3_acc={test_avg_bin3:.4f}\n")
        for po in test_per_output:
            sp_ci_str = ""
            if "spearman_ci_low" in po:
                sp_ci_str = f" [95%CI: {po['spearman_ci_low']:.4f}–{po['spearman_ci_high']:.4f}]"
            f.write(f"  {po['name']}: mse={po['mse']:.6f} r2={po['r2']:.4f} spearman={po['spearman']:.4f}{sp_ci_str} bin3_acc={po['bin3_acc']:.4f}\n")
            f.write(f"    Per-source:\n")
            for src, sm in po['by_source'].items():
                f.write(f"      {src}: r2={sm['r2']:.4f} spearman={sm['spearman']:.4f} bin3={sm['bin3_acc']:.4f}\n")
        f.write(f"\nArtifacts:\n")
        f.write(f"  head: {run_head_path}\n")
        f.write(f"  backbone: {run_backbone_dir}\n")

    # Export to runtime paths
    export_head = str(args.export_head)
    export_backbone = str(args.export_backbone_dir)
    _safe_makedirs(os.path.dirname(export_head) or ".")
    shutil.copyfile(run_head_path, export_head)
    if os.path.exists(export_backbone):
        shutil.rmtree(export_backbone)
    shutil.copytree(run_backbone_dir, export_backbone)

    # ---- Generate separate Test Report ----
    test_run_dir = _next_run_dir(os.path.join("Test Report"), "MiniBERT Test")
    test_metrics_out = {
        "test_run_dir": test_run_dir,
        "training_run_dir": run_dir,
        "data": str(args.data),
        "labels_file": labels_path or None,
        "test_samples": len(split.test),
        "config": metrics_out["config"],
        "best_epoch": int(best["epoch"]),
        "test": {
            "avg": {"mse": test_avg_mse, "r2": test_avg_r2, "spearman": test_avg_spearman, "bin3_acc": test_avg_bin3},
            "per_output": test_per_output,
        },
    }
    with open(os.path.join(test_run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics_out, f, indent=2, ensure_ascii=False)

    with open(os.path.join(test_run_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write("MiniBERT LCR v2 — Test Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training run: {run_dir}\n")
        f.write(f"Data: {args.data}\n")
        if labels_path:
            f.write(f"Labels file: {labels_path}\n")
        f.write(f"Test samples: {len(split.test)}\n")
        f.write(f"Best epoch: {best['epoch']}\n\n")
        f.write("=== TEST RESULTS ===\n")
        f.write(f"  Average: mse={test_avg_mse:.6f} r2={test_avg_r2:.4f} spearman={test_avg_spearman:.4f} bin3_acc={test_avg_bin3:.4f}\n")
        for po in test_per_output:
            sp_ci_str = ""
            if "spearman_ci_low" in po:
                sp_ci_str = f" [95%CI: {po['spearman_ci_low']:.4f}–{po['spearman_ci_high']:.4f}]"
            f.write(f"  {po['name']}: mse={po['mse']:.6f} r2={po['r2']:.4f} spearman={po['spearman']:.4f}{sp_ci_str} bin3_acc={po['bin3_acc']:.4f}\n")
            f.write(f"    Per-source:\n")
            for src, sm in po['by_source'].items():
                f.write(f"      {src}: r2={sm['r2']:.4f} spearman={sm['spearman']:.4f} bin3={sm['bin3_acc']:.4f}\n")

    print(f"\n[LCR-v2] Run saved: {run_dir}")
    print(f"[LCR-v2] Test report saved: {test_run_dir}")
    print(f"[LCR-v2] Exported head → {export_head}")
    print(f"[LCR-v2] Exported backbone → {export_backbone}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  TEST RESULTS SUMMARY (n_outputs={n_outputs})")
    print(f"{'=' * 60}")
    print(f"  Average R²:       {test_avg_r2:.4f}")
    print(f"  Average Spearman: {test_avg_spearman:.4f}")
    for po in test_per_output:
        print(f"  [{po['name']}] R²={po['r2']:.4f}  Spearman={po['spearman']:.4f}")
    print(f"{'=' * 60}")
    print("[LCR-v2] done")


if __name__ == "__main__":
    main()
