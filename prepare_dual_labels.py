"""Prepare dual per-method sensitivity labels + auxiliary features from oracle CSV.

Reads the multi-method oracle CSV (with per-config gap columns) and produces
an augmented CSV for dual-output LCR training.

Output columns (added):
  - sensitivity_head:  normalized head-pruning sensitivity [0, 1]
  - sensitivity_layer: normalized layer-skipping sensitivity [0, 1]
  - log_token_count, compression_ratio, avg_word_length,
    special_char_ratio, unique_token_ratio, has_code_markers

Usage:
  python prepare_dual_labels.py \\
    --input oracle_lcr_10k_composite.csv \\
    --output oracle_lcr_10k_dual.csv \\
    --head-gap-col loss_gap_attention_heads_0p30 \\
    --layer-gap-col loss_gap_transformer_layers_0p25 \\
    --clip-percentiles 5,95
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List, Sequence, Tuple

from lcr_tinybert import compute_aux_features, AUX_FEATURE_NAMES


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (0..100)."""
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if p <= 0.0:
        return float(xs[0])
    if p >= 100.0:
        return float(xs[-1])
    k = (len(xs) - 1) * (p / 100.0)
    f, c = int(math.floor(k)), int(math.ceil(k))
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def pclip_normalize(values: Sequence[float], p_low: float = 5.0, p_high: float = 95.0) -> Tuple[List[float], Dict[str, float]]:
    """Percentile-clipped min-max normalize values to [0, 1]."""
    clip_low = _percentile(values, p_low)
    clip_high = _percentile(values, p_high)
    denom = max(1e-8, clip_high - clip_low)

    labels = []
    for v in values:
        vc = min(max(float(v), clip_low), clip_high)
        labels.append(max(0.0, min(1.0, (vc - clip_low) / denom)))

    return labels, {
        "method": "pclip_minmax",
        "p_low": p_low, "p_high": p_high,
        "clip_low": float(clip_low), "clip_high": float(clip_high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dual per-method labels and aux features.")
    parser.add_argument("--input", required=True, help="Input oracle CSV")
    parser.add_argument("--output", required=True, help="Output augmented CSV")
    parser.add_argument("--text-column", default="Prompt")
    parser.add_argument("--head-gap-col", default="loss_gap_attention_heads_0p30",
                        help="Column name for head-pruning loss gap")
    parser.add_argument("--layer-gap-col", default="loss_gap_transformer_layers_0p25",
                        help="Column name for layer-skipping loss gap")
    parser.add_argument("--clip-percentiles", default="5,95",
                        help="Low,high percentiles for normalization (e.g. '5,95')")
    args = parser.parse_args()

    parts = args.clip_percentiles.split(",")
    p_low, p_high = float(parts[0].strip()), float(parts[1].strip())

    # Read input
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    n = len(rows)
    print(f"[DualLabels] Read {n} rows from {args.input}")
    print(f"[DualLabels] Head gap column: {args.head_gap_col}")
    print(f"[DualLabels] Layer gap column: {args.layer_gap_col}")

    # Validate columns exist
    for col in [args.head_gap_col, args.layer_gap_col, args.text_column]:
        if col not in fieldnames:
            raise SystemExit(f"Column '{col}' not found. Available: {fieldnames}")

    # Extract gaps (clamp negatives at 0 for stable sensitivity)
    head_gaps = [max(0.0, float(r[args.head_gap_col])) for r in rows]
    layer_gaps = [max(0.0, float(r[args.layer_gap_col])) for r in rows]

    # Independently normalize each method
    head_labels, head_meta = pclip_normalize(head_gaps, p_low, p_high)
    layer_labels, layer_meta = pclip_normalize(layer_gaps, p_low, p_high)

    print(f"[DualLabels] Head normalization: {head_meta}")
    print(f"[DualLabels] Layer normalization: {layer_meta}")

    # Per-source statistics
    sources: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        src = r.get("SourceDataset", "unknown")
        if src not in sources:
            sources[src] = []
        sources[src].append(i)

    print(f"\n[DualLabels] Per-source label statistics:")
    for src in sorted(sources.keys()):
        idx = sources[src]
        h = [head_labels[i] for i in idx]
        l = [layer_labels[i] for i in idx]
        h_mean = sum(h) / len(h)
        l_mean = sum(l) / len(l)
        print(f"  {src:20s} n={len(idx):5d}  head_mean={h_mean:.3f}  layer_mean={l_mean:.3f}")

    # Compute auxiliary features
    print(f"\n[DualLabels] Computing auxiliary features for {n} prompts...")
    all_aux = []
    for i, r in enumerate(rows):
        aux = compute_aux_features(r[args.text_column])
        all_aux.append(aux)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n} done")

    # Write output
    new_cols = ["sensitivity_head", "sensitivity_layer"] + list(AUX_FEATURE_NAMES)
    out_fieldnames = fieldnames + [c for c in new_cols if c not in fieldnames]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows):
            r["sensitivity_head"] = f"{head_labels[i]:.6f}"
            r["sensitivity_layer"] = f"{layer_labels[i]:.6f}"
            for j, fname in enumerate(AUX_FEATURE_NAMES):
                r[fname] = f"{all_aux[i][j]:.6f}"
            writer.writerow(r)

    print(f"\n[DualLabels] Wrote {n} rows → {args.output}")
    print(f"[DualLabels] New columns: {new_cols}")


if __name__ == "__main__":
    main()
