"""One-click full-dataset pipeline for oracle labeling + MiniBERT/TinyBERT LCR training.

This keeps the raw dataset intact by writing oracle labels to a separate CSV,
then trains and evaluates the LCR router using the dataset's existing Split column.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import List


def _quote(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _run(cmd: List[str], dry_run: bool) -> None:
    print(f"[pipeline] {_quote(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run oracle labeling and MiniBERT/TinyBERT LCR training in one command.",
    )
    parser.add_argument("--input", default="Oracle_dataset.csv", help="Raw dataset CSV.")
    parser.add_argument("--text-column", default="Prompt", help="Prompt column name.")
    parser.add_argument("--labels-file", default="oracle_lcr_labels.csv", help="Separate oracle labels CSV output.")
    parser.add_argument("--meta-json", default="", help="Optional oracle metadata JSON path.")
    parser.add_argument("--output-dir", default="checkpoints", help="Runtime export directory.")
    parser.add_argument("--run-dir", default="", help="Optional training report run directory.")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Device for oracle labeling.")
    parser.add_argument("--samples", type=int, default=0, help="Rows to label. Use 0 for the full dataset.")
    parser.add_argument(
        "--sparse-configs",
        default="attention_heads:0.30,transformer_layers:0.25",
        help="Sparse oracle configs used for composite sensitivity labels.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-ratio", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--label-smooth", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model-name", default="prajjwal1/bert-mini")
    parser.add_argument("--label-columns", default="normalized_sensitivity")
    parser.add_argument(
        "--join-columns",
        default="SourceDataset,SourceSplit,SourceId,Category,Subject,Split,Prompt",
        help="Columns used to join the raw dataset with the labels CSV.",
    )
    parser.add_argument("--skip-labeling", action="store_true", help="Skip oracle labeling and reuse --labels-file.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training after generating labels.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    args = parser.parse_args()

    dataset_path = os.path.abspath(str(args.input))
    labels_path = os.path.abspath(str(args.labels_file))
    output_dir = os.path.abspath(str(args.output_dir))

    if not os.path.exists(dataset_path):
        raise SystemExit(f"Input dataset not found: {dataset_path}")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable
    oracle_script = os.path.join(root_dir, "oracle_labeler.py")
    train_script = os.path.join(root_dir, "train_minibert_lcr.py")

    if not args.skip_labeling:
        oracle_cmd = [
            python_exe,
            oracle_script,
            "--input",
            dataset_path,
            "--output",
            labels_path,
            "--text-column",
            str(args.text_column),
            "--samples",
            str(args.samples),
            "--device",
            str(args.device),
            "--sparse-configs",
            str(args.sparse_configs),
        ]
        if args.meta_json:
            oracle_cmd.extend(["--meta-json", str(args.meta_json)])
        _run(oracle_cmd, dry_run=bool(args.dry_run))

    if args.skip_training:
        return

    train_cmd = [
        python_exe,
        train_script,
        "--data",
        dataset_path,
        "--labels-file",
        labels_path,
        "--text-column",
        str(args.text_column),
        "--label-columns",
        str(args.label_columns),
        "--join-columns",
        str(args.join_columns),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--backbone-lr-factor",
        str(args.backbone_lr_factor),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--dropout",
        str(args.dropout),
        "--label-smooth",
        str(args.label_smooth),
        "--max-length",
        str(args.max_length),
        "--model-name",
        str(args.model_name),
        "--loss",
        "huber",
        "--huber-delta",
        "0.15",
        "--output-dir",
        output_dir,
    ]
    if args.run_dir:
        train_cmd.extend(["--run-dir", str(args.run_dir)])
    _run(train_cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()