"""Minimal smoke test for the TinyBERT LCR v2 scorer (multi-output).

This is intentionally small and safe to run:
    python smoke_test_lcr.py
"""

from __future__ import annotations

import os
import time

from lcr_tinybert import TinyBertLcrConfig, TinyBertLcrScorer


def main() -> None:
    backbone_dir = os.path.join("checkpoints", "tinybert_lcr_backbone")
    model_name = backbone_dir if os.path.isdir(backbone_dir) else "prajjwal1/bert-mini"
    scorer = TinyBertLcrScorer(
        TinyBertLcrConfig(
            model_name=model_name,
            max_length=128,
            device="cpu",
            head_checkpoint_path="checkpoints/tinybert_lcr_head.pt",
            n_outputs=2,          # dual-output: head_sensitivity + layer_sensitivity
            use_aux_features=True, # include statistical features
        )
    )

    prompts = [
        ("simple",     "Write a short poem about the sea."),
        ("math",       "Solve: If 3x + 5 = 20, what is x? Show steps."),
        ("code",       "Implement quicksort in Python and explain time complexity."),
        ("knowledge",  "What is the capital of France and what is its population?"),
        ("reasoning",  "If all cats have tails and Mittens is a cat, does Mittens have a tail?"),
    ]

    print(f"\n{'='*70}")
    print(f"  LCR v2 Smoke Test (n_outputs={scorer.config.n_outputs})")
    print(f"  Backbone: {model_name}")
    print(f"  Head loaded: {scorer._head_loaded if hasattr(scorer, '_head_loaded') else 'N/A'}")
    print(f"{'='*70}\n")

    for label, p in prompts:
        t0 = time.time()
        score = scorer.score(p)
        dt_ms = (time.time() - t0) * 1000

        # Also test dual output
        head_s, layer_s = scorer.score_dual(p)
        print(f"[{label:10s}] score={score:.3f}  head={head_s:.3f}  layer={layer_s:.3f}  | {dt_ms:.1f}ms")

    print(f"\n{'='*70}")
    print("  All tests passed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
