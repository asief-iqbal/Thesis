"""Check that TinyBERT (prajjwal1/bert-tiny) can be loaded.

This will download the model weights if they are not already cached.
Run:
  python check_tinybert_install.py
"""

from __future__ import annotations

from transformers import AutoModel, AutoTokenizer


def main() -> None:
    name = "prajjwal1/bert-tiny"
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModel.from_pretrained(name, use_safetensors=True)
    _ = tok("hello", return_tensors="pt")
    print(f"OK: loaded tokenizer={type(tok).__name__} model={type(model).__name__} for {name}")


if __name__ == "__main__":
    main()
