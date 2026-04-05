"""Build the Oracle Dataset (10K) for CASRAP adaptive pruning.

Creates oracle_dataset.csv with 2000 samples from each of 5 sources:
  - GSM8K    (Logic/Math)
  - MBPP     (Code)
  - WikiText-2 (Narrative)
  - MMLU     (Reasoning)
  - BoolQ    (QA / Reading Comprehension)

Includes all answer/evaluation metadata and an 80-20 train/test Split column.
The Split column is stratified by source (each source has 80% train, 20% test).

Columns:
  Prompt, Category, Subject, Context Dependency, SourceDataset, SourceSplit,
  SourceId, Choices, AnswerIndex, AnswerLetter, AnswerText, Gsm8kAnswer,
  MbppTests, Split

Usage:
  python build_oracle_dataset.py
  python build_oracle_dataset.py --out oracle_dataset.csv --per-source 2000 --train-ratio 0.8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import disable_caching, get_dataset_config_names, load_dataset


# ---------------------------------------------------------------------------
# WikiText filtering
# ---------------------------------------------------------------------------

_WIKITEXT_HEADING_RE = re.compile(r"^\s*=+\s*[^=].*[^=]\s*=+\s*$")


def _is_wikitext_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _WIKITEXT_HEADING_RE.match(s):
        return True
    if s.startswith("=") and s.endswith("=") and s.count("=") >= 2:
        return True
    return False


def _is_wikitext_good_paragraph(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _is_wikitext_heading(s):
        return False
    if len(s) < 50 or len(s.split()) < 8:
        return False
    return True


def _non_empty_wikitext_paragraphs(stream: Iterable[Dict]) -> Iterator[Dict]:
    for ex in stream:
        text = (ex.get("text") or "").strip()
        if _is_wikitext_good_paragraph(text):
            yield ex


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def _take(stream: Iterable[Dict], n: int) -> Iterator[Dict]:
    i = 0
    for ex in stream:
        yield ex
        i += 1
        if i >= n:
            return


def _stream_dataset(name: str, config: Optional[str], split: str) -> Iterable[Dict]:
    kwargs = {"streaming": True}
    if config is not None:
        return load_dataset(name, config, split=split, **kwargs)
    return load_dataset(name, split=split, **kwargs)


# ---------------------------------------------------------------------------
# Prompt formatting (same as build_lcr_mixture_dataset.py)
# ---------------------------------------------------------------------------

def _format_gsm8k(ex: Dict) -> str:
    return (ex.get("question") or "").strip()


def _format_mbpp(ex: Dict) -> str:
    return (ex.get("text") or "").strip()


def _format_wikitext2(ex: Dict) -> str:
    return (ex.get("text") or "").strip()


def _format_mmlu(ex: Dict) -> str:
    q = (ex.get("question") or "").strip()
    choices = ex.get("choices")
    if isinstance(choices, (list, tuple)) and choices:
        opts = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices[:4])])
        return f"{q}\n{opts}"
    return q


def _format_boolq(ex: Dict) -> str:
    passage = (ex.get("passage") or "").strip()
    question = (ex.get("question") or "").strip()
    if passage and question:
        return f"Passage: {passage}\nQuestion: {question}\nA. False\nB. True"
    return (question or passage).strip()


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "Prompt", "Category", "Subject", "Context Dependency",
    "SourceDataset", "SourceSplit", "SourceId",
    "Choices", "AnswerIndex", "AnswerLetter", "AnswerText",
    "Gsm8kAnswer", "MbppTests", "Split",
]


def _make_row(
    prompt: str,
    category: str,
    subject: str,
    source_ds: str,
    source_split: str,
    source_id: str,
    context_dependency: str = "No",
    choices: str = "",
    answer_index: str = "",
    answer_letter: str = "",
    answer_text: str = "",
    gsm8k_answer: str = "",
    mbpp_tests: str = "",
) -> Optional[Dict[str, str]]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    return {
        "Prompt": prompt,
        "Category": category,
        "Subject": subject,
        "Context Dependency": context_dependency,
        "SourceDataset": source_ds,
        "SourceSplit": source_split,
        "SourceId": source_id,
        "Choices": choices,
        "AnswerIndex": answer_index,
        "AnswerLetter": answer_letter,
        "AnswerText": answer_text,
        "Gsm8kAnswer": gsm8k_answer,
        "MbppTests": mbpp_tests,
        "Split": "",  # assigned later
    }


def _oversample(rows: List[Dict], target: int, seed: int) -> List[Dict]:
    """Duplicate rows randomly until we reach `target` count."""
    if len(rows) >= target:
        return rows[:target]
    rng = random.Random(seed)
    result = list(rows)
    while len(result) < target:
        result.append(rng.choice(rows).copy())
    return result[:target]


# ---------------------------------------------------------------------------
# Collection from each source
# ---------------------------------------------------------------------------

def collect_gsm8k(n: int, split: str, seed: int, shuffle_buffer: int) -> List[Dict]:
    print(f"[Oracle Dataset] GSM8K: streaming {n} samples...")
    try:
        stream = _stream_dataset("gsm8k", "main", split)
    except Exception:
        split = "train"
        stream = _stream_dataset("gsm8k", "main", split)
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    rows = []
    for ex in _take(stream, n):
        prompt = _format_gsm8k(ex)
        answer = str(ex.get("answer", "") or "").strip()
        row = _make_row(prompt, "Logic/Math", "GSM8K", "gsm8k", split, str(ex.get("idx", "")),
                        gsm8k_answer=answer)
        if row:
            rows.append(row)
    print(f"[Oracle Dataset] GSM8K: collected {len(rows)} unique samples")
    if len(rows) < n:
        rows = _oversample(rows, n, seed)
        print(f"[Oracle Dataset] GSM8K: oversampled to {len(rows)}")
    return rows


def collect_mbpp(n: int, seed: int, shuffle_buffer: int) -> List[Dict]:
    print(f"[Oracle Dataset] MBPP: streaming up to {n} samples (dataset is small, ~974 total)...")
    all_rows = []
    mbpp_splits = ["train", "test", "validation", "prompt"]
    for sp in mbpp_splits:
        try:
            stream = _stream_dataset("mbpp", None, sp)
        except Exception:
            continue
        if shuffle_buffer > 0:
            stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
        for ex in stream:
            prompt = _format_mbpp(ex)
            tests = ex.get("test_list")
            mbpp_tests = json.dumps(list(tests) if isinstance(tests, (list, tuple)) else [], ensure_ascii=False)
            row = _make_row(prompt, "Code", "MBPP", "mbpp", sp, str(ex.get("task_id", "")),
                            mbpp_tests=mbpp_tests)
            if row:
                all_rows.append(row)
    print(f"[Oracle Dataset] MBPP: collected {len(all_rows)} unique samples")
    if len(all_rows) < n:
        all_rows = _oversample(all_rows, n, seed)
        print(f"[Oracle Dataset] MBPP: oversampled to {len(all_rows)}")
    else:
        rng = random.Random(seed)
        rng.shuffle(all_rows)
        all_rows = all_rows[:n]
    return all_rows


def collect_wikitext2(n: int, split: str, seed: int, shuffle_buffer: int) -> List[Dict]:
    print(f"[Oracle Dataset] WikiText-2: streaming {n} samples...")
    try:
        stream = _stream_dataset("wikitext", "wikitext-2-raw-v1", split)
    except Exception:
        split = "train"
        stream = _stream_dataset("wikitext", "wikitext-2-raw-v1", split)
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    stream = _non_empty_wikitext_paragraphs(stream)
    rows = []
    for ex in _take(stream, n):
        prompt = _format_wikitext2(ex)
        row = _make_row(prompt, "Narrative", "WikiText-2", "wikitext", split, "")
        if row:
            rows.append(row)
    print(f"[Oracle Dataset] WikiText-2: collected {len(rows)} unique samples")
    if len(rows) < n:
        rows = _oversample(rows, n, seed)
        print(f"[Oracle Dataset] WikiText-2: oversampled to {len(rows)}")
    return rows


def collect_mmlu(n: int, seed: int, shuffle_buffer: int) -> List[Dict]:
    print(f"[Oracle Dataset] MMLU: streaming {n} samples...")
    cfg_candidates = ["all"]
    try:
        cfg_candidates = ["all"] + [c for c in get_dataset_config_names("cais/mmlu") if c != "all"]
    except Exception:
        pass

    split_candidates = ["train", "test", "validation", "dev"]
    stream = None
    used_split = None
    used_cfg = None
    for cfg in cfg_candidates:
        for sp in split_candidates:
            try:
                stream = _stream_dataset("cais/mmlu", cfg, sp)
                used_split = sp
                used_cfg = cfg
                break
            except Exception:
                continue
        if stream is not None:
            break
    if stream is None:
        raise RuntimeError("Failed to stream MMLU dataset (cais/mmlu).")

    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    rows = []
    for ex in _take(stream, n):
        prompt = _format_mmlu(ex)
        subject = (ex.get("subject") or "MMLU").strip() if isinstance(ex.get("subject"), str) else "MMLU"
        choices = ex.get("choices")
        answer_index = ex.get("answer")
        answer_letter = ""
        answer_text = ""
        if isinstance(answer_index, int) and 0 <= answer_index < 26:
            answer_letter = chr(ord("A") + answer_index)
        if isinstance(choices, (list, tuple)) and isinstance(answer_index, int) and 0 <= answer_index < len(choices):
            answer_text = str(choices[answer_index])
        choices_json = json.dumps(list(choices) if isinstance(choices, (list, tuple)) else [], ensure_ascii=False)
        row = _make_row(
            prompt, "Reasoning", subject, f"cais/mmlu:{used_cfg}", used_split,
            str(ex.get("question_id", "")),
            choices=choices_json,
            answer_index=str(answer_index) if answer_index is not None else "",
            answer_letter=answer_letter,
            answer_text=answer_text,
        )
        if row:
            rows.append(row)
    print(f"[Oracle Dataset] MMLU: collected {len(rows)} unique samples")
    if len(rows) < n:
        rows = _oversample(rows, n, seed)
        print(f"[Oracle Dataset] MMLU: oversampled to {len(rows)}")
    return rows


def collect_boolq(n: int, split: str, seed: int, shuffle_buffer: int) -> List[Dict]:
    print(f"[Oracle Dataset] BoolQ: streaming {n} samples...")
    try:
        stream = _stream_dataset("boolq", None, split)
    except Exception:
        split = "train"
        stream = _stream_dataset("boolq", None, split)
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    rows = []
    for j, ex in enumerate(_take(stream, n), start=1):
        prompt = _format_boolq(ex)
        choices = ["False", "True"]
        answer_val = ex.get("answer")
        answer_index = 1 if bool(answer_val) else 0
        answer_letter = "B" if answer_index == 1 else "A"
        answer_text = choices[answer_index]
        row = _make_row(
            prompt, "QA", "BoolQ", "boolq", split, str(j),
            context_dependency="Yes",
            choices=json.dumps(choices, ensure_ascii=False),
            answer_index=str(answer_index),
            answer_letter=answer_letter,
            answer_text=answer_text,
        )
        if row:
            rows.append(row)
    print(f"[Oracle Dataset] BoolQ: collected {len(rows)} unique samples")
    if len(rows) < n:
        rows = _oversample(rows, n, seed)
        print(f"[Oracle Dataset] BoolQ: oversampled to {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(rows: List[Dict], train_ratio: float, seed: int) -> List[Dict]:
    """Assign 80-20 train/test split stratified by SourceDataset."""
    by_source: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        src = row.get("SourceDataset", "unknown")
        by_source.setdefault(src, []).append(i)

    rng = random.Random(seed)
    for src, indices in sorted(by_source.items()):
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        for j, idx in enumerate(indices):
            rows[idx]["Split"] = "train" if j < n_train else "test"

    # Report
    train_count = sum(1 for r in rows if r["Split"] == "train")
    test_count = sum(1 for r in rows if r["Split"] == "test")
    print(f"[Oracle Dataset] Split assignment: {train_count} train, {test_count} test "
          f"({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f})")
    for src, indices in sorted(by_source.items()):
        t = sum(1 for i in indices if rows[i]["Split"] == "train")
        e = sum(1 for i in indices if rows[i]["Split"] == "test")
        print(f"  {src}: {t} train, {e} test")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 10K Oracle Dataset for CASRAP")
    parser.add_argument("--out", default="oracle_dataset.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=2000, help="Streaming shuffle buffer size")
    parser.add_argument("--per-source", type=int, default=2000, help="Target samples per source (default: 2000)")
    parser.add_argument("--gsm8k", type=int, default=None, help="Override GSM8K sample count")
    parser.add_argument("--mbpp", type=int, default=None, help="Override MBPP sample count")
    parser.add_argument("--wikitext2", type=int, default=None, help="Override WikiText-2 sample count")
    parser.add_argument("--mmlu", type=int, default=None, help="Override MMLU sample count")
    parser.add_argument("--boolq", type=int, default=None, help="Override BoolQ sample count")
    parser.add_argument("--split", default="train", help="Preferred HF dataset split to stream from")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio (default: 0.8)")
    args = parser.parse_args()

    random.seed(args.seed)
    disable_caching()

    n_default = int(args.per_source)
    n_gsm8k = int(args.gsm8k) if args.gsm8k is not None else n_default
    n_mbpp = int(args.mbpp) if args.mbpp is not None else n_default
    n_wikitext2 = int(args.wikitext2) if args.wikitext2 is not None else n_default
    n_mmlu = int(args.mmlu) if args.mmlu is not None else n_default
    n_boolq = int(args.boolq) if args.boolq is not None else n_default

    print(f"[Oracle Dataset] Building {n_gsm8k + n_mbpp + n_wikitext2 + n_mmlu + n_boolq} samples "
          f"({n_gsm8k} GSM8K, {n_mbpp} MBPP, {n_wikitext2} WikiText-2, {n_mmlu} MMLU, {n_boolq} BoolQ)")

    all_rows: List[Dict] = []
    all_rows.extend(collect_gsm8k(n_gsm8k, args.split, args.seed, args.shuffle_buffer))
    all_rows.extend(collect_mbpp(n_mbpp, args.seed, args.shuffle_buffer))
    all_rows.extend(collect_wikitext2(n_wikitext2, args.split, args.seed, args.shuffle_buffer))
    all_rows.extend(collect_mmlu(n_mmlu, args.seed, args.shuffle_buffer))
    all_rows.extend(collect_boolq(n_boolq, args.split, args.seed, args.shuffle_buffer))

    # Assign train/test split
    all_rows = assign_splits(all_rows, train_ratio=float(args.train_ratio), seed=args.seed)

    # Write CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    total = len(all_rows)
    print(f"\n[Oracle Dataset] Wrote {total} rows -> {args.out}")
    print(f"[Oracle Dataset] Done.")


if __name__ == "__main__":
    main()
