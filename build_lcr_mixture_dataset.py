"""Build a small cumulative LCR training mixture dataset using HF streaming.

IMPORTANT:
- Uses Hugging Face `datasets` with `streaming=True` to avoid downloading entire datasets.
- Only a small subset from each dataset is sampled (configurable).

Outputs a CSV compatible with this repo's existing prompt format.

Default output columns:
  Prompt, Category, Subject, Context Dependency, SourceDataset, SourceSplit, SourceId

Example:
    python build_lcr_mixture_dataset.py --out lcr_mixture.csv --gsm8k 200 --mbpp 200 --wikitext2 200 --mmlu 200 --boolq 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Dict, Iterable, Iterator, Optional

from datasets import disable_caching, get_dataset_config_names, load_dataset


_WIKITEXT_HEADING_RE = re.compile(r"^\s*=+\s*[^=].*[^=]\s*=+\s*$")


def _is_wikitext_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    # Common raw WikiText headings look like: "= = Early life = =" or "= There ... =".
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
    # Filter very short or metadata-like lines which tend to be noisy for LCR.
    if len(s) < 50:
        return False
    if len(s.split()) < 8:
        return False
    return True


def _non_empty_wikitext_paragraphs(stream: Iterable[Dict]) -> Iterator[Dict]:
    for ex in stream:
        text = (ex.get("text") or "").strip()
        if _is_wikitext_good_paragraph(text):
            yield ex


def _take(stream: Iterable[Dict], n: int) -> Iterator[Dict]:
    i = 0
    for ex in stream:
        yield ex
        i += 1
        if i >= n:
            return


def _format_gsm8k(ex: Dict) -> str:
    q = (ex.get("question") or "").strip()
    return q


def _format_mbpp(ex: Dict) -> str:
    # mbpp uses field 'text' for instruction.
    t = (ex.get("text") or "").strip()
    return t


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
        # Keep it simple and consistent with other MC-like prompts.
        # BoolQ is binary yes/no.
        return f"Passage: {passage}\nQuestion: {question}\nA. False\nB. True"
    return (question or passage).strip()


def _stream_dataset(name: str, config: Optional[str], split: str) -> Iterable[Dict]:
    kwargs = {"streaming": True}
    if config is not None:
        ds = load_dataset(name, config, split=split, **kwargs)
    else:
        ds = load_dataset(name, split=split, **kwargs)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="lcr_mixture.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help="Streaming shuffle buffer size (0 disables shuffling).",
    )

    parser.add_argument("--gsm8k", type=int, default=1000, help="Samples from GSM8K")
    parser.add_argument("--mbpp", type=int, default=1000, help="Samples from MBPP")
    parser.add_argument("--wikitext2", type=int, default=1000, help="Samples from WikiText-2")
    parser.add_argument("--mmlu", type=int, default=1000, help="Samples from MMLU")
    parser.add_argument("--boolq", type=int, default=1000, help="Samples from BoolQ")

    parser.add_argument(
        "--include-mmlu-answers",
        action="store_true",
        help="Add choice/answer columns (Choices/AnswerIndex/AnswerLetter/AnswerText) for evaluation (MMLU, BoolQ).",
    )
    parser.add_argument(
        "--include-gsm8k-answer",
        action="store_true",
        help="Add GSM8K-only column (Gsm8kAnswer) for evaluation.",
    )
    parser.add_argument(
        "--include-mbpp-tests",
        action="store_true",
        help="Add MBPP-only column (MbppTests) containing JSON of test_list for evaluation.",
    )
    parser.add_argument(
        "--include-mbpp-code",
        action="store_true",
        help="Add MBPP-only column (MbppReferenceCode) containing the reference solution code.",
    )

    parser.add_argument("--split", default="train", help="Preferred split (falls back if missing)")
    args = parser.parse_args()

    random.seed(int(args.seed))
    disable_caching()

    rows = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fieldnames = [
        "Prompt",
        "Category",
        "Subject",
        "Context Dependency",
        "SourceDataset",
        "SourceSplit",
        "SourceId",
    ]
    include_any_extras = bool(args.include_mmlu_answers) or bool(args.include_gsm8k_answer) or bool(args.include_mbpp_tests) or bool(args.include_mbpp_code)
    if bool(args.include_mmlu_answers):
        fieldnames += ["Choices", "AnswerIndex", "AnswerLetter", "AnswerText"]
    if bool(args.include_gsm8k_answer):
        fieldnames += ["Gsm8kAnswer"]
    if bool(args.include_mbpp_tests):
        fieldnames += ["MbppTests"]
    if bool(args.include_mbpp_code):
        fieldnames += ["MbppReferenceCode"]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def write_prompt(
            prompt: str,
            category: str,
            subject: str,
            source_ds: str,
            source_split: str,
            source_id: str,
            extra: Optional[Dict[str, str]] = None,
            context_dependency: str = "No",
        ) -> None:
            nonlocal rows
            prompt = (prompt or "").strip()
            if not prompt:
                return
            row = {
                "Prompt": prompt,
                "Category": category,
                "Subject": subject,
                "Context Dependency": context_dependency,
                "SourceDataset": source_ds,
                "SourceSplit": source_split,
                "SourceId": source_id,
            }
            if extra:
                row.update(extra)
            if include_any_extras:
                # Ensure optional columns exist for rows that don't have them.
                if bool(args.include_mmlu_answers):
                    row.setdefault("Choices", "")
                    row.setdefault("AnswerIndex", "")
                    row.setdefault("AnswerLetter", "")
                    row.setdefault("AnswerText", "")
                if bool(args.include_gsm8k_answer):
                    row.setdefault("Gsm8kAnswer", "")
                if bool(args.include_mbpp_tests):
                    row.setdefault("MbppTests", "")
                if bool(args.include_mbpp_code):
                    row.setdefault("MbppReferenceCode", "")

            writer.writerow(row)
            rows += 1

        # GSM8K (Logic/Math)
        if int(args.gsm8k) > 0:
            print(f"[Mixture] GSM8K: sampling {int(args.gsm8k)}")
            # gsm8k has config 'main'
            split = args.split
            try:
                stream = _stream_dataset("gsm8k", "main", split)
            except Exception:
                split = "train"
                stream = _stream_dataset("gsm8k", "main", split)
            if int(args.shuffle_buffer) > 0:
                stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer))
            for ex in _take(stream, int(args.gsm8k)):
                prompt = _format_gsm8k(ex)
                extra = None
                if bool(args.include_gsm8k_answer):
                    extra = {"Gsm8kAnswer": str(ex.get("answer", "") or "").strip()}
                write_prompt(prompt, "Logic/Math", "GSM8K", "gsm8k", split, str(ex.get("idx", "")), extra=extra)

        # MBPP (Code)
        # MBPP is a small dataset: train=374, test=500, validation=90 (~964 total).
        # To reach 1000 samples, we combine ALL available splits automatically.
        if int(args.mbpp) > 0:
            print(f"[Mixture] MBPP: sampling {int(args.mbpp)}")
            collected = 0
            target = int(args.mbpp)
            # Try all splits in priority order to maximize sample count.
            mbpp_splits = ["train", "test", "validation", "prompt"]
            for sp in mbpp_splits:
                if collected >= target:
                    break
                try:
                    stream = _stream_dataset("mbpp", None, sp)
                except Exception:
                    continue
                if int(args.shuffle_buffer) > 0:
                    stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer))
                remaining = target - collected
                for ex in _take(stream, remaining):
                    prompt = _format_mbpp(ex)
                    extra = None
                    if bool(args.include_mbpp_tests) or bool(args.include_mbpp_code):
                        extra = {}
                        if bool(args.include_mbpp_tests):
                            tests = ex.get("test_list")
                            extra["MbppTests"] = json.dumps(list(tests) if isinstance(tests, (list, tuple)) else [], ensure_ascii=False)
                        if bool(args.include_mbpp_code):
                            extra["MbppReferenceCode"] = str(ex.get("code", "") or "").strip()
                    write_prompt(prompt, "Code", "MBPP", "mbpp", sp, str(ex.get("task_id", "")), extra=extra)
                    collected += 1
            print(f"[Mixture] MBPP: collected {collected} from {len(mbpp_splits)} splits")

        # WikiText-2 (Narrative)
        if int(args.wikitext2) > 0:
            print(f"[Mixture] WikiText-2: sampling {int(args.wikitext2)}")
            split = args.split
            try:
                stream = _stream_dataset("wikitext", "wikitext-2-raw-v1", split)
            except Exception:
                split = "train"
                stream = _stream_dataset("wikitext", "wikitext-2-raw-v1", split)
            if int(args.shuffle_buffer) > 0:
                stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer))
            stream = _non_empty_wikitext_paragraphs(stream)
            for ex in _take(stream, int(args.wikitext2)):
                prompt = _format_wikitext2(ex)
                write_prompt(prompt, "Narrative", "WikiText-2", "wikitext", split, "")

        # MMLU (Reasoning)
        if int(args.mmlu) > 0:
            print(f"[Mixture] MMLU: sampling {int(args.mmlu)}")
            split = args.split
            # MMLU requires a config name; prefer the aggregated 'all' config.
            # Some environments require using 'test' instead of 'train'.
            cfg_candidates = ["all"]
            try:
                cfg_candidates = ["all"] + [c for c in get_dataset_config_names("cais/mmlu") if c != "all"]
            except Exception:
                pass

            split_candidates = [split, "test", "validation", "dev"]
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
            if stream is None or used_split is None or used_cfg is None:
                raise RuntimeError("Failed to stream MMLU dataset (cais/mmlu).")

            if int(args.shuffle_buffer) > 0:
                stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer))
            for ex in _take(stream, int(args.mmlu)):
                prompt = _format_mmlu(ex)
                subject = (ex.get("subject") or "MMLU").strip() if isinstance(ex.get("subject"), str) else "MMLU"
                src_id = str(ex.get("question_id", ""))
                extra = None
                if bool(args.include_mmlu_answers):
                    choices = ex.get("choices")
                    answer_index = ex.get("answer")
                    answer_letter = ""
                    answer_text = ""
                    if isinstance(answer_index, int) and 0 <= answer_index < 26:
                        answer_letter = chr(ord("A") + answer_index)
                    if isinstance(choices, (list, tuple)) and isinstance(answer_index, int) and 0 <= answer_index < len(choices):
                        answer_text = str(choices[answer_index])
                    extra = {
                        "Choices": json.dumps(list(choices) if isinstance(choices, (list, tuple)) else [], ensure_ascii=False),
                        "AnswerIndex": str(answer_index) if answer_index is not None else "",
                        "AnswerLetter": answer_letter,
                        "AnswerText": answer_text,
                    }
                write_prompt(prompt, "Reasoning", subject, f"cais/mmlu:{used_cfg}", used_split, src_id, extra=extra)

        # BoolQ (QA / Reading Comprehension)
        if int(args.boolq) > 0:
            print(f"[Mixture] BoolQ: sampling {int(args.boolq)}")
            split = args.split
            try:
                stream = _stream_dataset("boolq", None, split)
            except Exception:
                split = "train"
                stream = _stream_dataset("boolq", None, split)
            if int(args.shuffle_buffer) > 0:
                stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer))
            for j, ex in enumerate(_take(stream, int(args.boolq)), start=1):
                prompt = _format_boolq(ex)
                extra = None
                if bool(args.include_mmlu_answers):
                    # Reuse the same choice/answer columns.
                    choices = ["False", "True"]
                    answer_val = ex.get("answer")
                    answer_index = 1 if bool(answer_val) else 0
                    answer_letter = "B" if answer_index == 1 else "A"
                    answer_text = choices[answer_index]
                    extra = {
                        "Choices": json.dumps(choices, ensure_ascii=False),
                        "AnswerIndex": str(answer_index),
                        "AnswerLetter": answer_letter,
                        "AnswerText": answer_text,
                    }
                write_prompt(
                    prompt,
                    "QA",
                    "BoolQ",
                    "boolq",
                    split,
                    str(j),
                    extra=extra,
                    context_dependency="Yes",
                )

    print(f"[Mixture] Wrote {rows} prompts -> {args.out}")


if __name__ == "__main__":
    main()
