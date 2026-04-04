"""Audit and optionally clean the LCR mixture CSV.

Why this exists
--------------
The raw WikiText-2 split includes section headings like "= = Early life = =" which
are not narrative paragraphs. Those rows degrade training and evaluation.

This script:
- audits a mixture CSV (counts, length stats, common issues)
- optionally writes a cleaned CSV (drops low-quality rows)
- writes a JSON report you can include in thesis appendices

Usage
-----
  python audit_lcr_mixture_dataset.py --input lcr_mixture.csv --report lcr_mixture_audit.json
  python audit_lcr_mixture_dataset.py --input lcr_mixture.csv --output-clean lcr_mixture.cleaned.csv --report lcr_mixture_audit.json

Notes
-----
- Uses only the Python standard library.
- Row numbers in the report are 1-based data rows (excluding the header).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


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


def _word_count(text: str) -> int:
    return len((text or "").strip().split())


def _alpha_count(text: str) -> int:
    return sum(1 for ch in (text or "") if ch.isalpha())


@dataclass(frozen=True)
class AuditConfig:
    min_chars_default: int = 30
    min_words_default: int = 5
    min_chars_wikitext: int = 50
    min_words_wikitext: int = 8


def _classify_row(row: Dict[str, str], cfg: AuditConfig) -> List[str]:
    prompt = (row.get("Prompt") or "").strip()
    src = (row.get("SourceDataset") or "").strip().lower()

    reasons: List[str] = []
    if not prompt:
        return ["empty_prompt"]

    if src == "wikitext":
        if _is_wikitext_heading(prompt):
            reasons.append("wikitext_heading")
        if len(prompt) < cfg.min_chars_wikitext:
            reasons.append("too_short_chars")
        if _word_count(prompt) < cfg.min_words_wikitext:
            reasons.append("too_short_words")
        if _alpha_count(prompt) < 20:
            reasons.append("low_alpha_content")
    else:
        if len(prompt) < cfg.min_chars_default:
            reasons.append("too_short_chars")
        if _word_count(prompt) < cfg.min_words_default:
            reasons.append("too_short_words")
        if _alpha_count(prompt) < 10:
            reasons.append("low_alpha_content")

    return reasons


def _read_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
        return fieldnames, rows


def _write_rows(path: str, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-clean", default="")
    parser.add_argument("--report", default="lcr_mixture_audit.json")
    args = parser.parse_args()

    cfg = AuditConfig()

    fieldnames, rows = _read_rows(args.input)
    total = len(rows)

    issues_by_reason: Counter[str] = Counter()
    issues_by_source: Dict[str, Counter[str]] = defaultdict(Counter)
    bad_row_numbers_by_reason: Dict[str, List[int]] = defaultdict(list)

    lengths: List[int] = []
    word_counts: List[int] = []

    good_rows: List[Dict[str, str]] = []
    bad_rows: List[Dict[str, str]] = []

    for i, row in enumerate(rows, start=1):
        prompt = (row.get("Prompt") or "").strip()
        src = (row.get("SourceDataset") or "").strip() or "(missing)"

        lengths.append(len(prompt))
        word_counts.append(_word_count(prompt))

        reasons = _classify_row(row, cfg)
        if reasons:
            bad_rows.append(row)
            for r in reasons:
                issues_by_reason[r] += 1
                issues_by_source[src][r] += 1
                bad_row_numbers_by_reason[r].append(i)
        else:
            good_rows.append(row)

    by_source = Counter((r.get("SourceDataset") or "(missing)") for r in rows)

    report = {
        "input": args.input,
        "rows_total": total,
        "rows_good": len(good_rows),
        "rows_flagged": len(bad_rows),
        "by_source": dict(by_source),
        "issues_by_reason": dict(issues_by_reason),
        "issues_by_source": {k: dict(v) for k, v in issues_by_source.items()},
        "bad_row_numbers_by_reason": {k: v for k, v in bad_row_numbers_by_reason.items()},
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "median": int(statistics.median(lengths)) if lengths else 0,
            "p95": int(statistics.quantiles(lengths, n=20)[-1]) if len(lengths) >= 20 else (max(lengths) if lengths else 0),
            "max": max(lengths) if lengths else 0,
        },
        "word_stats": {
            "min": min(word_counts) if word_counts else 0,
            "median": int(statistics.median(word_counts)) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
        },
        "config": {
            "min_chars_default": cfg.min_chars_default,
            "min_words_default": cfg.min_words_default,
            "min_chars_wikitext": cfg.min_chars_wikitext,
            "min_words_wikitext": cfg.min_words_wikitext,
        },
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.output_clean:
        _write_rows(args.output_clean, fieldnames, good_rows)

    print(f"[Audit] {args.input}: total={total} good={len(good_rows)} flagged={len(bad_rows)}")
    if issues_by_reason:
        print("[Audit] Top issues:")
        for reason, count in issues_by_reason.most_common(10):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
