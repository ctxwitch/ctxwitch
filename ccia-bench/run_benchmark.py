#!/usr/bin/env python3
"""Run CBIA against ccia-bench pairs and print an accuracy report.

Usage (from repo root):
    python ccia-bench/run_benchmark.py
    python ccia-bench/run_benchmark.py --pairs ccia-bench/pairs.jsonl
    python ccia-bench/run_benchmark.py --failures   # show only mismatches
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running without pip install
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctxwitch.core.behavioral import analyze_behavioral_impact
from ctxwitch.core.dimensions import Dimension, Severity

DEFAULT_PAIRS = Path(__file__).parent / "pairs.jsonl"

SEVERITY_FROM_STR = {
    "no_change": Severity.NO_CHANGE,
    "cosmetic": Severity.COSMETIC,
    "minor": Severity.MINOR,
    "significant": Severity.SIGNIFICANT,
    "breaking": Severity.BREAKING,
}

DIMENSION_FROM_STR = {d.value: d for d in Dimension}


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def parse_severity(value: str) -> Severity:
    key = value.lower().replace(" ", "_")
    if key not in SEVERITY_FROM_STR:
        raise ValueError(f"Unknown severity: {value}")
    return SEVERITY_FROM_STR[key]


def predict(pair: Dict[str, Any], use_judge: bool = False) -> Tuple[Severity, Dict[Dimension, Severity]]:
    report = analyze_behavioral_impact(pair["old"], pair["new"], use_judge=use_judge)
    by_dim = {imp.dimension: imp.severity for imp in report.impacts}
    return report.compound_severity, by_dim


def expected_labels(pair: Dict[str, Any]) -> Tuple[Severity, Dict[Dimension, Severity]]:
    labels = pair["labels"]
    compound = parse_severity(labels["compound"])
    dims: Dict[Dimension, Severity] = {}
    for dim_str, sev_str in labels.get("dimensions", {}).items():
        dim = DIMENSION_FROM_STR.get(dim_str)
        if dim is None:
            raise ValueError(f"Pair {pair['id']}: unknown dimension {dim_str}")
        dims[dim] = parse_severity(sev_str)
    return compound, dims


def severity_match(expected: Severity, predicted: Severity, tolerance: int = 0) -> bool:
    if tolerance == 0:
        return expected == predicted
    return abs(int(expected) - int(predicted)) <= tolerance


def run_benchmark(
    pairs: List[Dict[str, Any]],
    use_judge: bool = False,
    tolerance: int = 0,
    show_failures: bool = False,
) -> int:
    compound_correct = 0
    compound_within_1 = 0
    dim_checked = 0
    dim_correct = 0
    dim_within_1 = 0
    # Detection-level counts. Labels list every dimension the annotator
    # considers changed; anything else CBIA flags at MINOR+ is a false
    # positive — a scorer that ignores those measures recall only.
    det_tp = 0
    det_fn = 0
    det_fp = 0
    confusion: Dict[Tuple[Severity, Severity], int] = {}
    failures: List[str] = []

    for pair in pairs:
        exp_compound, exp_dims = expected_labels(pair)
        pred_compound, pred_dims = predict(pair, use_judge=use_judge)

        confusion[(exp_compound, pred_compound)] = (
            confusion.get((exp_compound, pred_compound), 0) + 1
        )
        if severity_match(exp_compound, pred_compound, tolerance):
            compound_correct += 1
        else:
            failures.append(
                f"  {pair['id']} compound: expected {exp_compound.label}, "
                f"got {pred_compound.label} — {pair['description']}"
            )
        if severity_match(exp_compound, pred_compound, 1):
            compound_within_1 += 1

        for dim, exp_sev in exp_dims.items():
            dim_checked += 1
            pred_sev = pred_dims.get(dim, Severity.NO_CHANGE)
            if pred_sev > Severity.NO_CHANGE:
                det_tp += 1
            else:
                det_fn += 1
            if severity_match(exp_sev, pred_sev, tolerance):
                dim_correct += 1
            else:
                failures.append(
                    f"  {pair['id']} {dim.value}: expected {exp_sev.label}, "
                    f"got {pred_sev.label} — {pair['description']}"
                )
            if severity_match(exp_sev, pred_sev, 1):
                dim_within_1 += 1

        for dim, pred_sev in pred_dims.items():
            if pred_sev >= Severity.MINOR and dim not in exp_dims:
                det_fp += 1
                failures.append(
                    f"  {pair['id']} {dim.value}: FALSE POSITIVE — predicted "
                    f"{pred_sev.label}, not in labels — {pair['description']}"
                )

    total = len(pairs)
    compound_pct = 100.0 * compound_correct / total if total else 0.0
    dim_pct = 100.0 * dim_correct / dim_checked if dim_checked else 0.0
    precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) else 0.0
    recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("=" * 60)
    print("CCIA-Bench v0 — CBIA accuracy report")
    print("=" * 60)
    print(f"Pairs:              {total}")
    print(f"Tier 6 judge:       {'on' if use_judge else 'off'}")
    print(f"Severity tolerance: ±{tolerance} level(s)")
    print()
    print(f"Compound severity:  {compound_correct}/{total} ({compound_pct:.1f}%)"
          f"   [within ±1: {compound_within_1}/{total}]")
    print(f"Labeled dimensions: {dim_correct}/{dim_checked} ({dim_pct:.1f}%)"
          f"   [within ±1: {dim_within_1}/{dim_checked}]")
    print()
    print("Dimension detection (labeled = ground-truth changed):")
    print(f"  precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}"
          f"   (TP {det_tp} / FP {det_fp} / FN {det_fn}; FP = predicted MINOR+ off-label)")
    print()
    _print_confusion(confusion)
    print()

    if failures:
        if show_failures or compound_correct < total:
            print(f"Mismatches ({len(failures)}):")
            for line in failures:
                print(line)
            print()
        print("Tip: mismatches show where CBIA differs from expert labels.")
        print("     Use this to prioritize accuracy improvements.")
    else:
        print("All pairs matched expected labels.")

    print("=" * 60)
    return 0 if compound_correct == total and dim_correct == dim_checked and det_fp == 0 else 1


def _print_confusion(confusion: Dict[Tuple[Severity, Severity], int]) -> None:
    """Print the compound-severity confusion matrix (rows=expected, cols=predicted)."""
    levels = list(Severity)
    label_w = max(len(s.label) for s in levels)
    header = " " * (label_w + 2) + "".join(f"{s.label[:8]:>10}" for s in levels)
    print("Compound confusion (rows expected / cols predicted):")
    print(header)
    for exp in levels:
        row_counts = [confusion.get((exp, pred), 0) for pred in levels]
        if not any(row_counts):
            continue
        cells = "".join(f"{c or '.':>10}" for c in row_counts)
        print(f"  {exp.label:<{label_w}}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCIA-Bench against CBIA")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS,
        help="Path to pairs.jsonl",
    )
    parser.add_argument("--judge", action="store_true", help="Enable Tier 6 LLM judge")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Allow severity to differ by N levels (0 = exact)",
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="Only print mismatch details (still prints summary)",
    )
    args = parser.parse_args()

    if not args.pairs.exists():
        print(f"Missing {args.pairs}. Run: python ccia-bench/generate_pairs.py", file=sys.stderr)
        sys.exit(2)

    pairs = load_pairs(args.pairs)
    code = run_benchmark(
        pairs,
        use_judge=args.judge,
        tolerance=args.tolerance,
        show_failures=args.failures,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
