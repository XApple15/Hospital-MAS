#!/usr/bin/env python
"""
Evaluation report generator for HospitalMAS.

Reads the JSON output from eval_runner.py and produces:
  - Overall accuracy metrics (Top-1, Top-3 for both phases)
  - Per-variant accuracy breakdown
  - Per-prognosis accuracy breakdown
  - Confusion matrix (predicted-top1 vs expected)
  - Detailed per-case result table
  - Failure analysis (cases where the expected disease was missed entirely)

Usage:
    # Generate a terminal report from the latest eval results
    python -m hospitalmas.eval_report eval_results/eval_all_20250329_120000.json

    # Export to CSV for spreadsheet analysis
    python -m hospitalmas.eval_report results.json --export-csv report.csv

    # Generate a full HTML report
    python -m hospitalmas.eval_report results.json --export-html report.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── Load results ──────────────────────────────────────────────────────────────

def load_results(path: str | Path) -> dict[str, Any]:
    """Load evaluation results JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Confusion matrix ─────────────────────────────────────────────────────────

def build_confusion_matrix(
    results: list[dict[str, Any]],
    phase: str = "phase2",
) -> dict[str, dict[str, int]]:
    """
    Build a confusion matrix: actual (expected) vs predicted (top-1).

    Returns:
        {actual_disease: {predicted_disease: count, ...}, ...}
    """
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    top_key = f"{phase}_top1"

    for r in results:
        if r.get("error"):
            continue
        actual = r["expected_prognosis"].strip()
        predicted = (r.get(top_key) or "NO_PREDICTION").strip()
        matrix[actual][predicted] += 1

    # Convert Counter → plain dict for serialisation
    return {k: dict(v) for k, v in sorted(matrix.items())}


# ── Failure analysis ──────────────────────────────────────────────────────────

def failure_analysis(
    results: list[dict[str, Any]],
    phase: str = "phase2",
) -> list[dict[str, Any]]:
    """
    Return cases where the expected prognosis was NOT in the top-3.
    These are the hardest / most problematic cases for the pipeline.
    """
    top3_key = f"top3_correct_{phase}"
    failures = []
    for r in results:
        if r.get("error"):
            failures.append({
                "case_id": r["case_id"],
                "variant": r["variant"],
                "expected": r["expected_prognosis"],
                "predicted_top1": None,
                "predicted_top3": [],
                "reason": f"Error: {r['error'][:200]}",
            })
        elif not r.get(top3_key, False):
            top1_key = f"{phase}_top1"
            top3_list_key = f"{phase}_top3"
            failures.append({
                "case_id": r["case_id"],
                "variant": r["variant"],
                "expected": r["expected_prognosis"],
                "predicted_top1": r.get(top1_key),
                "predicted_top3": r.get(top3_list_key, []),
                "reason": "Expected disease not in top-3",
            })
    return failures


# ── Terminal report ───────────────────────────────────────────────────────────

def print_terminal_report(report: dict[str, Any]) -> None:
    """Print a comprehensive terminal report."""
    meta = report.get("meta", {})
    summary = report.get("summary", {})
    cases = report.get("cases", [])

    print(f"\n{'═'*72}")
    print(f"  HOSPITALMAS EVALUATION REPORT")
    print(f"{'═'*72}")
    print(f"  Timestamp:       {meta.get('timestamp', 'N/A')}")
    print(f"  CSV source:      {meta.get('csv_path', 'N/A')}")
    print(f"  Variant filter:  {meta.get('variant_filter') or 'all'}")
    print(f"  Prognosis filter: {meta.get('prognosis_filter') or 'none'}")
    print(f"  Total cases:     {meta.get('total_cases', 0)}")
    print(f"  Errors:          {meta.get('errors', 0)}")

    # ── Overall accuracy ─────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  OVERALL ACCURACY")
    print(f"{'─'*72}")
    n = summary.get("total_evaluated", 0)
    print(f"  {'Metric':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'─'*58}")
    for label, correct_key, acc_key in [
        ("Phase 1 Top-1", "phase1_top1_correct", "phase1_top1_accuracy"),
        ("Phase 1 Top-3", "phase1_top3_correct", "phase1_top3_accuracy"),
        ("Phase 2 Top-1", "phase2_top1_correct", "phase2_top1_accuracy"),
        ("Phase 2 Top-3", "phase2_top3_correct", "phase2_top3_accuracy"),
    ]:
        c = summary.get(correct_key, 0)
        a = summary.get(acc_key, 0)
        print(f"  {label:<30} {c:>8} {n:>8} {a:>10.1%}")

    print(f"\n  Avg duration per case:   {summary.get('avg_duration_seconds', 0):.1f}s")
    print(f"  Avg follow-up questions: {summary.get('avg_followup_questions', 0):.1f}")

    # ── Per-variant breakdown ────────────────────────────────────────────
    has_variant_data = any(f"variant_{v}_count" in summary for v in ["100", "80", "50"])
    if has_variant_data:
        print(f"\n{'─'*72}")
        print(f"  PER-VARIANT BREAKDOWN")
        print(f"{'─'*72}")
        print(f"  {'Variant':<12} {'Count':>6} {'P1 Top-1':>10} {'P2 Top-1':>10}")
        print(f"  {'─'*40}")
        for var_key in ["100", "80", "50"]:
            count_key = f"variant_{var_key}_count"
            if count_key not in summary:
                continue
            cnt = summary[count_key]
            p1 = summary.get(f"variant_{var_key}_phase1_top1_acc", 0)
            p2 = summary.get(f"variant_{var_key}_phase2_top1_acc", 0)
            print(f"  {var_key + '%':<12} {cnt:>6} {p1:>10.1%} {p2:>10.1%}")

    # ── Per-prognosis breakdown ──────────────────────────────────────────
    per_prog = summary.get("per_prognosis", {})
    if per_prog:
        print(f"\n{'─'*72}")
        print(f"  PER-PROGNOSIS BREAKDOWN")
        print(f"{'─'*72}")
        print(f"  {'Prognosis':<28} {'N':>4} {'P1 Top-1':>10} {'P2 Top-1':>10}")
        print(f"  {'─'*54}")
        for prog, stats in sorted(per_prog.items()):
            n_prog = stats["count"]
            p1a = stats.get("phase1_top1_acc", 0)
            p2a = stats.get("phase2_top1_acc", 0)
            print(f"  {prog:<28} {n_prog:>4} {p1a:>10.1%} {p2a:>10.1%}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    valid_cases = [c for c in cases if not c.get("error")]
    if valid_cases:
        cm = build_confusion_matrix(valid_cases, phase="phase2")
        print(f"\n{'─'*72}")
        print(f"  CONFUSION MATRIX (Phase 2 Top-1)")
        print(f"{'─'*72}")
        print(f"  {'Actual':<28} → {'Predicted (top counts)'}")
        print(f"  {'─'*60}")
        for actual, preds in sorted(cm.items()):
            sorted_preds = sorted(preds.items(), key=lambda x: -x[1])
            pred_str = ", ".join(f"{p}({c})" for p, c in sorted_preds[:3])
            print(f"  {actual:<28} → {pred_str}")

    # ── Failure analysis ─────────────────────────────────────────────────
    failures = failure_analysis(valid_cases, phase="phase2")
    if failures:
        print(f"\n{'─'*72}")
        print(f"  FAILURE ANALYSIS (expected NOT in Phase 2 top-3): {len(failures)} cases")
        print(f"{'─'*72}")
        for f in failures[:20]:  # Cap display at 20
            print(f"  Case #{f['case_id']} ({f['variant']}%)")
            print(f"    Expected:  {f['expected']}")
            print(f"    Got top-1: {f['predicted_top1']}")
            print(f"    Got top-3: {f['predicted_top3']}")
            print()
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more failures (see JSON for full list)")

    print(f"\n{'═'*72}\n")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(report: dict[str, Any], output_path: str | Path) -> None:
    """Export per-case results to a CSV file for spreadsheet analysis."""
    cases = report.get("cases", [])
    if not cases:
        print("[Report] No cases to export.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "variant",
        "expected_prognosis",
        "phase1_top1",
        "top1_correct_phase1",
        "phase1_top3",
        "top3_correct_phase1",
        "phase2_top1",
        "top1_correct_phase2",
        "phase2_top3",
        "top3_correct_phase2",
        "followup_questions_count",
        "duration_seconds",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in cases:
            row = dict(c)
            # Flatten lists to strings for CSV
            row["phase1_top3"] = " | ".join(c.get("phase1_top3", []))
            row["phase2_top3"] = " | ".join(c.get("phase2_top3", []))
            writer.writerow(row)

    print(f"[Report] CSV exported to: {output_path}")


# ── HTML export ───────────────────────────────────────────────────────────────

def export_html(report: dict[str, Any], output_path: str | Path) -> None:
    """Export a self-contained HTML report."""
    meta = report.get("meta", {})
    summary = report.get("summary", {})
    cases = report.get("cases", [])
    valid_cases = [c for c in cases if not c.get("error")]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = build_confusion_matrix(valid_cases, phase="phase2")
    failures = failure_analysis(valid_cases, phase="phase2")

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>HospitalMAS Evaluation Report</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "         max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }",
        "  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }",
        "  h2 { color: #34495e; margin-top: 2rem; }",
        "  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }",
        "  th, td { border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; font-size: 0.9rem; }",
        "  th { background: #f8f9fa; font-weight: 600; }",
        "  tr:nth-child(even) { background: #fdfdfd; }",
        "  .pass { color: #27ae60; font-weight: 600; }",
        "  .fail { color: #e74c3c; font-weight: 600; }",
        "  .metric-card { display: inline-block; background: #f8f9fa; border-radius: 8px;",
        "                  padding: 1rem 1.5rem; margin: 0.5rem; text-align: center; min-width: 150px; }",
        "  .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c3e50; }",
        "  .metric-label { font-size: 0.85rem; color: #7f8c8d; margin-top: 0.25rem; }",
        "</style></head><body>",
        "<h1>HospitalMAS Evaluation Report</h1>",
        f"<p><strong>Timestamp:</strong> {meta.get('timestamp', 'N/A')} · "
        f"<strong>Cases:</strong> {meta.get('total_cases', 0)} · "
        f"<strong>Errors:</strong> {meta.get('errors', 0)} · "
        f"<strong>Variant:</strong> {meta.get('variant_filter') or 'all'}</p>",
    ]

    # Metric cards
    html_parts.append("<h2>Overall Accuracy</h2>")
    for label, key in [
        ("P1 Top-1", "phase1_top1_accuracy"),
        ("P1 Top-3", "phase1_top3_accuracy"),
        ("P2 Top-1", "phase2_top1_accuracy"),
        ("P2 Top-3", "phase2_top3_accuracy"),
    ]:
        val = summary.get(key, 0)
        html_parts.append(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val:.1%}</div>"
            f"<div class='metric-label'>{label}</div></div>"
        )

    # Per-variant table
    has_variant = any(f"variant_{v}_count" in summary for v in ["100", "80", "50"])
    if has_variant:
        html_parts.append("<h2>Per-Variant Breakdown</h2>")
        html_parts.append("<table><tr><th>Variant</th><th>Count</th><th>P1 Top-1</th><th>P2 Top-1</th></tr>")
        for v in ["100", "80", "50"]:
            ck = f"variant_{v}_count"
            if ck not in summary:
                continue
            html_parts.append(
                f"<tr><td>{v}%</td><td>{summary[ck]}</td>"
                f"<td>{summary.get(f'variant_{v}_phase1_top1_acc', 0):.1%}</td>"
                f"<td>{summary.get(f'variant_{v}_phase2_top1_acc', 0):.1%}</td></tr>"
            )
        html_parts.append("</table>")

    # Per-prognosis table
    per_prog = summary.get("per_prognosis", {})
    if per_prog:
        html_parts.append("<h2>Per-Prognosis Breakdown</h2>")
        html_parts.append("<table><tr><th>Prognosis</th><th>N</th><th>P1 Top-1</th><th>P2 Top-1</th></tr>")
        for prog, stats in sorted(per_prog.items()):
            html_parts.append(
                f"<tr><td>{prog}</td><td>{stats['count']}</td>"
                f"<td>{stats.get('phase1_top1_acc', 0):.1%}</td>"
                f"<td>{stats.get('phase2_top1_acc', 0):.1%}</td></tr>"
            )
        html_parts.append("</table>")

    # Confusion matrix
    if cm:
        html_parts.append("<h2>Confusion Matrix (Phase 2 Top-1)</h2>")
        html_parts.append("<table><tr><th>Actual</th><th>Predicted (top counts)</th></tr>")
        for actual, preds in sorted(cm.items()):
            sorted_preds = sorted(preds.items(), key=lambda x: -x[1])
            pred_str = ", ".join(f"{p} ({c})" for p, c in sorted_preds[:4])
            html_parts.append(f"<tr><td>{actual}</td><td>{pred_str}</td></tr>")
        html_parts.append("</table>")

    # Per-case results
    html_parts.append("<h2>Per-Case Results</h2>")
    html_parts.append(
        "<table><tr><th>#</th><th>Var</th><th>Expected</th>"
        "<th>P1 Top-1</th><th>P1 ✓</th><th>P2 Top-1</th><th>P2 ✓</th>"
        "<th>FU Qs</th><th>Time</th></tr>"
    )
    for c in cases:
        p1_cls = "pass" if c.get("top1_correct_phase1") else "fail"
        p2_cls = "pass" if c.get("top1_correct_phase2") else "fail"
        err = c.get("error")
        html_parts.append(
            f"<tr><td>{c['case_id']}</td><td>{c['variant']}%</td>"
            f"<td>{c['expected_prognosis']}</td>"
            f"<td>{c.get('phase1_top1') or ('ERR' if err else '—')}</td>"
            f"<td class='{p1_cls}'>{'✓' if c.get('top1_correct_phase1') else '✗'}</td>"
            f"<td>{c.get('phase2_top1') or ('ERR' if err else '—')}</td>"
            f"<td class='{p2_cls}'>{'✓' if c.get('top1_correct_phase2') else '✗'}</td>"
            f"<td>{c.get('followup_questions_count', 0)}</td>"
            f"<td>{c.get('duration_seconds', 0):.1f}s</td></tr>"
        )
    html_parts.append("</table>")

    # Failures
    if failures:
        html_parts.append(f"<h2>Failure Analysis ({len(failures)} cases)</h2>")
        html_parts.append("<table><tr><th>#</th><th>Var</th><th>Expected</th><th>Got Top-1</th><th>Got Top-3</th></tr>")
        for f in failures:
            t3 = ", ".join(f.get("predicted_top3", []))
            html_parts.append(
                f"<tr><td>{f['case_id']}</td><td>{f['variant']}%</td>"
                f"<td>{f['expected']}</td><td>{f.get('predicted_top1') or '—'}</td>"
                f"<td>{t3 or '—'}</td></tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html_parts))

    print(f"[Report] HTML report exported to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HospitalMAS Evaluation Report Generator",
    )
    parser.add_argument(
        "results_json",
        type=str,
        help="Path to the eval results JSON (output of eval_runner.py).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export per-case results to CSV.",
    )
    parser.add_argument(
        "--export-html",
        type=str,
        default=None,
        help="Export a self-contained HTML report.",
    )

    args = parser.parse_args()

    report = load_results(args.results_json)
    print_terminal_report(report)

    if args.export_csv:
        export_csv(report, args.export_csv)
    if args.export_html:
        export_html(report, args.export_html)


if __name__ == "__main__":
    main()