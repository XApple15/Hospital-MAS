#!/usr/bin/env python
"""
Batch evaluation runner for the HospitalMAS diagnostic pipeline.

Loads patient test cases from Patient_Cases.csv, runs each through the
full two-phase diagnostic pipeline (Phase 1 → auto-answered follow-up →
Phase 2 refinement), and writes structured results to a JSON file for
scoring and reporting.

Usage:
    # Run all cases, all variants
    python -m hospitalmas.eval_runner

    # Run only the 100% variant
    python -m hospitalmas.eval_runner --variant 100

    # Run only the 80% variant
    python -m hospitalmas.eval_runner --variant 80

    # Run only the 50% variant
    python -m hospitalmas.eval_runner --variant 50

    # Limit to N cases (useful for quick smoke tests)
    python -m hospitalmas.eval_runner --limit 5

    # Filter by prognosis
    python -m hospitalmas.eval_runner --prognosis "Dengue"

    # Custom CSV path
    python -m hospitalmas.eval_runner --csv /path/to/Patient_Cases.csv

    # Custom output path
    python -m hospitalmas.eval_runner --output results/my_run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from hospitalmas.crew import Hospitalmas
from hospitalmas.tools.graphdb_ontology_query_tool import GraphDbOntologyQueryTool
from hospitalmas.tools.batch_disease_query_tool import BatchDiseaseQueryTool
from hospitalmas.scoring import (
    compute_ranking,
    compute_refinement,
    deduplicate_symp_mappings,
    filter_followup_questions,
)


# ── Self-contained helpers ────────────────────────────────────────────────────

def _build_runtime_tools() -> list:
    return [
        GraphDbOntologyQueryTool(),
        BatchDiseaseQueryTool(),
    ]


def _build_runtime_log_file() -> str:
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(logs_dir / f"flow_{ts}.txt")


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a raw string."""
    if not raw:
        return None
    try:
        p = json.loads(raw)
        return p if isinstance(p, dict) else None
    except json.JSONDecodeError:
        pass
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        p = json.loads(raw[s:e + 1])
        return p if isinstance(p, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_phase1_payloads(
    crew_result: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Walk task outputs and separate:
      symp_payload     — has 'symp_mappings'
      disease_payload  — has 'disease_results'
      followup_payload — has 'followup_needed' + 'questions_asked'
    """
    symp: dict[str, Any] = {}
    disease: dict[str, Any] = {}
    followup: dict[str, Any] = {}

    for task_out in getattr(crew_result, "tasks_output", []) or []:
        parsed = _parse_json(getattr(task_out, "raw", ""))
        if not parsed:
            continue
        if "symp_mappings" in parsed:
            symp = parsed
        if "disease_results" in parsed:
            disease = parsed
        if "followup_needed" in parsed and "questions_asked" in parsed:
            followup = parsed

    return symp, disease, followup


# ── Constants ─────────────────────────────────────────────────────────────────

VARIANT_COLUMN_MAP = {
    "100": "100% Case",
    "80": "80% Case",
    "50": "50% Case",
}

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "Patient_Cases.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "eval_results"


# ── Simulated patient: auto-answer follow-up questions ────────────────────────

def _symptom_matches(question_symptom: str, known_symptom: str) -> bool:
    """
    Check whether a follow-up question symptom matches one of the known
    ground-truth symptoms from the CSV.  Uses normalised fuzzy matching
    to bridge ontology labels vs CSV underscore-separated names.
    """
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.strip().lower())

    if _norm(question_symptom) == _norm(known_symptom):
        return True

    q_tokens = set(re.split(r"[_\s\-]+", question_symptom.strip().lower()))
    k_tokens = set(re.split(r"[_\s\-]+", known_symptom.strip().lower()))
    if q_tokens and k_tokens and q_tokens == k_tokens:
        return True

    if q_tokens and k_tokens:
        if q_tokens.issubset(k_tokens) or k_tokens.issubset(q_tokens):
            return True

    return False


def _auto_answer_followup(
    followup_payload: dict[str, Any],
    known_symptoms: list[str],
) -> dict[str, Any]:
    """
    Simulate a patient answering follow-up questions.
    For each question, check if the symptom matches any known ground-truth
    symptoms. If yes → "yes", otherwise → "no".
    """
    if not followup_payload.get("followup_needed", False):
        return followup_payload

    questions: list[dict] = followup_payload.get("questions_asked", [])
    for q in questions:
        if not isinstance(q, dict):
            continue
        if (q.get("patient_answer") or "").strip():
            continue

        symptom = str(q.get("symptom", "")).strip()
        matched = any(
            _symptom_matches(symptom, ks) for ks in known_symptoms
        )
        q["patient_answer"] = "yes" if matched else "no"

    return followup_payload


# ── Single-case evaluation ────────────────────────────────────────────────────

def _evaluate_single_case(
    case_id: int,
    user_message: str,
    expected_prognosis: str,
    known_symptoms: list[str],
    variant: str,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run one patient case through the full pipeline and return a result dict.
    """
    import time

    result: dict[str, Any] = {
        "case_id": case_id,
        "variant": variant,
        "expected_prognosis": expected_prognosis,
        "known_symptoms": known_symptoms,
        "phase1_top1": None,
        "phase1_top3": [],
        "phase1_ranking": [],
        "phase2_top1": None,
        "phase2_top3": [],
        "phase2_ranking": [],
        "followup_questions_count": 0,
        "followup_auto_answers": {},
        "top1_correct_phase1": False,
        "top3_correct_phase1": False,
        "top1_correct_phase2": False,
        "top3_correct_phase2": False,
        "error": None,
        "duration_seconds": 0.0,
    }

    start = time.time()

    try:
        # ── Set up runtime tools and log file ─────────────────────────────
        Hospitalmas.runtime_tools = _build_runtime_tools()
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            Hospitalmas.runtime_log_file = str(
                log_dir / f"eval_case{case_id}_{variant}pct_{ts}.txt"
            )
        else:
            Hospitalmas.runtime_log_file = _build_runtime_log_file()

        # ── Phase 1: Run crew (sequential, no orchestrator) ──────────────
        phase1_result = Hospitalmas().diagnostic_crew().kickoff(
            inputs={"user_message": user_message}
        )
        symp_payload, disease_payload, followup_payload = _extract_phase1_payloads(phase1_result)

        # ── (d) Deduplicate SYMP URIs ────────────────────────────────────
        symp_mappings = symp_payload.get("symp_mappings", [])
        deduped_mappings = deduplicate_symp_mappings(symp_mappings)
        deduped_uris = {
            m.get("matched_symp_uri")
            for m in deduped_mappings
            if m.get("matched_symp_uri")
        }

        disease_results = disease_payload.get("disease_results", [])
        filtered_disease_results = []
        for entry in disease_results:
            uri = entry.get("symp_uri")
            status = entry.get("status", "")
            if status in ("skipped", "unmapped") or uri in deduped_uris:
                filtered_disease_results.append(entry)

        # ── (b) Compute ranking deterministically ────────────────────────
        ranking_payload = compute_ranking(filtered_disease_results)

        if not ranking_payload or not ranking_payload.get("differential_diagnosis"):
            result["error"] = "No ranking payload: no diseases found"
            result["duration_seconds"] = time.time() - start
            return result

        # Extract Phase 1 rankings
        diff_diag = ranking_payload.get("differential_diagnosis", [])
        phase1_diseases = [d.get("disease", "") for d in diff_diag]

        result["phase1_ranking"] = diff_diag
        result["phase1_top1"] = phase1_diseases[0] if phase1_diseases else None
        result["phase1_top3"] = phase1_diseases[:3]
        result["top1_correct_phase1"] = _disease_match(
            result["phase1_top1"], expected_prognosis
        )
        result["top3_correct_phase1"] = any(
            _disease_match(d, expected_prognosis) for d in result["phase1_top3"]
        )

        # ── (c) Filter follow-up questions ───────────────────────────────
        extracted_symptoms = [s.get("name", "") for s in symp_payload.get("symptoms", [])]
        mapped_labels = [
            m.get("matched_symp_label", "")
            for m in symp_mappings
            if m.get("matched_symp_label")
        ]
        all_known = extracted_symptoms + mapped_labels

        if followup_payload.get("followup_needed", False):
            followup_payload = filter_followup_questions(followup_payload, all_known)

        # ── Check if follow-up is needed ─────────────────────────────────
        total_diseases = ranking_payload.get("total_diseases_found", 0)
        if total_diseases <= 1 or not followup_payload.get("followup_needed", False):
            result["phase2_top1"] = result["phase1_top1"]
            result["phase2_top3"] = result["phase1_top3"]
            result["phase2_ranking"] = result["phase1_ranking"]
            result["top1_correct_phase2"] = result["top1_correct_phase1"]
            result["top3_correct_phase2"] = result["top3_correct_phase1"]
            result["duration_seconds"] = time.time() - start
            return result

        # ── Auto-answer follow-up questions ──────────────────────────────
        followup_with_answers = _auto_answer_followup(
            followup_payload, known_symptoms
        )
        questions = followup_with_answers.get("questions_asked", [])
        result["followup_questions_count"] = len(questions)
        result["followup_auto_answers"] = {
            q.get("symptom", f"q{i}"): q.get("patient_answer", "")
            for i, q in enumerate(questions)
            if isinstance(q, dict)
        }

        # ── (b) Phase 2: Deterministic refinement ────────────────────────
        refined = compute_refinement(ranking_payload, followup_with_answers)

        refined_diag = refined.get("refined_differential_diagnosis", [])
        phase2_diseases = [d.get("disease", "") for d in refined_diag]

        result["phase2_ranking"] = refined_diag
        result["phase2_top1"] = phase2_diseases[0] if phase2_diseases else None
        result["phase2_top3"] = phase2_diseases[:3]
        result["top1_correct_phase2"] = _disease_match(
            result["phase2_top1"], expected_prognosis
        )
        result["top3_correct_phase2"] = any(
            _disease_match(d, expected_prognosis) for d in result["phase2_top3"]
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    result["duration_seconds"] = time.time() - start
    return result


def _disease_match(predicted: str | None, expected: str) -> bool:
    """
    Fuzzy case-insensitive match between predicted disease and expected prognosis.
    """
    if not predicted:
        return False
    norm_pred = re.sub(r"[^a-z0-9]", "", predicted.strip().lower())
    norm_exp = re.sub(r"[^a-z0-9]", "", expected.strip().lower())
    if norm_pred == norm_exp:
        return True
    if norm_pred in norm_exp or norm_exp in norm_pred:
        return True
    return False


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_test_cases(
    csv_path: str | Path,
    variant_filter: str | None = None,
    prognosis_filter: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and prepare test cases from Patient_Cases.csv."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    variants = (
        {variant_filter: VARIANT_COLUMN_MAP[variant_filter]}
        if variant_filter and variant_filter in VARIANT_COLUMN_MAP
        else VARIANT_COLUMN_MAP
    )

    cases: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            prognosis = row.get("prognosis", "").strip()
            if not prognosis:
                continue

            if prognosis_filter:
                if prognosis_filter.lower() not in prognosis.lower():
                    continue

            raw_symptoms = row.get("symptoms", "")
            known_symptoms = [
                s.strip() for s in raw_symptoms.split(",") if s.strip()
            ]

            for var_key, var_col in variants.items():
                user_message = (row.get(var_col) or "").strip()
                if not user_message:
                    continue
                cases.append({
                    "case_id": row_idx + 1,
                    "variant": var_key,
                    "expected_prognosis": prognosis,
                    "known_symptoms": known_symptoms,
                    "user_message": user_message,
                })

                if limit and len(cases) >= limit:
                    return cases

    return cases


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_evaluation(
    csv_path: str | Path = DEFAULT_CSV,
    variant_filter: str | None = None,
    prognosis_filter: str | None = None,
    limit: int | None = None,
    output_path: str | Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full batch evaluation and write results to JSON."""
    cases = load_test_cases(csv_path, variant_filter, prognosis_filter, limit)
    total = len(cases)

    if total == 0:
        print("[Eval] No test cases matched the filters. Exiting.")
        return {"cases": [], "summary": {}, "error": "No matching cases"}

    print(f"\n{'='*70}")
    print(f"  HospitalMAS Batch Evaluation")
    print(f"  Cases to run: {total}")
    print(f"  Variant filter: {variant_filter or 'all (100%, 80%, 50%)'}")
    print(f"  Prognosis filter: {prognosis_filter or 'none'}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    results: list[dict[str, Any]] = []
    errors = 0

    for idx, case in enumerate(cases, start=1):
        case_label = (
            f"[{idx}/{total}] "
            f"Case #{case['case_id']} ({case['variant']}%) — "
            f"Expected: {case['expected_prognosis']}"
        )
        print(f"\n{case_label}")
        print(f"{'─'*60}")

        res = _evaluate_single_case(
            case_id=case["case_id"],
            user_message=case["user_message"],
            expected_prognosis=case["expected_prognosis"],
            known_symptoms=case["known_symptoms"],
            variant=case["variant"],
            log_dir=log_dir,
        )
        results.append(res)

        if res["error"]:
            errors += 1
            print(f"  ✗ ERROR: {res['error'][:120]}")
        else:
            p1 = "✓" if res["top1_correct_phase1"] else "✗"
            p2 = "✓" if res["top1_correct_phase2"] else "✗"
            print(f"  Phase 1 top-1: {res['phase1_top1']} {p1}")
            print(f"  Phase 2 top-1: {res['phase2_top1']} {p2}")
            print(f"  Follow-up Qs:  {res['followup_questions_count']}")
            print(f"  Duration:      {res['duration_seconds']:.1f}s")

    # ── Build summary ────────────────────────────────────────────────────
    summary = _build_summary(results)

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "csv_path": str(csv_path),
            "variant_filter": variant_filter,
            "prognosis_filter": prognosis_filter,
            "total_cases": total,
            "errors": errors,
        },
        "summary": summary,
        "cases": results,
    }

    # ── Write output ─────────────────────────────────────────────────────
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        var_tag = f"_{variant_filter}pct" if variant_filter else "_all"
        output_path = DEFAULT_OUTPUT_DIR / f"eval{var_tag}_{ts}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  Evaluation complete — results saved to: {output_path}")
    _print_summary(summary)
    print(f"{'='*70}\n")

    return report


# ── Summary statistics ────────────────────────────────────────────────────────

def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy metrics from evaluation results."""
    valid = [r for r in results if r.get("error") is None]
    total_valid = len(valid)

    if total_valid == 0:
        return {
            "total_evaluated": 0,
            "total_errors": len(results),
            "phase1_top1_accuracy": 0.0,
            "phase1_top3_accuracy": 0.0,
            "phase2_top1_accuracy": 0.0,
            "phase2_top3_accuracy": 0.0,
        }

    p1_top1 = sum(1 for r in valid if r["top1_correct_phase1"])
    p1_top3 = sum(1 for r in valid if r["top3_correct_phase1"])
    p2_top1 = sum(1 for r in valid if r["top1_correct_phase2"])
    p2_top3 = sum(1 for r in valid if r["top3_correct_phase2"])

    summary: dict[str, Any] = {
        "total_evaluated": total_valid,
        "total_errors": len(results) - total_valid,
        "phase1_top1_accuracy": round(p1_top1 / total_valid, 4),
        "phase1_top3_accuracy": round(p1_top3 / total_valid, 4),
        "phase2_top1_accuracy": round(p2_top1 / total_valid, 4),
        "phase2_top3_accuracy": round(p2_top3 / total_valid, 4),
        "phase1_top1_correct": p1_top1,
        "phase1_top3_correct": p1_top3,
        "phase2_top1_correct": p2_top1,
        "phase2_top3_correct": p2_top3,
        "avg_duration_seconds": round(
            sum(r["duration_seconds"] for r in valid) / total_valid, 2
        ),
        "avg_followup_questions": round(
            sum(r["followup_questions_count"] for r in valid) / total_valid, 2
        ),
    }

    # Per-variant breakdown
    for var_key in ["100", "80", "50"]:
        var_results = [r for r in valid if r["variant"] == var_key]
        if not var_results:
            continue
        n = len(var_results)
        summary[f"variant_{var_key}_count"] = n
        summary[f"variant_{var_key}_phase1_top1_acc"] = round(
            sum(1 for r in var_results if r["top1_correct_phase1"]) / n, 4
        )
        summary[f"variant_{var_key}_phase2_top1_acc"] = round(
            sum(1 for r in var_results if r["top1_correct_phase2"]) / n, 4
        )

    # Per-prognosis breakdown
    prognoses = set(r["expected_prognosis"] for r in valid)
    per_prognosis: dict[str, dict] = {}
    for prog in sorted(prognoses):
        prog_results = [r for r in valid if r["expected_prognosis"] == prog]
        n = len(prog_results)
        per_prognosis[prog] = {
            "count": n,
            "phase1_top1_acc": round(
                sum(1 for r in prog_results if r["top1_correct_phase1"]) / n, 4
            ),
            "phase2_top1_acc": round(
                sum(1 for r in prog_results if r["top1_correct_phase2"]) / n, 4
            ),
        }
    summary["per_prognosis"] = per_prognosis

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n  ── Overall Accuracy ──")
    print(f"  Cases evaluated:       {summary.get('total_evaluated', 0)}")
    print(f"  Errors:                {summary.get('total_errors', 0)}")
    print(f"  Phase 1 Top-1 acc:     {summary.get('phase1_top1_accuracy', 0):.1%}")
    print(f"  Phase 1 Top-3 acc:     {summary.get('phase1_top3_accuracy', 0):.1%}")
    print(f"  Phase 2 Top-1 acc:     {summary.get('phase2_top1_accuracy', 0):.1%}")
    print(f"  Phase 2 Top-3 acc:     {summary.get('phase2_top3_accuracy', 0):.1%}")
    print(f"  Avg duration:          {summary.get('avg_duration_seconds', 0):.1f}s")
    print(f"  Avg follow-up Qs:      {summary.get('avg_followup_questions', 0):.1f}")

    for var_key in ["100", "80", "50"]:
        count_key = f"variant_{var_key}_count"
        if count_key in summary:
            print(f"\n  ── Variant {var_key}% (n={summary[count_key]}) ──")
            print(f"  Phase 1 Top-1 acc:     {summary[f'variant_{var_key}_phase1_top1_acc']:.1%}")
            print(f"  Phase 2 Top-1 acc:     {summary[f'variant_{var_key}_phase2_top1_acc']:.1%}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HospitalMAS Batch Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hospitalmas.eval_runner                        # all cases, all variants
  python -m hospitalmas.eval_runner --variant 100          # only 100%% cases
  python -m hospitalmas.eval_runner --variant 50 --limit 5 # 5 cases, 50%% variant
  python -m hospitalmas.eval_runner --prognosis Dengue     # only Dengue cases
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"Path to Patient_Cases.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["100", "80", "50"],
        default=None,
        help="Run only a specific case variant (100, 80, or 50). Default: all.",
    )
    parser.add_argument(
        "--prognosis",
        type=str,
        default=None,
        help="Filter by expected prognosis (case-insensitive substring match).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of (case, variant) pairs to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. Default: eval_results/eval_<variant>_<timestamp>.json",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for per-case CrewAI log files. Default: logs/",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parents[2] / "logs"

    run_evaluation(
        csv_path=args.csv,
        variant_filter=args.variant,
        prognosis_filter=args.prognosis,
        limit=args.limit,
        output_path=args.output,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()