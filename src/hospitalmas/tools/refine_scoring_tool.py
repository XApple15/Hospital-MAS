"""
Deterministic scoring tool for Phase 2 diagnosis refinement.

Replaces LLM-based arithmetic with exact Python computation to eliminate
calculation errors, inconsistent scores, and meaningless confidence tiers.

Solves:
  - Problem 6: Inconsistent scoring math in Phase 2
  - Problem 7: Arithmetic errors and self-contradictory reasoning
  - Problem 9: All-HIGH confidence tiers (now computed correctly)
"""

from __future__ import annotations

import json
import math
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class RefineScoringToolInput(BaseModel):
    """Input schema for the Phase 2 refinement scoring tool."""

    initial_ranking_json: str = Field(
        ...,
        description=(
            "JSON string of the Phase 1 ranking payload. Must contain "
            "'differential_diagnosis' (list of ranked diseases with composite_score "
            "and match_count) and optionally 'scoring_metadata'."
        ),
    )
    followup_payload_json: str = Field(
        ...,
        description=(
            "JSON string of the follow-up payload. Must contain "
            "'disease_symptom_profiles' (list of profiled diseases with "
            "candidate_symptoms) and 'questions_asked' (list of questions "
            "with symptom and patient_answer fields)."
        ),
    )


class RefineScoring(BaseTool):
    """
    Deterministic Phase 2 scoring: computes discriminative weights,
    support/conflict deltas, final scores, re-ranking, and confidence
    tiers using exact arithmetic.

    The agent calls this tool ONCE, then uses the returned structured
    result to write the reasoning narrative.
    """

    name: str = "refine_scoring"
    description: str = (
        "Deterministic Phase 2 refinement scorer. Computes discriminative "
        "weights for each follow-up symptom, then calculates support_delta "
        "and conflict_delta for every disease based on patient answers. "
        "Returns the re-ranked differential diagnosis with correct final "
        "scores and confidence tiers.\n\n"
        "Call with:\n"
        '  {"initial_ranking_json": "<Phase 1 ranking JSON string>",\n'
        '   "followup_payload_json": "<answered follow-up JSON string>"}\n\n'
        "Returns a JSON object with:\n"
        "  refined_differential_diagnosis — re-ranked list with final_score, "
        "baseline_score, followup_support, followup_conflict, updated_confidence\n"
        "  discriminative_weights — symptom → weight mapping used\n"
        "  per_disease_details — full breakdown of which symptoms were confirmed/"
        "denied and their weights per disease\n"
    )
    args_schema: Type[BaseModel] = RefineScoringToolInput

    def _run(
        self,
        initial_ranking_json: str,
        followup_payload_json: str,
    ) -> str:
        try:
            ranking = _parse_json_safe(initial_ranking_json)
            followup = _parse_json_safe(followup_payload_json)

            result = compute_refinement(ranking, followup)
            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as exc:
            return json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "refined_differential_diagnosis": [],
            })


# ── Pure computation functions (no LLM, no side effects) ─────────────────────

def _parse_json_safe(raw: str) -> dict[str, Any]:
    """Parse JSON from a string, handling common LLM quirks."""
    if not raw:
        raise ValueError("Empty JSON input")
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting first JSON object
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        return json.loads(raw[s:e + 1])
    raise ValueError(f"Cannot parse JSON from input (first 200 chars): {raw[:200]}")


def compute_refinement(
    ranking_payload: dict[str, Any],
    followup_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Full Phase 2 refinement computation.

    Implements the exact algorithm from refine_diagnosis_task:
      Step 1: Compute discriminative weight per follow-up symptom
      Step 2: Compute support_delta and conflict_delta per disease
      Step 3: Re-rank by final_score (ties: match_count desc, then alpha)
      Step 4: Assign confidence tiers
    """
    # ── Extract inputs ────────────────────────────────────────────────
    diff_diag = ranking_payload.get("differential_diagnosis", [])
    profiles = followup_payload.get("disease_symptom_profiles", [])
    questions = followup_payload.get("questions_asked", [])

    # Build profile lookup: disease name (lowered) → set of candidate symptoms (lowered)
    profile_map: dict[str, set[str]] = {}
    for p in profiles:
        if p.get("status") != "profiled":
            continue
        dname = p.get("disease", "").strip().lower()
        symptoms = {s.strip().lower() for s in p.get("candidate_symptoms", [])}
        profile_map[dname] = symptoms

    n_profiled = len(profile_map)

    # ── Step 1: Discriminative weights ────────────────────────────────
    disc_weights: dict[str, float] = {}
    for q in questions:
        symptom = q.get("symptom", "").strip().lower()
        if not symptom:
            continue
        # Count how many profiled diseases have this symptom
        diseases_with = sum(
            1 for syms in profile_map.values() if symptom in syms
        )
        if diseases_with == 0 or n_profiled == 0:
            disc_weights[symptom] = 0.0
        else:
            disc_weights[symptom] = round(1.0 - (diseases_with / n_profiled), 3)

    # ── Step 2: Per-disease adjustments ───────────────────────────────
    per_disease_details: dict[str, dict[str, Any]] = {}
    disease_scores: list[dict[str, Any]] = []

    for d in diff_diag:
        dname = d.get("disease", "").strip()
        dname_lower = dname.lower()
        baseline = float(d.get("composite_score", 0.0))
        match_count = int(d.get("match_count", 0))

        # Get this disease's candidate symptoms from profile
        candidate_syms = profile_map.get(dname_lower, set())

        support_delta = 0.0
        conflict_delta = 0.0
        confirmed: list[dict[str, Any]] = []
        denied: list[dict[str, Any]] = []
        neutral: list[str] = []

        for q in questions:
            symptom = q.get("symptom", "").strip().lower()
            answer = (q.get("patient_answer") or "").strip().lower()
            w = disc_weights.get(symptom, 0.0)

            if answer.startswith("yes"):
                if symptom in candidate_syms:
                    support_delta += w
                    confirmed.append({"symptom": symptom, "weight": w})
            elif answer.startswith("no"):
                if symptom in candidate_syms:
                    conflict_delta += w
                    denied.append({"symptom": symptom, "weight": w})
            else:
                neutral.append(symptom)

        support_delta = round(support_delta, 3)
        conflict_delta = round(conflict_delta, 3)
        final_score = round(baseline + support_delta - conflict_delta, 3)

        per_disease_details[dname] = {
            "baseline_score": round(baseline, 3),
            "confirmed_symptoms": confirmed,
            "denied_symptoms": denied,
            "neutral_symptoms": neutral,
            "support_delta": support_delta,
            "conflict_delta": conflict_delta,
            "final_score": final_score,
        }

        disease_scores.append({
            "disease": dname,
            "match_count": match_count,
            "baseline_score": round(baseline, 3),
            "followup_support": support_delta,
            "followup_conflict": conflict_delta,
            "final_score": final_score,
        })

    # ── Step 3: Re-rank ───────────────────────────────────────────────
    disease_scores.sort(
        key=lambda x: (-x["final_score"], -x["match_count"], x["disease"])
    )

    # ── Step 4: Confidence assignment ─────────────────────────────────
    if disease_scores:
        top_score = disease_scores[0]["final_score"]
        second_score = disease_scores[1]["final_score"] if len(disease_scores) > 1 else 0.0
        gap = top_score - second_score
    else:
        top_score = 0.0
        second_score = 0.0
        gap = 0.0

    refined: list[dict[str, Any]] = []
    for rank_idx, ds in enumerate(disease_scores, start=1):
        if rank_idx == 1:
            # Rank-1 confidence
            if top_score > 0 and gap >= 0.5 * top_score:
                confidence = "HIGH"
            elif gap > 0:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            # Special: only one disease
            if len(disease_scores) == 1:
                confidence = "HIGH"
        else:
            # Non-rank-1 confidence
            if top_score > 0 and ds["final_score"] > 0.5 * top_score:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

        refined.append({
            "rank": rank_idx,
            "disease": ds["disease"],
            "baseline_score": ds["baseline_score"],
            "followup_support": ds["followup_support"],
            "followup_conflict": ds["followup_conflict"],
            "final_score": ds["final_score"],
            "updated_confidence": confidence,
        })

    # ── Build discriminative weights output (readable form) ───────────
    disc_weights_readable = {
        q.get("symptom", "").strip(): disc_weights.get(
            q.get("symptom", "").strip().lower(), 0.0
        )
        for q in questions
        if q.get("symptom", "").strip()
    }

    # ── Find most impactful answer ────────────────────────────────────
    most_impactful = _find_most_impactful(questions, disc_weights, profile_map)

    return {
        "refined_differential_diagnosis": refined,
        "discriminative_weights": disc_weights_readable,
        "per_disease_details": per_disease_details,
        "most_impactful_answer": most_impactful,
        "scoring_metadata": {
            "n_profiled_diseases": n_profiled,
            "n_questions": len(questions),
            "top_score": top_score,
            "second_score": second_score,
            "gap": round(gap, 3),
        },
    }


def _find_most_impactful(
    questions: list[dict],
    disc_weights: dict[str, float],
    profile_map: dict[str, set[str]],
) -> dict[str, Any]:
    """Find the single follow-up answer with the highest impact on ranking."""
    best: dict[str, Any] = {"symptom": "none", "answer": "neutral", "weight": 0.0, "impact": "none"}
    best_weight = 0.0

    for q in questions:
        symptom = q.get("symptom", "").strip().lower()
        answer = (q.get("patient_answer") or "").strip().lower()
        w = disc_weights.get(symptom, 0.0)

        if answer in ("yes", "no") and w > best_weight:
            # Count how many diseases are affected
            affected = sum(1 for syms in profile_map.values() if symptom in syms)
            direction = "supported" if answer == "yes" else "conflicted"
            best = {
                "symptom": q.get("symptom", "").strip(),
                "answer": answer,
                "weight": w,
                "diseases_affected": affected,
                "direction": direction,
            }
            best_weight = w

    return best