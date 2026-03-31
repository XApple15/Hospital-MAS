"""
Deterministic scoring module for the HospitalMAS diagnostic pipeline.

Replaces LLM-based arithmetic for both Phase 1 (TF-IDF ranking) and
Phase 2 (weighted refinement). All calculations are done in Python code
to eliminate the math errors observed when LLMs compute log2, sums,
and composite scores.

This module is called from main.py / eval_runner.py AFTER the LLM agents
produce their structured outputs (disease_results, followup data).
"""

from __future__ import annotations

import math
import re
from typing import Any


# ── Phase 1: Deterministic TF-IDF Ranking ────────────────────────────────────

def compute_ranking(disease_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Given the disease_results from the disease_mapper task, compute the
    TF-IDF-inspired differential diagnosis ranking deterministically.

    Args:
        disease_results: list of dicts, each with keys:
            "symptom", "symp_uri", "parsed_symp_number",
            "disease_candidates", "disease_entries", "status"

    Returns:
        A complete ranking payload dict with:
            "differential_diagnosis", "scoring_metadata",
            "total_symptoms_processed", "total_diseases_found",
            "unmapped_symptoms", "notes"
    """
    # ── Step 1: Count disease frequency per symptom ──────────────────
    symptom_disease_counts: dict[str, int] = {}
    symptom_diseases: dict[str, list[str]] = {}
    unmapped_symptoms: list[str] = []
    all_disease_entries: dict[str, list[dict[str, str]]] = {}

    for entry in disease_results:
        symptom = entry.get("symptom", "")
        status = entry.get("status", "")
        candidates = entry.get("disease_candidates", [])
        entries = entry.get("disease_entries", [])

        if status in ("skipped", "unmapped") or not candidates:
            if status in ("skipped", "unmapped"):
                unmapped_symptoms.append(symptom)
            continue

        symptom_disease_counts[symptom] = len(candidates)
        symptom_diseases[symptom] = candidates
        all_disease_entries[symptom] = entries

    # ── Step 2: Compute specificity weight per symptom ───────────────
    specificity_weights: dict[str, float] = {}
    for symptom, count in symptom_disease_counts.items():
        if count == 0:
            continue
        specificity_weights[symptom] = round(1.0 / math.log2(1 + count), 3)

    total_mapped_symptoms = len(specificity_weights)

    # ── Step 3: Build disease scores ─────────────────────────────────
    # Collect all unique diseases and their supporting symptoms
    disease_supporting: dict[str, list[str]] = {}
    disease_entries_map: dict[str, dict[str, str]] = {}

    for symptom, candidates in symptom_diseases.items():
        entries = all_disease_entries.get(symptom, [])
        entry_lookup = {e.get("disease_label", ""): e for e in entries}

        for disease in candidates:
            if disease not in disease_supporting:
                disease_supporting[disease] = []
            disease_supporting[disease].append(symptom)

            if disease not in disease_entries_map and disease in entry_lookup:
                disease_entries_map[disease] = entry_lookup[disease]

    # Compute scores
    disease_scores: list[dict[str, Any]] = []
    for disease, symptoms in disease_supporting.items():
        # Deduplicate symptoms (a disease might appear under same symptom twice)
        unique_symptoms = list(dict.fromkeys(symptoms))
        match_count = len(unique_symptoms)
        raw_score = round(sum(specificity_weights.get(s, 0) for s in unique_symptoms), 3)
        coverage = round(match_count / total_mapped_symptoms, 3) if total_mapped_symptoms > 0 else 0
        composite_score = round(raw_score * (0.5 + 0.5 * coverage), 3)

        supporting = [
            {"symptom": s, "specificity_weight": specificity_weights.get(s, 0)}
            for s in unique_symptoms
        ]

        disease_scores.append({
            "disease": disease,
            "match_count": match_count,
            "raw_score": raw_score,
            "coverage": coverage,
            "composite_score": composite_score,
            "supporting_symptoms": supporting,
            "disease_uri": disease_entries_map.get(disease, {}).get("disease_uri"),
        })

    # ── Step 4: Rank ─────────────────────────────────────────────────
    disease_scores.sort(key=lambda d: (-d["composite_score"], -d["match_count"], d["disease"]))

    # Top 20 only
    top20 = disease_scores[:20]

    # ── Step 5: Assign confidence tiers ──────────────────────────────
    top_score = top20[0]["composite_score"] if top20 else 0
    weak_evidence = top_score < 0.5

    for i, d in enumerate(top20):
        d["rank"] = i + 1
        cs = d["composite_score"]
        mc = d["match_count"]

        if weak_evidence:
            d["confidence"] = "MEDIUM"
        elif cs >= 0.7 * top_score and mc >= 3:
            d["confidence"] = "HIGH"
        elif cs >= 0.4 * top_score and mc >= 2:
            d["confidence"] = "MEDIUM"
        else:
            d["confidence"] = "LOW"

    total_diseases_found = len(disease_supporting)

    notes = "Differential diagnosis computed deterministically with TF-IDF specificity weighting."
    if weak_evidence:
        notes += " Warning: top score < 0.5, all confidences capped at MEDIUM (weak evidence)."

    return {
        "differential_diagnosis": top20,
        "scoring_metadata": {
            "total_mapped_symptoms": total_mapped_symptoms,
            "symptom_disease_counts": {k: v for k, v in sorted(symptom_disease_counts.items())},
            "specificity_weights": {k: v for k, v in sorted(specificity_weights.items())},
        },
        "total_symptoms_processed": len(disease_results),
        "total_diseases_found": total_diseases_found,
        "unmapped_symptoms": unmapped_symptoms,
        "notes": notes,
    }


# ── Phase 2: Deterministic Refinement Scoring ────────────────────────────────

def compute_refinement(
    ranking_payload: dict[str, Any],
    followup_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Given the Phase 1 ranking and the answered follow-up payload,
    compute the refined differential diagnosis deterministically.

    Args:
        ranking_payload: the output of compute_ranking() or the LLM ranking task
        followup_payload: dict with "disease_symptom_profiles" and "questions_asked"
                          (with patient_answer fields filled in)

    Returns:
        A complete refined diagnosis payload dict.
    """
    differential = ranking_payload.get("differential_diagnosis", [])
    profiles = followup_payload.get("disease_symptom_profiles", [])
    questions = followup_payload.get("questions_asked", [])

    # Build profile lookup: disease → set of candidate symptoms (lowercased)
    profile_lookup: dict[str, set[str]] = {}
    for profile in profiles:
        if profile.get("status") != "profiled":
            continue
        disease = profile.get("disease", "")
        symptoms = {s.strip().lower() for s in profile.get("candidate_symptoms", [])}
        profile_lookup[disease] = symptoms

    n_profiled = len(profile_lookup)

    # ── Step 1: Compute discriminative weight for each follow-up symptom ──
    discriminative_weights: dict[str, float] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        symptom = q.get("symptom", "").strip()
        if not symptom:
            continue

        symptom_lower = symptom.lower()
        diseases_with_symptom = sum(
            1 for syms in profile_lookup.values()
            if symptom_lower in syms
        )

        if diseases_with_symptom == 0 or n_profiled == 0:
            discriminative_weights[symptom] = 0.0
        else:
            discriminative_weights[symptom] = round(
                1.0 - (diseases_with_symptom / n_profiled), 3
            )

    # ── Step 2: Compute adjustments for each disease ─────────────────
    refined_diseases: list[dict[str, Any]] = []

    for d in differential:
        disease_name = d.get("disease", "")
        baseline = d.get("composite_score", 0.0)
        match_count = d.get("match_count", 0)
        original_rank = d.get("rank", 999)

        # Get this disease's symptom profile (lowercased)
        disease_symptoms = profile_lookup.get(disease_name, set())

        support_delta = 0.0
        conflict_delta = 0.0
        confirmed_symptoms: list[dict[str, float]] = []
        denied_symptoms: list[dict[str, float]] = []

        for q in questions:
            if not isinstance(q, dict):
                continue
            symptom = q.get("symptom", "").strip()
            answer = (q.get("patient_answer") or "").strip().lower()
            w = discriminative_weights.get(symptom, 0.0)

            symptom_lower = symptom.lower()

            if answer.startswith("yes"):
                if symptom_lower in disease_symptoms:
                    support_delta += w
                    confirmed_symptoms.append({"symptom": symptom, "weight": w})

            elif answer.startswith("no"):
                if symptom_lower in disease_symptoms:
                    conflict_delta += w
                    denied_symptoms.append({"symptom": symptom, "weight": w})

        support_delta = round(support_delta, 3)
        conflict_delta = round(conflict_delta, 3)
        final_score = round(baseline + support_delta - conflict_delta, 3)

        refined_diseases.append({
            "disease": disease_name,
            "baseline_score": round(baseline, 3),
            "followup_support": support_delta,
            "followup_conflict": conflict_delta,
            "final_score": final_score,
            "match_count": match_count,
            "original_rank": original_rank,
            "confirmed_symptoms": confirmed_symptoms,
            "denied_symptoms": denied_symptoms,
        })

    # ── Step 3: Re-rank ──────────────────────────────────────────────
    refined_diseases.sort(key=lambda d: (-d["final_score"], -d["match_count"], d["disease"]))

    # ── Step 4: Confidence assignment ────────────────────────────────
    top_score = refined_diseases[0]["final_score"] if refined_diseases else 0
    second_score = refined_diseases[1]["final_score"] if len(refined_diseases) > 1 else 0
    gap = top_score - second_score

    for i, d in enumerate(refined_diseases):
        d["rank"] = i + 1

        if i == 0:
            if gap >= 0.5 * top_score or len(refined_diseases) == 1:
                d["updated_confidence"] = "HIGH"
            elif gap > 0:
                d["updated_confidence"] = "MEDIUM"
            else:
                d["updated_confidence"] = "LOW"
        else:
            if d["final_score"] > 0.5 * top_score:
                d["updated_confidence"] = "MEDIUM"
            else:
                d["updated_confidence"] = "LOW"

        # Build reasoning string
        confirmed_str = ", ".join(
            f"{c['symptom']}({c['weight']})" for c in d["confirmed_symptoms"]
        ) or "none"
        denied_str = ", ".join(
            f"{c['symptom']}({c['weight']})" for c in d["denied_symptoms"]
        ) or "none"
        d["reasoning"] = (
            f"Baseline {d['baseline_score']}. "
            f"Confirmed: {confirmed_str}. "
            f"Denied: {denied_str}. "
            f"Net: +{d['followup_support']} -{d['followup_conflict']}. "
            f"Moved from rank {d['original_rank']} to {d['rank']}."
        )

        # Clean up internal fields
        del d["confirmed_symptoms"]
        del d["denied_symptoms"]
        del d["original_rank"]
        del d["match_count"]

    # Find most impactful answer
    max_weight_symptom = ""
    max_weight = 0.0
    for symptom, w in discriminative_weights.items():
        if w > max_weight:
            max_weight = w
            max_weight_symptom = symptom

    all_neutral = all(
        (q.get("patient_answer") or "").strip().lower() not in ("yes", "no")
        for q in questions if isinstance(q, dict)
    )

    if all_neutral:
        followup_summary = "All follow-up answers were neutral; ranking unchanged from baseline."
        notes = "All follow-up answers were neutral; ranking unchanged from baseline."
    else:
        followup_summary = (
            f"Most impactful follow-up symptom: '{max_weight_symptom}' "
            f"(discriminative weight {max_weight})."
        )
        notes = (
            f"Refinement applied to {len(refined_diseases)} diseases from Phase 1. "
            f"Discriminative weights computed across {n_profiled} profiled diseases."
        )

    return {
        "refined_differential_diagnosis": refined_diseases,
        "discriminative_weights": discriminative_weights,
        "followup_summary": followup_summary,
        "notes": notes,
    }


# ── SYMP URI Deduplication ────────────────────────────────────────────────────

def deduplicate_symp_mappings(symp_mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate symptom mappings by SYMP URI.

    When multiple extracted symptoms map to the same SYMP URI (e.g.
    "yellowish skin" and "yellowing of eyes" both → SYMP_0000539/jaundice),
    keep only the first one to avoid double-counting in disease queries.

    Unmapped symptoms are always kept.

    Args:
        symp_mappings: list of mapping dicts from the symp_mapper task

    Returns:
        Deduplicated list (order preserved, first occurrence wins)
    """
    seen_uris: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for mapping in symp_mappings:
        uri = mapping.get("matched_symp_uri")
        status = mapping.get("status", "")

        if status == "unmapped" or not uri:
            # Always keep unmapped symptoms
            deduped.append(mapping)
            continue

        if uri in seen_uris:
            # Skip duplicate URI — already have a symptom mapping to this
            continue

        seen_uris.add(uri)
        deduped.append(mapping)

    return deduped


# ── Follow-up Symptom Deduplication & Classification ─────────────────────────

# Symptoms that ALWAYS require clinical investigation (not patient-reportable)
_INVESTIGATION_KEYWORDS = {
    "inflammation", "pressure", "saturation", "heart rate", "count",
    "enzyme", "level", "finding", "imaging", "ecg", "ekg", "result",
    "test", "measurement", "biopsy", "culture", "titer", "antibody",
    "antigen", "x-ray", "ct", "mri", "ultrasound", "murmur",
    "thrombocytopenia", "hypotension", "shock", "leukocytosis",
    "leukopenia", "anemia", "elevated", "decreased", "abnormal",
}

# Symptoms that are definitively patient-reportable
_PATIENT_REPORTABLE_COMMON = {
    "fever", "cough", "headache", "pain", "nausea", "vomiting",
    "diarrhea", "bleeding", "rash", "fatigue", "dizziness",
    "sore throat", "chills", "sweating", "itching", "swelling",
    "numbness", "tingling", "weakness", "cramps", "stiffness",
    "backache", "acholic stool", "ascites", "hematuria",
    "hematochezia", "epistaxis", "petechiae", "bloodshot eye",
    "jaundice", "constipation", "loss of appetite",
}


def classify_symptom(symptom: str) -> str:
    """
    Classify a symptom as 'investigation_required' or 'patient_reportable'.

    Uses keyword-based rules that mirror the reasoning the LLM is supposed
    to do, but deterministically.

    Returns:
        "investigation_required" or "patient_reportable"
    """
    symptom_lower = symptom.strip().lower()

    # Check investigation keywords first
    for keyword in _INVESTIGATION_KEYWORDS:
        if keyword in symptom_lower:
            return "investigation_required"

    # Specific known investigation findings
    investigation_patterns = [
        "liver inflammation",
        "hepatomegaly", "splenomegaly",
        "lymphadenopathy",
        "tachycardia", "bradycardia",
        "hypertension", "hypotension",
        "pleural effusion", "pulmonary edema",
        "retinal",
    ]
    for pattern in investigation_patterns:
        if pattern in symptom_lower:
            return "investigation_required"

    return "patient_reportable"


def is_symptom_already_known(
    candidate_symptom: str,
    known_symptoms: list[str],
) -> bool:
    """
    Check if a candidate follow-up symptom overlaps with an already-known
    patient symptom using fuzzy containment matching.

    Handles cases like "fever" matching "high fever", "rash" matching
    "skin rash", etc.
    """
    candidate_norm = re.sub(r"[^a-z0-9 ]", "", candidate_symptom.strip().lower())
    candidate_tokens = set(candidate_norm.split())

    for known in known_symptoms:
        known_norm = re.sub(r"[^a-z0-9 ]", "", known.strip().lower())
        known_tokens = set(known_norm.split())

        # Exact match
        if candidate_norm == known_norm:
            return True

        # Token containment: "fever" tokens ⊂ "high fever" tokens
        if candidate_tokens and known_tokens:
            if candidate_tokens.issubset(known_tokens) or known_tokens.issubset(candidate_tokens):
                return True

    return False


def filter_followup_questions(
    followup_payload: dict[str, Any],
    known_symptoms: list[str],
) -> dict[str, Any]:
    """
    Post-process the followup_interviewer output to:
    1. Remove questions about symptoms already known to the patient (fuzzy match)
    2. Move investigation_required symptoms from questions_asked to investigations_required
    3. Deduplicate questions

    Mutates and returns the followup_payload.
    """
    questions = followup_payload.get("questions_asked", [])
    investigations = followup_payload.get("investigations_required", [])
    if not isinstance(investigations, list):
        investigations = []

    filtered_questions: list[dict[str, Any]] = []
    seen_symptoms: set[str] = set()

    for q in questions:
        if not isinstance(q, dict):
            continue

        symptom = q.get("symptom", "").strip()
        symptom_lower = symptom.lower()

        # Skip duplicates
        if symptom_lower in seen_symptoms:
            continue

        # Skip symptoms already known
        if is_symptom_already_known(symptom, known_symptoms):
            continue

        # Classify
        classification = classify_symptom(symptom)
        if classification == "investigation_required":
            investigations.append({
                "finding": symptom,
                "reason": "Requires clinical measurement, laboratory, or imaging investigation",
            })
            continue

        seen_symptoms.add(symptom_lower)
        filtered_questions.append(q)

    followup_payload["questions_asked"] = filtered_questions
    followup_payload["investigations_required"] = investigations

    return followup_payload