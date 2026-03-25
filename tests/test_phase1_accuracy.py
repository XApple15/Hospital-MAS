"""
test_phase1_accuracy.py — Phase 1 diagnostic accuracy tests.

For each test case in tests/cases/*.json, Phase 1 is run and the following
assertions are checked:

1. OUTPUT SCHEMA        — ranking and follow-up payloads contain all required keys.
2. DISEASE FOUND        — total_diseases_found >= case["min_diseases_found"].
3. EXPECTED DISEASES    — every label in case["expected_diseases"] appears in the
                          differential_diagnosis list (case-insensitive substring).
4. NO SYMPTOM REPEAT    — none of the patient's original symptoms appear in
                          questions_asked.
5. QUESTION FORMAT      — every generated question ends with "?" and has no duplicates.
6. NO INVESTIGATION     — findings requiring equipment must not appear as patient questions.

Running
-------
    uv run pytest tests/test_phase1_accuracy.py -v
    uv run pytest tests/test_phase1_accuracy.py -v -k "flu"
"""

from __future__ import annotations

from typing import Any

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _disease_labels(ranking: dict[str, Any]) -> list[str]:
    return [
        str(entry.get("disease", "")).casefold()
        for entry in ranking.get("differential_diagnosis", [])
        if entry.get("disease")
    ]


def _question_symptoms(followup: dict[str, Any]) -> list[str]:
    return [
        str(q.get("symptom", "")).casefold()
        for q in followup.get("questions_asked", [])
        if q.get("symptom")
    ]


def _matches_any(needle: str, haystack: list[str]) -> bool:
    needle_cf = needle.casefold()
    return any(needle_cf in item or item in needle_cf for item in haystack)


@pytest.fixture
def phase1_result(case: dict, phase1_runner) -> tuple[dict, dict]:
    return phase1_runner(case["user_message"])


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_ranking_has_required_keys(self, phase1_result: tuple[dict, dict]) -> None:
        ranking, _ = phase1_result
        required = {"differential_diagnosis", "total_symptoms_processed", "total_diseases_found"}
        missing = required - ranking.keys()
        assert not missing, f"Ranking payload missing keys: {missing}"

    def test_followup_has_required_keys(self, phase1_result: tuple[dict, dict]) -> None:
        ranking, followup = phase1_result
        if ranking.get("total_diseases_found", 0) <= 1:
            pytest.skip("No follow-up expected when ≤1 disease found")
        required = {"followup_needed", "questions_asked", "investigations_required"}
        missing = required - followup.keys()
        assert not missing, f"Follow-up payload missing keys: {missing}"

    def test_differential_diagnosis_entries_have_required_keys(
        self, phase1_result: tuple[dict, dict]
    ) -> None:
        ranking, _ = phase1_result
        for entry in ranking.get("differential_diagnosis", []):
            for key in ("disease", "co_occurrence_score", "confidence"):
                assert key in entry, f"Entry missing key '{key}': {entry}"


class TestDiseaseDiscovery:
    def test_min_diseases_found(self, case: dict, phase1_result: tuple[dict, dict]) -> None:
        ranking, _ = phase1_result
        found = ranking.get("total_diseases_found", 0)
        minimum = case.get("min_diseases_found", 1)
        assert found >= minimum, (
            f"Expected ≥{minimum} diseases, found {found}.\n"
            f"Message: {case['user_message']}\n"
            f"Differential: {ranking.get('differential_diagnosis', [])}"
        )

    def test_expected_diseases_present(self, case: dict, phase1_result: tuple[dict, dict]) -> None:
        expected = case.get("expected_diseases", [])
        if not expected:
            pytest.skip("No expected_diseases in this test case")
        ranking, _ = phase1_result
        found_labels = _disease_labels(ranking)
        missing = [d for d in expected if not _matches_any(d, found_labels)]
        assert not missing, (
            f"Expected diseases not found: {missing}\n"
            f"Actual diseases: {[e.get('disease') for e in ranking.get('differential_diagnosis', [])]}"
        )

    def test_confidence_tiers_are_valid(self, phase1_result: tuple[dict, dict]) -> None:
        ranking, _ = phase1_result
        valid_tiers = {"HIGH", "MEDIUM", "LOW"}
        for entry in ranking.get("differential_diagnosis", []):
            conf = entry.get("confidence", "")
            assert conf in valid_tiers, f"Invalid confidence tier '{conf}' in: {entry}"


class TestNoSymptomRepetition:
    def test_initial_symptoms_not_in_questions(
        self, case: dict, phase1_result: tuple[dict, dict]
    ) -> None:
        ranking, followup = phase1_result
        if not followup.get("followup_needed"):
            pytest.skip("No follow-up questions generated")
        no_repeat = [s.casefold() for s in case.get("no_repeat_symptoms", [])]
        if not no_repeat:
            pytest.skip("no_repeat_symptoms not specified in this case")
        asked = _question_symptoms(followup)
        repeated = [s for s in no_repeat if _matches_any(s, asked)]
        assert not repeated, (
            f"Initial symptoms re-asked: {repeated}\n"
            f"Questions asked about: {asked}"
        )


class TestQuestionFormat:
    def test_questions_end_with_question_mark(self, phase1_result: tuple[dict, dict]) -> None:
        _, followup = phase1_result
        for q in followup.get("questions_asked", []):
            text = q.get("question", "")
            assert text.endswith("?"), f"Question missing '?': '{text}'"

    def test_no_duplicate_symptom_questions(self, phase1_result: tuple[dict, dict]) -> None:
        _, followup = phase1_result
        symptoms = [
            str(q.get("symptom", "")).casefold()
            for q in followup.get("questions_asked", [])
        ]
        seen: set[str] = set()
        duplicates: list[str] = []
        for s in symptoms:
            if s in seen:
                duplicates.append(s)
            seen.add(s)
        assert not duplicates, f"Duplicate symptom questions: {duplicates}"

    def test_no_investigation_in_questions(self, phase1_result: tuple[dict, dict]) -> None:
        _, followup = phase1_result
        investigation_findings = {
            str(item.get("finding", "")).casefold()
            for item in followup.get("investigations_required", [])
        }
        if not investigation_findings:
            pytest.skip("No investigations_required in this result")
        asked_symptoms = set(_question_symptoms(followup))
        overlap = investigation_findings & asked_symptoms
        assert not overlap, (
            f"Investigation findings asked as patient questions: {overlap}"
        )
