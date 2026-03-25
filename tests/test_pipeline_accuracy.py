"""
test_pipeline_accuracy.py — Full two-phase pipeline accuracy tests.

For each test case in tests/cases/*.json the complete pipeline is run
(Phase 1 + ProgrammaticAnswerCollector + Phase 2) and the following
assertions are checked:

1. REFINED OUTPUT SCHEMA  — output has all required keys.
2. RANKING PRESERVED      — all Phase 1 diseases still appear after refinement.
3. ANSWER INFLUENCE       — confirmed symptoms don't reduce scores; denied don't raise them.
4. NEUTRAL ANSWERS        — when all answers are neutral, ranked order is unchanged.
5. TOP DISEASE PRESENT    — the top-ranked refined disease matches an expected disease.

Running
-------
    uv run pytest tests/test_pipeline_accuracy.py -v
    uv run pytest tests/test_pipeline_accuracy.py -v -k "flu"
"""

from __future__ import annotations

from typing import Any

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _refined_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result.get("refined_differential_diagnosis", [])


def _disease_labels(entries: list[dict[str, Any]]) -> list[str]:
    return [str(e.get("disease", "")).casefold() for e in entries if e.get("disease")]


def _matches_any(needle: str, haystack: list[str]) -> bool:
    needle_cf = needle.casefold()
    return any(needle_cf in item or item in needle_cf for item in haystack)



@pytest.fixture
def pipeline_result(case: dict, full_pipeline_runner) -> dict[str, Any]:
    return full_pipeline_runner(
        case["user_message"],
        answers=case.get("followup_answers"),
    )


def _phase2_reached(result: dict[str, Any]) -> bool:
    """True when Phase 2 ran (result has refined_differential_diagnosis)."""
    return "refined_differential_diagnosis" in result


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRefinedOutputSchema:
    def test_has_refined_differential_diagnosis(self, pipeline_result: dict[str, Any]) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached (≤1 disease in Phase 1)")
        assert "refined_differential_diagnosis" in pipeline_result

    def test_refined_entries_have_required_keys(self, pipeline_result: dict[str, Any]) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached")
        for entry in _refined_entries(pipeline_result):
            for key in ("rank", "disease", "baseline_score", "final_score", "updated_confidence"):
                assert key in entry, f"Refined entry missing key '{key}': {entry}"

    def test_ranks_are_sequential(self, pipeline_result: dict[str, Any]) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached")
        ranks = [e.get("rank") for e in _refined_entries(pipeline_result)]
        assert ranks == list(range(1, len(ranks) + 1)), f"Ranks not sequential: {ranks}"


class TestRefinementCorrectness:
    def test_all_phase1_diseases_preserved(
        self, case: dict, pipeline_result: dict[str, Any], phase1_runner
    ) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached")
        ranking, _ = phase1_runner(case["user_message"])
        phase1_diseases = [
            str(e.get("disease", "")).casefold()
            for e in ranking.get("differential_diagnosis", [])
        ]
        refined_diseases = _disease_labels(_refined_entries(pipeline_result))
        missing = [d for d in phase1_diseases if not _matches_any(d, refined_diseases)]
        assert not missing, f"Phase 1 diseases missing from refined output: {missing}"

    def test_confirmed_symptoms_do_not_decrease_score(
        self, case: dict, pipeline_result: dict[str, Any]
    ) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached")
        yes_answers = {k for k, v in case.get("followup_answers", {}).items()
                       if str(v).casefold() == "yes"}
        if not yes_answers:
            pytest.skip("No 'yes' answers in this test case")
        for entry in _refined_entries(pipeline_result):
            if entry.get("followup_support_count", 0) > 0:
                assert entry["final_score"] >= entry["baseline_score"], (
                    f"'{entry['disease']}' had confirmed symptoms but score decreased: "
                    f"baseline={entry['baseline_score']}, final={entry['final_score']}"
                )

    def test_denied_symptoms_do_not_increase_score(
        self, case: dict, pipeline_result: dict[str, Any]
    ) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached")
        no_answers = {k for k, v in case.get("followup_answers", {}).items()
                      if str(v).casefold() == "no"}
        if not no_answers:
            pytest.skip("No 'no' answers in this test case")
        for entry in _refined_entries(pipeline_result):
            if entry.get("followup_conflict_count", 0) > 0:
                assert entry["final_score"] <= entry["baseline_score"], (
                    f"'{entry['disease']}' had denied symptoms but score increased: "
                    f"baseline={entry['baseline_score']}, final={entry['final_score']}"
                )

    def test_neutral_answers_preserve_ranking_order(
        self, case: dict, pipeline_result: dict[str, Any], phase1_runner
    ) -> None:
        if not _phase2_reached(pipeline_result):
            pytest.skip("Phase 2 not reached")
        answers = case.get("followup_answers", {})
        all_neutral = all(
            str(v).casefold() in {"neutral", "unsure", ""}
            for v in answers.values()
        ) if answers else True
        if not all_neutral:
            pytest.skip("Test case has non-neutral answers")
        ranking, _ = phase1_runner(case["user_message"])
        phase1_order = [
            str(e.get("disease", "")).casefold()
            for e in ranking.get("differential_diagnosis", [])
        ]
        refined_order = _disease_labels(_refined_entries(pipeline_result))
        assert phase1_order == refined_order, (
            f"Neutral answers changed ranking order.\n"
            f"Phase 1:  {phase1_order}\n"
            f"Refined: {refined_order}"
        )


class TestTopDiagnosis:
    def test_top_disease_matches_expected(
        self, case: dict, pipeline_result: dict[str, Any]
    ) -> None:
        expected = case.get("expected_diseases", [])
        if not expected:
            pytest.skip("No expected_diseases in this test case")
        if _phase2_reached(pipeline_result):
            entries = _refined_entries(pipeline_result)
            if not entries:
                pytest.fail("No refined entries in output")
            top = str(entries[0].get("disease", "")).casefold()
        else:
            top = str(
                (pipeline_result.get("differential_diagnosis") or [{}])[0].get("disease", "")
            ).casefold()
        assert _matches_any(top, [d.casefold() for d in expected]), (
            f"Top disease '{top}' does not match any expected: {expected}"
        )
