"""
Shared pytest fixtures and hooks for the Hospital-MAS test suite.

Cases are loaded at collection time (module import) so pytest_generate_tests
can parametrize tests before any fixtures run.

Caching strategy
----------------
Phase 1 is expensive (multiple LLM calls). A single session-scoped cache
dict (_phase1_cache) is shared by both phase1_runner and full_pipeline_runner
so the pipeline runs exactly ONCE per unique user_message per test session,
regardless of how many test modules or test functions use it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from hospitalmas.answer_collector import ProgrammaticAnswerCollector
from hospitalmas.crew import Hospitalmas
from hospitalmas.main import (
    _build_runtime_tools,
    _run_phase1,
    _run_refine_phase,
    _tee_terminal_to_log,
)

# ── Test log helper ───────────────────────────────────────────────────────────

def _build_test_log_file() -> str:
    from datetime import datetime
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(logs_dir / f"test_{ts}.txt")


# ── Load cases at collection time ─────────────────────────────────────────────

CASES_DIR = Path(__file__).parent / "cases"


def _load_all_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cases.extend(data)
        elif isinstance(data, dict):
            cases.append(data)
    return [c for c in cases if not c.get("skip")]


# Available to test files at import time — used by pytest_generate_tests.
ALL_CASES: list[dict[str, Any]] = _load_all_cases()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize any test that declares a `case` fixture from ALL_CASES."""
    if "case" in metafunc.fixturenames:
        ids = [c.get("id", f"case_{i}") for i, c in enumerate(ALL_CASES)]
        metafunc.parametrize("case", ALL_CASES, ids=ids)


# ── Runtime setup ─────────────────────────────────────────────────────────────

def _setup_runtime() -> None:
    Hospitalmas.runtime_tools = _build_runtime_tools()
    Hospitalmas.runtime_log_file = None


# ── Session-level caches ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def _phase1_cache() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """user_message → (ranking_payload, followup_payload)"""
    return {}


@pytest.fixture(scope="session")
def _phase2_cache() -> dict[str, dict[str, Any]]:
    """(user_message, answers_json) → refined_result"""
    return {}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def phase1_runner(_phase1_cache):
    """
    phase1_runner(user_message) → (ranking_payload, followup_payload)

    Runs Phase 1 and caches the result. Subsequent calls with the same
    user_message return immediately from cache without re-running the pipeline.
    A timestamped log file is written to logs/test_<timestamp>.txt.
    """
    _setup_runtime()
    log_file = _build_test_log_file()
    Hospitalmas.runtime_log_file = log_file
    print(f"\n[Test logging] Full agent trace → {log_file}\n")

    def _run(user_message: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if user_message not in _phase1_cache:
            with _tee_terminal_to_log(log_file):
                _, ranking, followup = _run_phase1(user_message)
            _phase1_cache[user_message] = (ranking, followup)
        return _phase1_cache[user_message]

    return _run


@pytest.fixture(scope="session")
def full_pipeline_runner(_phase1_cache, _phase2_cache):
    """
    full_pipeline_runner(user_message, answers) → final_result

    Phase 1 and Phase 2 are each cached at session scope, keyed by
    (user_message, answers). The entire pipeline runs at most once per
    unique (message, answer set) combination per test session.

    answers: dict[symptom_name, "yes"|"no"|"neutral"]
    Symptoms listed → their value (or "neutral" if blank).
    Symptoms not listed → "no".
    """
    log_file = _build_test_log_file()
    _setup_runtime()

    def _run(
        user_message: str,
        answers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        # ── Phase 1 (cached) ──────────────────────────────────────────
        if user_message not in _phase1_cache:
            with _tee_terminal_to_log(log_file):
                _, ranking, followup = _run_phase1(user_message)
            _phase1_cache[user_message] = (ranking, followup)

        ranking, followup = _phase1_cache[user_message]

        if not ranking:
            return {"error": "No ranking payload from Phase 1"}

        if not followup or not followup.get("followup_needed"):
            return ranking

        # ── Phase 2 (cached by message + answers) ─────────────────────
        answers_key = json.dumps(answers or {}, sort_keys=True)
        cache_key = f"{user_message}|{answers_key}"

        if cache_key not in _phase2_cache:
            normalised = {k: v if v else "neutral" for k, v in (answers or {}).items()}
            collector = ProgrammaticAnswerCollector(
                answers=normalised,
                default_answer="no",
            )
            with _tee_terminal_to_log(log_file):
                followup_with_answers = asyncio.run(collector.collect(followup))
                _phase2_cache[cache_key] = _run_refine_phase(ranking, followup_with_answers)

        return _phase2_cache[cache_key]

    return _run


@pytest.fixture(scope="session")
def all_cases() -> list[dict[str, Any]]:
    """Expose the pre-loaded case list as a fixture for tests that need it."""
    return ALL_CASES
