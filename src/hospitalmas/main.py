#!/usr/bin/env python
import sys
import warnings
import json
import asyncio
import io
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from hospitalmas.crew import Hospitalmas
from hospitalmas.tools.graphdb_ontology_query_tool import GraphDbOntologyQueryTool
from hospitalmas.answer_collector import (
    AnswerCollector,
    TerminalAnswerCollector,
    normalize_answer,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# ── Tool factory ──────────────────────────────────────────────────────────────

def _build_runtime_tools() -> list:
    return [
        GraphDbOntologyQueryTool(),
    ]


def _build_runtime_log_file() -> str:
    """Create a timestamped log file path in logs/ for the current run."""
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(logs_dir / f"flow_{ts}.txt")


class _TeeStream(io.TextIOBase):
    """Mirror writes to multiple streams (terminal + log file)."""

    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextmanager
def _tee_terminal_to_log(log_file_path: str):
    """Duplicate terminal stdout/stderr to a run-specific log file."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_file_path, "a", encoding="utf-8") as log_handle:
        sys.stdout = _TeeStream(original_stdout, log_handle)
        sys.stderr = _TeeStream(original_stderr, log_handle)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# ── JSON extraction helpers ───────────────────────────────────────────────────

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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Walk task outputs and separate:
      ranking_payload  — has 'differential_diagnosis'
      followup_payload — has 'followup_needed' + 'questions_asked'
    """
    ranking: dict[str, Any] = {}
    followup: dict[str, Any] = {}

    for task_out in getattr(crew_result, "tasks_output", []) or []:
        parsed = _parse_json(getattr(task_out, "raw", ""))
        if not parsed:
            continue
        if "differential_diagnosis" in parsed:
            ranking = parsed
        if "followup_needed" in parsed and "questions_asked" in parsed:
            followup = parsed

    return ranking, followup


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def _run_phase1(user_message: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run Phase 1 once and return raw result plus parsed ranking/follow-up payloads."""
    Hospitalmas.runtime_tools = _build_runtime_tools()
    phase1_result = Hospitalmas().diagnostic_crew().kickoff(
        inputs={"user_message": user_message}
    )
    ranking_payload, followup_payload = _extract_phase1_payloads(phase1_result)
    return phase1_result, ranking_payload, followup_payload


# ── Phase 2 ──────────────────────────────────────────────────────────────────

def _run_refine_phase(
    ranking_payload: dict[str, Any],
    followup_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Kick off the refine_crew with the answered follow-up data baked into the
    task description.  Returns the refined diagnosis dict.
    """
    ranking_json = json.dumps(ranking_payload, indent=2, ensure_ascii=True)
    followup_json = json.dumps(followup_payload, indent=2, ensure_ascii=True)

    hospitalmas = Hospitalmas()
    hospitalmas.runtime_tools = _build_runtime_tools()

    result = hospitalmas.refine_crew(ranking_json, followup_json).kickoff()

    raw = getattr(result, "raw", str(result))
    parsed = _parse_json(raw)
    if parsed and "refined_differential_diagnosis" in parsed:
        return parsed

    return {
        "refined_differential_diagnosis": [],
        "followup_summary": "Refiner did not return valid JSON.",
        "notes": raw,
    }


# ── Full pipeline (transport-agnostic) ────────────────────────────────────────

async def run_diagnostic_pipeline(
    user_message: str,
    collector: AnswerCollector,
    log_file: str | None = None,
) -> dict[str, Any]:
    """
    Complete two-phase diagnostic pipeline with pluggable answer collection.

    This is the main integration point. Any frontend (CLI, REST API, WebSocket,
    chatbot, test harness) calls this with the appropriate AnswerCollector.

    Args:
        user_message: Raw patient symptom description.
        collector:    Any AnswerCollector implementation.
        log_file:     Optional path for terminal logging.

    Returns:
        The final diagnosis dict — either refined (if follow-up was done)
        or the initial ranking (if ≤1 disease was found).
    """
    Hospitalmas.runtime_log_file = log_file

    # ── Phase 1: diagnostic pipeline → question generation ────────────
    print("\n[Phase 1] Running diagnostic pipeline...\n")
    phase1_result, ranking_payload, followup_payload = _run_phase1(user_message)

    if not ranking_payload:
        raw = getattr(phase1_result, "raw", str(phase1_result))
        return {
            "error": "No ranking payload found in Phase 1 output.",
            "raw_output": raw,
        }

    total_diseases = ranking_payload.get("total_diseases_found", 0)
    if total_diseases <= 1:
        return ranking_payload

    if not followup_payload:
        return ranking_payload

    # ── Human-in-the-loop: collect patient answers ─────────────────────
    followup_with_answers = await collector.collect(followup_payload)

    # ── Phase 2: refine diagnosis with answers ─────────────────────────
    print("\n[Phase 2] Refining diagnosis...\n")
    refined = _run_refine_phase(ranking_payload, followup_with_answers)

    return refined


# ── Entry points ──────────────────────────────────────────────────────────────

def run():
    """Interactive CLI entry point — original behaviour preserved."""
    user_message = input("Describe your symptoms: ").strip()
    if not user_message:
        user_message = "I have bone pain."

    log_file = _build_runtime_log_file()
    print(f"[Logging] Full terminal output will be saved to: {log_file}")

    collector = TerminalAnswerCollector()

    with _tee_terminal_to_log(log_file):
        result = asyncio.run(
            run_diagnostic_pipeline(user_message, collector, log_file)
        )
        print("\n── Diagnosis result ──────────────────────────────────────────")
        print(json.dumps(result, indent=2, ensure_ascii=True))


def train():
    inputs = {
        "user_message": (
            "I have persistent dry cough for two weeks, chest tightness, "
            "and shortness of breath on exertion."
        )
    }
    try:
        Hospitalmas.runtime_log_file = _build_runtime_log_file()
        Hospitalmas.runtime_tools = _build_runtime_tools()
        Hospitalmas().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}") from e


def replay():
    try:
        Hospitalmas.runtime_log_file = _build_runtime_log_file()
        Hospitalmas().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    inputs = {
        "user_message": (
            "I feel extreme fatigue, muscle weakness, and dizziness since yesterday."
        )
    }
    try:
        Hospitalmas.runtime_log_file = _build_runtime_log_file()
        Hospitalmas.runtime_tools = _build_runtime_tools()
        Hospitalmas().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}") from e


def run_with_trigger():
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided.")
    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "user_message": trigger_payload.get("user_message", ""),
    }
    try:
        Hospitalmas.runtime_log_file = _build_runtime_log_file()
        Hospitalmas.runtime_tools = _build_runtime_tools()
        return Hospitalmas().diagnostic_crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}") from e