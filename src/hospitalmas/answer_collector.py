"""
Answer collection protocol for the human-in-the-loop diagnostic pipeline.

Decouples question presentation and answer collection from any specific
I/O transport (terminal, REST API, WebSocket, chatbot, test harness).

Usage:
    # CLI (original behaviour)
    collector = TerminalAnswerCollector()

    # Async API (FastAPI / WebSocket / chatbot)
    collector = AsyncAnswerCollector()

    # Programmatic / tests
    collector = ProgrammaticAnswerCollector(answers={"fever": "yes", ...})

    # Then pass to the pipeline
    followup_with_answers = await collector.collect(followup_payload)
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any

# ── Answer normalisation (shared across all collectors) ───────────────────────

_YES_TOKENS = {
    "yes", "y", "yeah", "yep", "yup", "sure", "affirmative", "correct", "true",
    "da", "sigur", "corect",  # Romanian
}
_NO_TOKENS = {
    "no", "n", "nope", "nah", "negative", "false", "never",
    "nu",  # Romanian
}
_UNSURE_TOKENS = {
    "unsure", "maybe", "idk", "unknown", "dontknow", "notsure", "neutral",
}


def normalize_answer(raw: str) -> str:
    """Canonicalise patient input → 'yes' | 'no' | 'neutral'."""
    token = ""
    if raw.strip():
        token = raw.strip().lower().split()[0]
        token = re.sub(r"[^a-z]", "", token)
    if token in _YES_TOKENS:
        return "yes"
    if token in _NO_TOKENS:
        return "no"
    return "neutral"


def build_patient_question(symptom: str, suggested_question: str) -> str:
    """Preserve the agent-generated question; fallback only when missing."""
    clean_symptom = re.sub(r"\s+", " ", symptom.strip()) or "this symptom"
    clean_q = re.sub(r"\s+", " ", (suggested_question or "").strip())

    if not clean_q:
        return f"Do you currently have {clean_symptom}?"

    return clean_q


# ── Shared pre-processing ────────────────────────────────────────────────────

def _prepare_followup(followup_payload: dict[str, Any]) -> tuple[
    list[dict[str, Any]],   # investigations
    list[dict[str, Any]],   # deduplicated unanswered questions
]:
    """
    Extract investigations and deduplicate unanswered questions.
    Pure function — does not mutate the payload.
    """
    investigations = followup_payload.get("investigations_required") or []

    questions: list[dict] = followup_payload.get("questions_asked") or []
    seen: set[str] = set()
    deduped: list[dict] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        key = str(q.get("symptom", "")).strip().casefold()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(q)

    unanswered = [q for q in deduped if not (q.get("patient_answer") or "").strip()]
    return investigations, unanswered


# ── Abstract protocol ─────────────────────────────────────────────────────────

class AnswerCollector(ABC):
    """
    Protocol for collecting patient answers to follow-up questions.

    Subclass this and implement the three hooks:
      - present_investigations(): show lab/imaging requirements
      - ask_question(): present one yes/no question, return the raw answer
      - on_no_followup_needed(): called when follow-up is skipped

    Then call collect() which handles deduplication, normalisation,
    and payload mutation.
    """

    @abstractmethod
    async def present_investigations(
        self, investigations: list[dict[str, Any]]
    ) -> None:
        """Display required clinical investigations (not patient-answerable)."""
        ...

    @abstractmethod
    async def ask_question(
        self,
        index: int,
        total: int,
        symptom: str,
        question_text: str,
    ) -> str:
        """
        Present a single yes/no question and return the patient's raw answer.
        Must return a non-empty string. Implementations should re-prompt if blank.
        """
        ...

    @abstractmethod
    async def on_no_followup_needed(self) -> None:
        """Called when follow-up is unnecessary (≤1 disease candidate)."""
        ...

    async def collect(self, followup_payload: dict[str, Any]) -> dict[str, Any]:
        """
        Main entry point. Orchestrates the full collection flow:
        1. Check if follow-up is needed
        2. Present investigations
        3. Ask each unanswered question
        4. Normalise and store answers
        Returns the mutated followup_payload with patient_answer fields filled.
        """
        if not followup_payload.get("followup_needed", False):
            await self.on_no_followup_needed()
            return followup_payload

        investigations, unanswered = _prepare_followup(followup_payload)

        if investigations:
            await self.present_investigations(investigations)

        if not unanswered:
            return followup_payload

        for idx, q in enumerate(unanswered, start=1):
            symptom = str(q.get("symptom") or "this symptom").strip()
            question_text = build_patient_question(
                symptom, q.get("question", "")
            )
            raw_answer = await self.ask_question(idx, len(unanswered), symptom, question_text)
            q["patient_answer"] = normalize_answer(raw_answer)

        return followup_payload


# ── Implementation 1: Terminal (drop-in replacement for original) ─────────────

class TerminalAnswerCollector(AnswerCollector):
    """Collects answers interactively from stdin. Original CLI behaviour."""

    async def present_investigations(
        self, investigations: list[dict[str, Any]]
    ) -> None:
        print("\n[Clinical measurements/tests required before diagnosis refinement]")
        for item in investigations:
            if not isinstance(item, dict):
                continue
            finding = str(item.get("finding", "")).strip() or "unspecified finding"
            reason = str(item.get("reason", "")).strip() or "requires clinical measurement"
            print(f"  - {finding}: {reason}")
        print()

    async def ask_question(
        self, index: int, total: int, symptom: str, question_text: str
    ) -> str:
        while True:
            raw = input(f"[{index}/{total}] {question_text} ").strip()
            if raw:
                normalised = normalize_answer(raw)
                print(f"        → recorded: {normalised}\n")
                return raw
            print("        Please enter an answer (yes / no / unsure)")

    async def on_no_followup_needed(self) -> None:
        print("\n[No follow-up needed — only one disease candidate found.]\n")


# ── Implementation 2: Async / API-compatible ──────────────────────────────────

class AsyncAnswerCollector(AnswerCollector):
    """
    Collects answers via an async queue — suitable for FastAPI, WebSocket,
    or chatbot integrations where questions and answers arrive asynchronously.

    Workflow:
        1. Pipeline calls collect(), which puts questions into pending_questions.
        2. Your API layer reads from pending_questions and sends them to the client.
        3. Client answers arrive via submit_answer().
        4. collect() resumes and processes the next question.

    Example FastAPI integration:

        collector = AsyncAnswerCollector()

        @app.post("/diagnose")
        async def diagnose(msg: str):
            # Start pipeline in background, it will pause at collect()
            task = asyncio.create_task(run_pipeline(msg, collector))
            # Return questions as they appear
            ...

        @app.post("/answer")
        async def answer(raw: str):
            await collector.submit_answer(raw)
    """

    def __init__(self) -> None:
        self._answer_queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending_questions: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._investigations: list[dict[str, Any]] = []

    @property
    def pending_questions(self) -> asyncio.Queue[dict[str, Any]]:
        """Read from this queue in your API layer to get questions to send."""
        return self._pending_questions

    @property
    def investigations(self) -> list[dict[str, Any]]:
        """Investigations that should be communicated to the frontend."""
        return self._investigations

    async def submit_answer(self, raw_answer: str) -> None:
        """Called by the API layer when a patient answer arrives."""
        await self._answer_queue.put(raw_answer)

    async def present_investigations(
        self, investigations: list[dict[str, Any]]
    ) -> None:
        self._investigations = investigations

    async def ask_question(
        self, index: int, total: int, symptom: str, question_text: str
    ) -> str:
        await self._pending_questions.put({
            "index": index,
            "total": total,
            "symptom": symptom,
            "question": question_text,
        })
        # Block until the API layer calls submit_answer()
        return await self._answer_queue.get()

    async def on_no_followup_needed(self) -> None:
        await self._pending_questions.put({"done": True, "reason": "no_followup_needed"})


# ── Implementation 3: Programmatic / test harness ─────────────────────────────

class ProgrammaticAnswerCollector(AnswerCollector):
    """
    Pre-loaded answers for testing, CI pipelines, and batch processing.
    Answers are looked up by symptom name (case-insensitive).
    Missing symptoms get the configured default_answer.

    Usage:
        collector = ProgrammaticAnswerCollector(
            answers={"fever": "yes", "cough": "no", "rash": "unsure"},
            default_answer="neutral",
        )
        result = asyncio.run(collector.collect(followup_payload))
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default_answer: str = "neutral",
    ) -> None:
        self._answers = {k.casefold(): v for k, v in (answers or {}).items()}
        self._default_answer = default_answer
        self.asked_questions: list[dict[str, Any]] = []
        self.presented_investigations: list[dict[str, Any]] = []

    async def present_investigations(
        self, investigations: list[dict[str, Any]]
    ) -> None:
        self.presented_investigations = investigations

    async def ask_question(
        self, index: int, total: int, symptom: str, question_text: str
    ) -> str:
        self.asked_questions.append({
            "index": index, "total": total,
            "symptom": symptom, "question": question_text,
        })
        return self._answers.get(symptom.casefold(), self._default_answer)

    async def on_no_followup_needed(self) -> None:
        pass