#!/usr/bin/env python
import sys
import warnings
import json
import re
from typing import Any

from hospitalmas.crew import Hospitalmas
from hospitalmas.tools.graphdb_symp_search_tool import GraphDbSympSearchTool
from hospitalmas.tools.graphdb_sparql_query_tool import GraphDbSparqlQueryTool
from hospitalmas.tools.graphdb_disease_symptoms_tool import GraphDbDiseaseSymptomsTool

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ── Answer normalisation ──────────────────────────────────────────────────────

_YES_TOKENS = {"yes", "y", "yeah", "yep", "yup", "sure", "affirmative", "correct", "true"}
_NO_TOKENS  = {"no",  "n", "nope", "nah", "negative", "false", "never"}
_UNSURE_TOKENS = {"unsure", "maybe", "idk", "unknown", "dontknow", "notsure", "neutral"}

# Romanian aliases help with local patient inputs.
_YES_TOKENS.update({"da", "sigur", "corect"})
_NO_TOKENS.update({"nu"})


def _normalize_answer(raw: str) -> str:
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


def _build_patient_question(symptom: str, suggested_question: str) -> str:
    """Ensure the asked question is concise, single-symptom, and yes/no friendly."""
    clean_symptom = re.sub(r"\s+", " ", symptom.strip())
    if not clean_symptom:
        clean_symptom = "this symptom"

    clean_q = re.sub(r"\s+", " ", (suggested_question or "").strip())
    if not clean_q:
        return f"Do you currently have {clean_symptom}?"

    # If the generated question appears multi-part, replace with a strict template.
    if clean_q.count("?") > 1 or ", and " in clean_q.lower() or " or " in clean_q.lower():
        return f"Do you currently have {clean_symptom}?"

    if clean_q[-1] != "?":
        clean_q = f"{clean_q}?"

    # Keep wording short and clear for terminal UX.
    if len(clean_q.split()) > 16:
        return f"Do you currently have {clean_symptom}?"

    return clean_q


def _run_phase1(user_message: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run Phase 1 once and return raw result plus parsed ranking/follow-up payloads."""
    Hospitalmas.runtime_tools = _build_runtime_tools()
    phase1_result = Hospitalmas().diagnostic_crew().kickoff(
        inputs={"user_message": user_message}
    )
    ranking_payload, followup_payload = _extract_phase1_payloads(phase1_result)
    return phase1_result, ranking_payload, followup_payload


# ── Tool factory ──────────────────────────────────────────────────────────────

def _build_runtime_tools() -> list:
    return [
        GraphDbSympSearchTool(),
        GraphDbSparqlQueryTool(),
        GraphDbDiseaseSymptomsTool(),
    ]


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
    ranking: dict[str, Any]  = {}
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


# ── Human-in-the-loop question collection ────────────────────────────────────

def _collect_answers(followup_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Present each unanswered follow-up question to the patient at the terminal.
    Normalises every answer before storing it.
    Mutates and returns the followup_payload with patient_answer fields filled.
    """
    if not followup_payload.get("followup_needed", False):
        print("\n[No follow-up needed — only one disease candidate found.]\n")
        return followup_payload

    investigations = followup_payload.get("investigations_required", []) or []
    if investigations:
        print("\n[Clinical measurements/tests required before diagnosis refinement]")
        for item in investigations:
            if not isinstance(item, dict):
                continue
            finding = str(item.get("finding", "")).strip() or "unspecified finding"
            reason = str(item.get("reason", "")).strip() or "requires clinical measurement"
            print(f"- {finding}: {reason}")
        print("")

    questions: list[dict] = followup_payload.get("questions_asked", [])
    deduped_questions: list[dict] = []
    seen_symptoms: set[str] = set()
    for q in questions:
        if not isinstance(q, dict):
            continue
        symptom_key = str(q.get("symptom", "")).strip().casefold()
        if symptom_key and symptom_key in seen_symptoms:
            continue
        if symptom_key:
            seen_symptoms.add(symptom_key)
        deduped_questions.append(q)

    unanswered = [
        q for q in deduped_questions
        if not (q.get("patient_answer") or "").strip()
    ]

    if not unanswered:
        print("\n[All follow-up questions already have answers.]\n")
        return followup_payload

    total = len(unanswered)
    print(f"\n{'─'*60}")
    print(f"  Follow-up questions ({total})")
    print(f"  Answer each with: yes / no / unsure (or da / nu)")
    print(f"{'─'*60}\n")

    for idx, q in enumerate(unanswered, start=1):
        question_text = (q.get("question") or "").strip()
        symptom = str(q.get("symptom") or "this symptom").strip()
        question_text = _build_patient_question(symptom, question_text)

        # Keep prompting until a non-empty answer is given
        while True:
            raw = input(f"[{idx}/{total}] {question_text} ").strip()
            if raw:
                break
            print("        Please enter an answer (yes / no / unsure)")

        normalised = _normalize_answer(raw)
        cleaned = re.sub(r"[^a-z]", "", raw.strip().lower().split()[0]) if raw.strip() else ""
        if normalised == "neutral" and cleaned not in _UNSURE_TOKENS:
            print("        Interpreted as unsure/neutral. Use yes/no for stronger evidence.")
        q["patient_answer"] = normalised
        print(f"        → recorded: {normalised}\n")

    return followup_payload


# ── Phase 2: run the refinement crew ─────────────────────────────────────────

def _run_refine_phase(
    ranking_payload: dict[str, Any],
    followup_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Kick off the refine_crew with the answered follow-up data baked into the
    task description.  Returns the refined diagnosis dict.
    """
    ranking_json  = json.dumps(ranking_payload,  indent=2, ensure_ascii=True)
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


# ── Entry points ──────────────────────────────────────────────────────────────

def run():
    user_message = input("Describe your symptoms: ").strip()
    if not user_message:
        user_message = "I have bone pain."

    # ── Phase 1: diagnostic pipeline → question generation ────────────────
    print("\n[Phase 1] Running diagnostic pipeline...\n")
    phase1_result, ranking_payload, followup_payload = _run_phase1(user_message)

    if not ranking_payload:
        print("\n[Warning] No ranking payload found in Phase 1 output.")
        print(getattr(phase1_result, "raw", str(phase1_result)))
        return

    # If only 0–1 diseases found, skip follow-up entirely
    total_diseases = ranking_payload.get("total_diseases_found", 0)
    if total_diseases <= 1:
        print("\n── Initial diagnosis (no follow-up needed) ──────────────────")
        print(json.dumps(ranking_payload, indent=2, ensure_ascii=True))
        return

    if not followup_payload:
        print("\n[Warning] Ranking found but no follow-up payload. Showing initial ranking.")
        print(json.dumps(ranking_payload, indent=2, ensure_ascii=True))
        return

    # ── Human-in-the-loop: collect patient answers ─────────────────────────
    followup_with_answers = _collect_answers(followup_payload)

    # ── Phase 2: refine diagnosis with answers ─────────────────────────────
    print("\n[Phase 2] Refining diagnosis...\n")
    refined = _run_refine_phase(ranking_payload, followup_with_answers)

    print("\n── Refined diagnosis ─────────────────────────────────────────")
    print(json.dumps(refined, indent=2, ensure_ascii=True))


def train():
    inputs = {
        "user_message": (
            "I have persistent dry cough for two weeks, chest tightness, "
            "and shortness of breath on exertion."
        )
    }
    try:
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
        Hospitalmas.runtime_tools = _build_runtime_tools()
        return Hospitalmas().diagnostic_crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}") from e