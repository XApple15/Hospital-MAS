#!/usr/bin/env python
import sys
import warnings
import json
from typing import Any

from hospitalmas.crew import Hospitalmas
from hospitalmas.tools.graphdb_symp_search_tool import GraphDbSympSearchTool
from hospitalmas.tools.graphdb_sparql_query_tool import GraphDbSparqlQueryTool
from hospitalmas.tools.graphdb_disease_symptoms_tool import GraphDbDiseaseSymptomsTool

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def _build_runtime_tools() -> list:
   
    return [
        GraphDbSympSearchTool(),
        GraphDbSparqlQueryTool(),
        GraphDbDiseaseSymptomsTool(),
    ]


def _parse_json_from_text(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    try:
        parsed = json.loads(raw_text[start_idx : end_idx + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_rank_and_followup_payloads(crew_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking_payload: dict[str, Any] = {}
    followup_payload: dict[str, Any] = {}

    task_outputs = getattr(crew_result, "tasks_output", []) or []
    for task_output in task_outputs:
        raw = getattr(task_output, "raw", "")
        parsed = _parse_json_from_text(raw)
        if not parsed:
            continue

        if "differential_diagnosis" in parsed:
            ranking_payload = parsed

        if "followup_needed" in parsed and "questions_asked" in parsed:
            followup_payload = parsed

    return ranking_payload, followup_payload


def _ask_followup_questions_sequentially(followup_payload: dict[str, Any]) -> dict[str, Any]:
    followup_needed = bool(followup_payload.get("followup_needed", False))
    questions = followup_payload.get("questions_asked", [])

    if not followup_needed or not isinstance(questions, list) or not questions:
        return followup_payload

    print("\nFollow-up questions (answer one by one):")
    total_questions = len(questions)

    for idx, question_item in enumerate(questions, start=1):
        if not isinstance(question_item, dict):
            continue

        question_text = (question_item.get("question") or "").strip()
        if not question_text:
            symptom_name = str(question_item.get("symptom") or "this symptom").strip()
            question_text = f"Do you experience {symptom_name}?"

        answer = input(f"[{idx}/{total_questions}] {question_text} ").strip()
        question_item["patient_answer"] = answer

    return followup_payload


def _run_refinement_with_answers(
    ranking_payload: dict[str, Any],
    followup_payload_with_answers: dict[str, Any],
) -> dict[str, Any]:
    refiner_agent = Hospitalmas().diagnosis_refiner()

    prompt = (
        "Refine the diagnosis ranking using the initial differential diagnosis and "
        "the sequential follow-up answers. Return ONLY valid JSON with keys "
        "refined_differential_diagnosis, followup_summary, and notes.\n\n"
        f"Initial ranking JSON:\n{json.dumps(ranking_payload, ensure_ascii=True)}\n\n"
        f"Follow-up answers JSON:\n{json.dumps(followup_payload_with_answers, ensure_ascii=True)}"
    )

    refinement_result = refiner_agent.kickoff(prompt)
    refinement_raw = getattr(refinement_result, "raw", str(refinement_result))
    refinement_payload = _parse_json_from_text(refinement_raw)
    if refinement_payload is None:
        return {
            "refined_differential_diagnosis": [],
            "followup_summary": "Refinement agent did not return valid JSON.",
            "notes": refinement_raw,
        }

    return refinement_payload



def run():
    user_message = input("Describe your symptoms: ").strip()
    if not user_message:
        user_message = "I have bone pain."

    inputs = {"user_message": user_message}

    try:
        Hospitalmas.runtime_tools = _build_runtime_tools()
        first_phase_result = Hospitalmas().crew().kickoff(inputs=inputs)

        ranking_payload, followup_payload = _extract_rank_and_followup_payloads(first_phase_result)

        if not followup_payload:
            print("\nNo follow-up payload was produced. Initial output:")
            print(getattr(first_phase_result, "raw", str(first_phase_result)))
            return

        followup_with_answers = _ask_followup_questions_sequentially(followup_payload)
        refined_payload = _run_refinement_with_answers(ranking_payload, followup_with_answers)

        print("\nRefined diagnosis:")
        print(json.dumps(refined_payload, indent=2, ensure_ascii=True))
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


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
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    try:
        Hospitalmas().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


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
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

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
        result = Hospitalmas().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(
            f"An error occurred while running the crew with trigger: {e}"
        )