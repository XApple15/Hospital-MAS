#!/usr/bin/env python
import sys
import warnings
import json

from hospitalmas.crew import Hospitalmas
from hospitalmas.tools.graphdb_symp_search_tool import GraphDbSympSearchTool
from hospitalmas.tools.graphdb_sparql_query_tool import GraphDbSparqlQueryTool

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def _build_runtime_tools() -> list:
   
    return [
        GraphDbSympSearchTool(),
        GraphDbSparqlQueryTool(),
    ]



def run():
    inputs = {
        "user_message": (
            "I have bone pain."
        )
    }
    try:
        Hospitalmas.runtime_tools = _build_runtime_tools()
        Hospitalmas().crew().kickoff(inputs=inputs)
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