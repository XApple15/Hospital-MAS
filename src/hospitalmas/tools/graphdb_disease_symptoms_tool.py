from __future__ import annotations

import json
from typing import Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GraphDbDiseaseSymptomsToolInput(BaseModel):
    """Input schema for GraphDbDiseaseSymptomsTool."""

    disease_uri: str = Field(
        ...,
        description="Full disease URI to expand into related symptoms",
    )


class GraphDbDiseaseSymptomsTool(BaseTool):
    """Given a disease URI, return all linked symptoms from GraphDB."""

    name: str = "graphdb_disease_symptoms"
    description: str = (
        "Fetch all symptoms linked to a disease URI via RO:0002452 (has symptom) "
        "and return symptom URIs and labels."
    )
    args_schema: Type[BaseModel] = GraphDbDiseaseSymptomsToolInput

    def __init__(self, timeout_seconds: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_seconds = timeout_seconds

    def _run(self, disease_uri: str) -> str:
        clean_disease_uri = disease_uri.strip()
        if not clean_disease_uri:
            return "disease_uri is empty"

        sparql_query = f"""
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ro: <http://purl.obolibrary.org/obo/RO_>

SELECT ?symptom ?symptomLabel
WHERE {{
  <{clean_disease_uri}> rdfs:subClassOf ?restriction .
  ?restriction owl:onProperty ro:0002452 .
  ?restriction owl:someValuesFrom ?symptom .
  ?symptom rdfs:label ?symptomLabel .
}}
""".strip()

        payload = urlencode({"query": sparql_query}).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/sparql-results+json, application/json;q=0.9, */*;q=0.8",
        }
        request = Request(
            "http://localhost:7200/repositories/test1",
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                status_code = getattr(response, "status", 200)
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return (
                f"GraphDB request failed with HTTP {exc.code}. "
                f"Endpoint Response: {body}"
            )
        except URLError as exc:
            return (
                "Could not reach GraphDB endpoint. "
                f"Endpoint Error: {exc.reason}"
            )
        except TimeoutError:
            return f"GraphDB request timed out after {self._timeout_seconds}s."
        except Exception as exc:  # pragma: no cover - defensive fallback
            return f"Unexpected error while querying GraphDB: {exc}"

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return (
                f"Query executed (HTTP {status_code}) but response is not JSON. "
                f"Raw response: {raw}"
            )

        bindings = parsed.get("results", {}).get("bindings", [])
        symptoms: list[dict[str, str]] = []
        seen_labels: set[str] = set()

        for row in bindings:
            symptom_uri = row.get("symptom", {}).get("value", "")
            symptom_label = row.get("symptomLabel", {}).get("value", "")

            if symptom_label and symptom_label in seen_labels:
                continue

            if symptom_label:
                seen_labels.add(symptom_label)

            if symptom_uri or symptom_label:
                symptoms.append(
                    {
                        "symptom_uri": symptom_uri,
                        "symptom_label": symptom_label,
                    }
                )

        result = {
            "status_code": status_code,
            "disease_uri": clean_disease_uri,
            "executed_query": sparql_query,
            "result_row_count": len(bindings),
            "symptom_count": len(symptoms),
            "symptoms": symptoms,
        }
        return json.dumps(result, ensure_ascii=True)