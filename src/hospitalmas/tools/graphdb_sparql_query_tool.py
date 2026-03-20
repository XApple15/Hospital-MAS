from __future__ import annotations

import json
from typing import Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GraphDbSparqlQueryToolInput(BaseModel):
    """Input schema for GraphDbSparqlQueryTool."""

    symp_label: str = Field(
        ...,
        description="SYMP label to query for",
    )


class GraphDbSparqlQueryTool(BaseTool):
    """Execute a query for a SYMP label."""

    name: str = "graphdb_sparql_query"
    description: str = (
        "Execute a GraphDB SPARQL query for a provided SYMP label and return "
        "a JSON payload with disease_count and disease_candidates."
    )
    
    args_schema: Type[BaseModel] = GraphDbSparqlQueryToolInput

    def __init__(self, timeout_seconds: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_seconds = timeout_seconds

    def _run(self, symp_label: str) -> str:
        clean_symp_label = symp_label.strip()

        if not clean_symp_label:
            return "SYMP label is empty"

        clean_query = f"""PREFIX owl: <http://www.w3.org/2002/07/owl#>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        PREFIX doid: <http://purl.obolibrary.org/obo/DOID_>
                        PREFIX ro: <http://purl.obolibrary.org/obo/RO_>
                        PREFIX symp: <http://purl.obolibrary.org/obo/SYMP_>

                        SELECT ?disease ?diseaseLabel
                        WHERE {{
                        ?disease rdfs:subClassOf ?restriction .
                        ?restriction owl:onProperty ro:0002452 .
                        ?restriction owl:someValuesFrom symp:{clean_symp_label} .
                        ?disease rdfs:label ?diseaseLabel .
                        FILTER(LANG(?diseaseLabel) = "en")
                        }}"""

        payload = urlencode({"query": clean_query}).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/sparql-results+json, application/json;q=0.9, */*;q=0.8",
        }
        request = Request("http://localhost:7200/repositories/test1", data=payload, headers=headers, method="POST")

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
            return (
                f"GraphDB request timed out after {self._timeout_seconds}s. "
            )
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

        disease_candidates: list[str] = []
        seen_labels: set[str] = set()
        disease_rows: list[dict[str, str]] = []

        for row in bindings:
            disease_iri = row.get("disease", {}).get("value", "")
            disease_label = row.get("diseaseLabel", {}).get("value", "")

            if disease_label and disease_label not in seen_labels:
                seen_labels.add(disease_label)
                disease_candidates.append(disease_label)

            if disease_iri or disease_label:
                disease_rows.append(
                    {
                        "disease_iri": disease_iri,
                        "disease_label": disease_label,
                    }
                )

        payload = {
            "status_code": status_code,
            "symp_label": clean_symp_label,
            "result_row_count": len(bindings),
            "disease_count": len(disease_candidates),
            "disease_candidates": disease_candidates,
            "disease_rows": disease_rows,
        }
        return json.dumps(payload, ensure_ascii=True)
