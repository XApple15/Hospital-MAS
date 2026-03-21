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

    sparql_query: str = Field(
        ...,
        description="Complete SPARQL query to execute against GraphDB",
    )


class GraphDbSparqlQueryTool(BaseTool):
    """Execute an agent-provided SPARQL query against GraphDB."""

    name: str = "graphdb_sparql_query"
    description: str = (
        "Execute an agent-authored read-only GraphDB SPARQL query and return "
        "a JSON payload with disease_count and disease_candidates."
    )
    
    args_schema: Type[BaseModel] = GraphDbSparqlQueryToolInput

    def __init__(self, timeout_seconds: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_seconds = timeout_seconds

    def _run(self, sparql_query: str) -> str:
        clean_sparql_query = sparql_query.strip()

        if not clean_sparql_query:
            return "sparql_query is empty"

        upper_query = clean_sparql_query.lstrip().upper()
        if not (
            upper_query.startswith("SELECT")
            or upper_query.startswith("PREFIX")
            or upper_query.startswith("ASK")
            or upper_query.startswith("CONSTRUCT")
            or upper_query.startswith("DESCRIBE")
        ):
            return (
                "Only read-only SPARQL queries are allowed. "
                "Expected SELECT/ASK/CONSTRUCT/DESCRIBE (optionally with PREFIX lines)."
            )

        payload = urlencode({"query": clean_sparql_query}).encode("utf-8")
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
            "executed_query": clean_sparql_query,
            "result_row_count": len(bindings),
            "disease_count": len(disease_candidates),
            "disease_candidates": disease_candidates,
            "disease_rows": disease_rows,
        }
        return json.dumps(payload, ensure_ascii=True)
