from __future__ import annotations

import json
import re
from typing import Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GraphDbSympSearchToolInput(BaseModel):
    """Input schema for GraphDbSympSearchTool."""

    sparql_query: str = Field(
        ...,
        description="Complete SPARQL query to execute for SYMP term search",
    )


class GraphDbSympSearchTool(BaseTool):
    """Execute an agent-provided SPARQL query and return SYMP candidates."""

    name: str = "graphdb_symp_search"
    description: str = (
        "Execute an agent-authored read-only GraphDB SPARQL query and return "
        "candidate SYMP URIs, labels, and parsed numeric IDs."
    )
    args_schema: Type[BaseModel] = GraphDbSympSearchToolInput

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
        candidates: list[dict[str, str | None]] = []

        for row in bindings:
            symp_uri = (
                row.get("symptom", {}).get("value", "")
                or row.get("symp", {}).get("value", "")
                or row.get("term", {}).get("value", "")
            )
            symp_label = (
                row.get("symptomLabel", {}).get("value", "")
                or row.get("sympLabel", {}).get("value", "")
                or row.get("termLabel", {}).get("value", "")
                or row.get("label", {}).get("value", "")
            )

            parsed_number: str | None = None
            match = re.search(r"SYMP_(\d+)$", symp_uri)
            if match:
                parsed_number = match.group(1)

            if symp_uri or symp_label:
                candidates.append(
                    {
                        "symp_uri": symp_uri,
                        "symp_label": symp_label,
                        "parsed_symp_number": parsed_number,
                    }
                )

        result = {
            "status_code": status_code,
            "executed_query": clean_sparql_query,
            "result_row_count": len(bindings),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        return json.dumps(result, ensure_ascii=True)
