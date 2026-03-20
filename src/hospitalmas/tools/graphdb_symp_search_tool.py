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

    symptom_text: str = Field(
        ...,
        description="Free-text symptom name to search in the SYMP ontology labels",
    )


class GraphDbSympSearchTool(BaseTool):
    """Search SYMP terms in GraphDB by symptom text."""

    name: str = "graphdb_symp_search"
    description: str = (
        "Search SYMP terms in GraphDB by symptom text and return candidate "
        "SYMP URIs, labels, and parsed numeric IDs."
    )
    args_schema: Type[BaseModel] = GraphDbSympSearchToolInput

    def __init__(self, timeout_seconds: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_seconds = timeout_seconds

    def _run(self, symptom_text: str) -> str:
        clean_symptom_text = " ".join(symptom_text.split())
        if not clean_symptom_text:
            return "symptom_text is empty"

        escaped_text = clean_symptom_text.replace("\\", "\\\\").replace('"', '\\"')
        clean_query = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT DISTINCT ?symptom ?symptomLabel 
                        WHERE {{
                            ?symptom rdfs:label ?symptomLabel .
                       
                            FILTER (
                                CONTAINS(LCASE(STR(?symptomLabel)), "{escaped_text}"))
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
            symp_uri = row.get("symp", {}).get("value", "")
            symp_label = row.get("sympLabel", {}).get("value", "")

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
            "symptom_text": clean_symptom_text,
            "result_row_count": len(bindings),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        return json.dumps(result, ensure_ascii=True)
