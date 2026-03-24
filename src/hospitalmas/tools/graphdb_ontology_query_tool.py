from __future__ import annotations

import json
import re
from typing import Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GraphDbOntologyQueryToolInput(BaseModel):
    """Input schema for GraphDbOntologyQueryTool.

    Agents must call this tool with exactly one flat argument object:
    {"sparql_query": "<full query string>"}
    """

    sparql_query: str = Field(
        ...,
        description=(
            "Complete read-only SPARQL query to execute against GraphDB. "
            "Include PREFIX declarations when needed and keep query type read-only "
            "(SELECT/ASK/CONSTRUCT/DESCRIBE)."
        ),
    )


class GraphDbOntologyQueryTool(BaseTool):
    """Single GraphDB ontology query tool for all SYMP/DOID/RO query use cases.

    This tool is the canonical place for call-shape and SPARQL authoring guidance.
    """

    name: str = "graphdb_ontology_query"
    description: str = (
        "Execute an agent-authored read-only SPARQL query in GraphDB and return normalized results. "
        "CALL CONTRACT: pass exactly one flat object {\"sparql_query\": \"<full query>\"}. "
        "\n"
        "ONTOLOGY FOUNDATION:\n"
        "- SYMP = symptom ontology (http://purl.obolibrary.org/obo/SYMP_)\n"
        "- DOID = disease ontology (http://purl.obolibrary.org/obo/DOID_)\n"
        "- RO = relation ontology (http://purl.obolibrary.org/obo/RO_) with ro:0002452 = 'has symptom'\n"
        "\n"
        "REQUIRED PREFIXES (copy exactly):\n"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX ro: <http://purl.obolibrary.org/obo/RO_>\n"
        "\n"
        "★★★ CRITICAL QUERY PATTERNS (copy exactly, do NOT modify or add/remove filters) ★★★\n"
        "\n"
        "PATTERN 1: Symptom by label matching (case-insensitive)\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "SELECT ?symptom ?symptomLabel\n"
        "WHERE {\n"
        "  ?symptom rdfs:label ?symptomLabel .\n"
        "  FILTER(CONTAINS(LCASE(STR(?symptomLabel)), \"<lowercased_term>\"))\n"
        "}\n"
        "\n"
        "PATTERN 2: Fetch ALL symptoms linked to a disease via rdfs:subClassOf + owl:onProperty ro:0002452\n"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX ro: <http://purl.obolibrary.org/obo/RO_>\n"
        "SELECT ?symptom ?symptomLabel\n"
        "WHERE {\n"
        "  <disease_uri> rdfs:subClassOf ?restriction .\n"
        "  ?restriction owl:onProperty ro:0002452 .\n"
        "  ?restriction owl:someValuesFrom ?symptom .\n"
        "  ?symptom rdfs:label ?symptomLabel .\n"
        "}\n"
        "DO NOT ADD FILTERS. Query returns all symptoms with any label.\n"
        "\n"
        "PATTERN 3: Fetch diseases linked to a symptom (inverse traversal)\n"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX ro: <http://purl.obolibrary.org/obo/RO_>\n"
        "SELECT ?disease ?diseaseLabel\n"
        "WHERE {\n"
        "  ?disease rdfs:subClassOf ?restriction .\n"
        "  ?restriction owl:onProperty ro:0002452 .\n"
        "  ?restriction owl:someValuesFrom <symptom_uri> .\n"
        "  ?disease rdfs:label ?diseaseLabel .\n"
        "}\n"
        "\n"
        "★★★ ABSOLUTE RULES ★★★\n"
        "- Copy patterns exactly; do not add FILTER(LANG(...)) or any other filters unless shown above\n"
        "- Do not modify WHERE clause structure\n"
        "- Pattern 2 + 3 have NO language filters; results returned as-is\n"
        "- Replace <disease_uri>, <symptom_uri>, <lowercased_term> with actual values (URIs in angle brackets)\n"
        "- Use tool output ONLY; never invent URIs, labels, or relation IDs\n"
        "- All URIs must come from prior tool results or provided disease_entries\n"
    )
    args_schema: Type[BaseModel] = GraphDbOntologyQueryToolInput

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
        except Exception as exc:  # pragma: no cover
            return f"Unexpected error while querying GraphDB: {exc}"

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return (
                f"Query executed (HTTP {status_code}) but response is not JSON. "
                f"Raw response: {raw}"
            )

        bindings = parsed.get("results", {}).get("bindings", [])

        normalized_rows: list[dict[str, str]] = []
        for row in bindings:
            normalized_row: dict[str, str] = {}
            for key, val in row.items():
                if isinstance(val, dict) and "value" in val:
                    normalized_row[key] = str(val.get("value", ""))
            if normalized_row:
                normalized_rows.append(normalized_row)

        symp_candidates: list[dict[str, str | None]] = []
        seen_symp: set[tuple[str, str]] = set()

        disease_rows: list[dict[str, str]] = []
        seen_disease: set[tuple[str, str]] = set()

        symptom_rows: list[dict[str, str]] = []
        seen_symptom: set[tuple[str, str]] = set()

        for row in normalized_rows:
            symp_uri = row.get("symptom") or row.get("symp") or row.get("term") or ""
            symp_label = row.get("symptomLabel") or row.get("sympLabel") or row.get("termLabel") or row.get("label") or ""
            if symp_uri or symp_label:
                key = (symp_uri, symp_label)
                if key not in seen_symp:
                    seen_symp.add(key)
                    parsed_symp_number = None
                    match = re.search(r"SYMP_(\d+)$", symp_uri)
                    if match:
                        parsed_symp_number = match.group(1)
                    symp_candidates.append(
                        {
                            "symp_uri": symp_uri or None,
                            "symp_label": symp_label or None,
                            "parsed_symp_number": parsed_symp_number,
                        }
                    )

            disease_uri = row.get("disease_uri") or row.get("disease_iri") or row.get("disease") or ""
            disease_label = row.get("disease_label") or row.get("diseaseLabel") or ""
            if disease_uri or disease_label:
                key = (disease_uri, disease_label)
                if key not in seen_disease:
                    seen_disease.add(key)
                    disease_rows.append(
                        {
                            "disease_uri": disease_uri,
                            "disease_label": disease_label,
                        }
                    )

            symptom_uri = row.get("symptom_uri") or row.get("symptom") or ""
            symptom_label = row.get("symptom_label") or row.get("symptomLabel") or ""
            if symptom_uri or symptom_label:
                key = (symptom_uri, symptom_label)
                if key not in seen_symptom:
                    seen_symptom.add(key)
                    symptom_rows.append(
                        {
                            "symptom_uri": symptom_uri,
                            "symptom_label": symptom_label,
                        }
                    )

        disease_candidates = [
            r["disease_label"] for r in disease_rows if r.get("disease_label")
        ]
        symptom_candidates = [
            r["symptom_label"] for r in symptom_rows if r.get("symptom_label")
        ]

        result = {
            "status_code": status_code,
            "executed_query": clean_sparql_query,
            "result_row_count": len(bindings),
            "rows": normalized_rows,
            "symp_candidates": symp_candidates,
            "candidates": symp_candidates,
            "disease_count": len(disease_candidates),
            "disease_candidates": disease_candidates,
            "disease_rows": disease_rows,
            "symptom_count": len(symptom_candidates),
            "symptom_candidates": symptom_candidates,
            "symptom_rows": symptom_rows,
            "symptoms": symptom_rows,
        }
        return json.dumps(result, ensure_ascii=True)
