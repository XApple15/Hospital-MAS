#!/usr/bin/env python
"""
discover_cases.py — Seed test cases from real GraphDB data.

Queries GraphDB to find diseases that have at least --min-symptoms linked
symptoms, then generates a ready-to-use test case JSON file for each one.

Usage
-----
    python tests/discover_cases.py                        # defaults
    python tests/discover_cases.py --min-symptoms 3       # stricter
    python tests/discover_cases.py --limit 20 --out tests/cases/discovered.json

The output file contains a JSON array of test case objects that can be
used directly by test_phase1_accuracy.py and test_pipeline_accuracy.py.

How it works
------------
1. Fetch all diseases that have ≥ N "has symptom" (ro:0002452) restrictions.
2. For each disease, fetch its symptom labels.
3. Build a user_message like "I have <symptom1>, <symptom2>, and <symptom3>."
4. Set expected_diseases to [disease_label] and no_repeat_symptoms to the
   symptom list so the no-repetition assertion is pre-populated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the package is importable when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hospitalmas.tools.graphdb_ontology_query_tool import GraphDbOntologyQueryTool

TOOL = GraphDbOntologyQueryTool()

# ── SPARQL helpers ────────────────────────────────────────────────────────────

DISEASES_WITH_SYMPTOM_COUNT = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ro:   <http://purl.obolibrary.org/obo/RO_>
SELECT ?disease ?diseaseLabel (COUNT(?symptom) AS ?symptomCount)
WHERE {
  ?disease rdfs:subClassOf ?restriction .
  ?restriction owl:onProperty ro:0002452 .
  ?restriction owl:someValuesFrom ?symptom .
  ?disease rdfs:label ?diseaseLabel .
  FILTER(LANG(?diseaseLabel) = "en")
}
GROUP BY ?disease ?diseaseLabel
HAVING (COUNT(?symptom) >= {min_symptoms})
ORDER BY DESC(?symptomCount)
LIMIT {limit}
"""

SYMPTOMS_FOR_DISEASE = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ro:   <http://purl.obolibrary.org/obo/RO_>
SELECT ?symptom ?symptomLabel
WHERE {{
  {disease_uri} rdfs:subClassOf ?restriction .
  ?restriction owl:onProperty ro:0002452 .
  ?restriction owl:someValuesFrom ?symptom .
  ?symptom rdfs:label ?symptomLabel .
}}
"""


def _query(sparql: str) -> list[dict]:
    raw = TOOL._run(sparql)
    result = json.loads(raw)
    return result.get("rows", [])


def _fetch_diseases(min_symptoms: int, limit: int) -> list[dict]:
    sparql = DISEASES_WITH_SYMPTOM_COUNT.format(
        min_symptoms=min_symptoms, limit=limit
    )
    rows = _query(sparql)
    return [
        {
            "uri": r.get("disease", ""),
            "label": r.get("diseaseLabel", ""),
            "symptom_count": int(r.get("symptomCount", 0)),
        }
        for r in rows
        if r.get("disease") and r.get("diseaseLabel")
    ]


def _fetch_symptoms(disease_uri: str) -> list[str]:
    sparql = SYMPTOMS_FOR_DISEASE.format(disease_uri=f"<{disease_uri}>")
    rows = _query(sparql)
    labels = [r.get("symptomLabel", "") for r in rows if r.get("symptomLabel")]
    return sorted(set(labels))


def _build_user_message(symptom_labels: list[str]) -> str:
    if not symptom_labels:
        return ""
    if len(symptom_labels) == 1:
        return f"I have {symptom_labels[0]}."
    body = ", ".join(symptom_labels[:-1]) + f", and {symptom_labels[-1]}"
    return f"I have {body}."


def _build_case(disease: dict, symptoms: list[str], idx: int) -> dict:
    disease_label = disease["label"]
    safe_id = disease_label.lower().replace(" ", "_")[:40]
    return {
        "id": f"{safe_id}_{idx:03d}",
        "description": f"Auto-discovered: {disease_label}",
        "user_message": _build_user_message(symptoms),
        "min_diseases_found": 1,
        "expected_diseases": [disease_label],
        "no_repeat_symptoms": symptoms,
        "followup_answers": {},
        "_meta": {
            "disease_uri": disease["uri"],
            "symptom_count": disease["symptom_count"],
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Discover test cases from GraphDB.")
    parser.add_argument("--min-symptoms", type=int, default=2,
                        help="Minimum linked symptoms a disease must have (default: 2)")
    parser.add_argument("--limit", type=int, default=30,
                        help="Max diseases to retrieve (default: 30)")
    parser.add_argument("--out", type=str,
                        default="tests/cases/discovered.json",
                        help="Output JSON file path (default: tests/cases/discovered.json)")
    args = parser.parse_args()

    print(f"Querying GraphDB for diseases with ≥{args.min_symptoms} symptoms (limit={args.limit})...")
    diseases = _fetch_diseases(args.min_symptoms, args.limit)

    if not diseases:
        print("No diseases found. Check that GraphDB is running and has disease-symptom data loaded.")
        sys.exit(1)

    print(f"Found {len(diseases)} diseases. Fetching symptom profiles...")

    cases = []
    for idx, disease in enumerate(diseases, start=1):
        symptoms = _fetch_symptoms(disease["uri"])
        if not symptoms:
            print(f"  [{idx}/{len(diseases)}] Skipped (no symptoms returned): {disease['label']}")
            continue
        case = _build_case(disease, symptoms, idx)
        cases.append(case)
        print(f"  [{idx}/{len(diseases)}] {disease['label']} — {len(symptoms)} symptoms")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(cases)} test cases to {out_path}")
    print("Run tests with: pytest tests/")


if __name__ == "__main__":
    main()
