#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running from project root without installing package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hospitalmas.tools.graphdb_ontology_query_tool import GraphDbOntologyQueryTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraphDbOntologyQueryTool directly for manual testing.",
    )
    parser.add_argument(
        "labels",
        nargs="*",
        default=["0000132"],
        help="One or more SYMP labels to test, e.g. 0000132 0000613",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool = GraphDbOntologyQueryTool(timeout_seconds=args.timeout)

    for label in args.labels:
        print("=" * 80)
        print(f"Testing SYMP label: {label}")
        query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ro: <http://purl.obolibrary.org/obo/RO_>
    PREFIX symp: <http://purl.obolibrary.org/obo/SYMP_>
    SELECT ?disease ?diseaseLabel
    WHERE {{
      ?disease rdfs:subClassOf ?restriction .
      ?restriction owl:onProperty ro:0002452 .
      ?restriction owl:someValuesFrom symp:{label} .
      ?disease rdfs:label ?diseaseLabel .
      FILTER(LANG(?diseaseLabel) = "en")
    }}
    """.strip()
        result = tool._run(query)
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
