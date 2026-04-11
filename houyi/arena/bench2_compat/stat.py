from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_fact_stats(input_path: str | Path, output_path: str | Path) -> None:
    data = _load_jsonl(Path(input_path))
    total_citations = 0
    total_valid_citations = 0
    total_num = 0

    for entry in data:
        citations = entry.get("citations")
        if not citations:
            continue
        for citation in (entry.get("citations_deduped") or {}).values():
            if citation.get("validate_error") is not None:
                continue
            for item in citation.get("validate_res") or []:
                result = item.get("result")
                if result != "unknown":
                    total_citations += 1
                    if result == "supported":
                        total_valid_citations += 1
        total_num += 1

    avg_total_citations = total_citations / total_num if total_num else 0.0
    avg_valid_citations = total_valid_citations / total_num if total_num else 0.0
    valid_rate = total_valid_citations / total_citations if total_citations else 0.0

    output = Path(output_path)
    output.write_text(
        "".join(
            [
                f"total_citations: {avg_total_citations}\n",
                f"total_valid_citations: {avg_valid_citations}\n",
                f"valid_rate: {valid_rate}\n",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    write_fact_stats(args.input_path, args.output_path)
