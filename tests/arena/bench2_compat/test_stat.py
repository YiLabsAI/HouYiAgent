from __future__ import annotations

import json
from pathlib import Path

from houyi.arena.bench2_compat import stat as compat_stat


def test_handles_zero_denominator(tmp_path: Path) -> None:
    input_path = tmp_path / "validated.jsonl"
    output_path = tmp_path / "fact_result.txt"
    input_path.write_text(
        json.dumps({"id": "1", "citations": [], "citations_deduped": {}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    compat_stat.write_fact_stats(input_path, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "total_citations: 0.0" in text
    assert "valid_rate: 0.0" in text


def test_counts_supported(tmp_path: Path) -> None:
    input_path = tmp_path / "validated.jsonl"
    output_path = tmp_path / "fact_result.txt"
    entry = {
        "id": "1",
        "citations": [{"fact": "A"}],
        "citations_deduped": {
            "https://a": {
                "validate_error": None,
                "validate_res": [
                    {"idx": 0, "result": "supported"},
                    {"idx": 1, "result": "unknown"},
                ],
            }
        },
    }
    input_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    compat_stat.write_fact_stats(input_path, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "total_citations: 1.0" in text
    assert "total_valid_citations: 1.0" in text
    assert "valid_rate: 1.0" in text
