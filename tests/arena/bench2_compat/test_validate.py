from __future__ import annotations

import json
from pathlib import Path

from houyi.arena.bench2_compat import validate as compat_validate


def test_coerces_empty_list() -> None:
    facts = ["A", "B"]
    result = compat_validate._coerce_results("[]", facts)
    assert result == [
        {"idx": 0, "result": "unknown"},
        {"idx": 1, "result": "unknown"},
    ]


def test_normalizes_valid_list() -> None:
    facts = ["A", "B"]
    response = json.dumps(
        [
            {"idx": 1, "result": "supported"},
            {"idx": 2, "result": "unsupported"},
        ]
    )
    result = compat_validate._coerce_results(response, facts)
    assert result == [
        {"idx": 0, "result": "supported"},
        {"idx": 1, "result": "unsupported"},
    ]


def test_skips_inaccessible_ref(monkeypatch) -> None:
    payload = (
        "https://a.example",
        {
            "url_content": "too short",
            "facts": ["Fact A"],
            "article_id": "1",
        },
    )
    monkeypatch.setattr(
        compat_validate,
        "call_model",
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM should not be used")),
    )

    result = compat_validate.validate(payload, {"1": "en"})

    assert result == {
        "url": "https://a.example",
        "validate_res": [{"idx": 0, "result": "unknown"}],
        "error": None,
    }


def test_run_writes_output(monkeypatch, tmp_path: Path) -> None:
    raw_data_path = tmp_path / "scraped.jsonl"
    output_path = tmp_path / "validated.jsonl"
    query_path = tmp_path / "query.jsonl"

    raw_entry = {
        "id": "1",
        "citations_deduped": {
            "https://a.example": {
                "url_content": "reference text",
                "facts": ["Fact A", "Fact B"],
            }
        },
    }
    query_entry = {"id": "1", "language": "en"}
    raw_data_path.write_text(json.dumps(raw_entry, ensure_ascii=False) + "\n", encoding="utf-8")
    query_path.write_text(json.dumps(query_entry, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(compat_validate, "call_model", lambda prompt: "[]")

    compat_validate.run_validation(
        raw_data_path=raw_data_path,
        output_path=output_path,
        query_data_path=query_path,
        n_total_process=1,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    citation = row["citations_deduped"]["https://a.example"]
    assert citation["validate_error"] is None
    assert citation["validate_res"] == [
        {"idx": 0, "result": "unknown"},
        {"idx": 1, "result": "unknown"},
    ]


def test_skips_empty_facts(monkeypatch) -> None:
    payload = (
        "https://a.example",
        {
            "url_content": "reference text",
            "facts": [],
            "article_id": "1",
        },
    )
    monkeypatch.setattr(
        compat_validate,
        "call_model",
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM should not be used")),
    )

    result = compat_validate.validate(payload, {"1": "en"})

    assert result == {
        "url": "https://a.example",
        "validate_res": [],
        "error": None,
    }


def test_resolves_worker_env(monkeypatch) -> None:
    monkeypatch.setenv("BENCH2_VALIDATE_TOTAL_PROCESS", "3")

    assert compat_validate._resolve_worker_count(0) == 3
