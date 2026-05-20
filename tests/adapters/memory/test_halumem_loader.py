"""Tests for loading and validating the HaluMem QA benchmark dataset."""

from __future__ import annotations

import json

import pytest

from houyi.adapters.memory.bench import (
    HaluMemCase,
    HaluMemTask,
    cases_by_task,
    load_halumem,
)


def _case_dict(
    *,
    sid="h-1",
    task="memory_integrity",
    question="q",
    answer="a",
    conversation=None,
    evidence=None,
):
    return {
        "sample_id": sid,
        "task": task,
        "question": question,
        "answer": answer,
        "conversation": conversation
        if conversation is not None
        else [{"speaker": "Alice", "text": "hi"}],
        "evidence": evidence if evidence is not None else ["turn-1"],
    }


@pytest.fixture()
def list_path(tmp_path):
    p = tmp_path / "halumem.json"
    p.write_text(
        json.dumps(
            [
                _case_dict(sid="h-1", task="memory_integrity"),
                _case_dict(sid="h-2", task="memory_accuracy"),
                _case_dict(sid="h-3", task="qa_accuracy"),
            ]
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def dict_path(tmp_path):
    p = tmp_path / "halumem_dict.json"
    p.write_text(
        json.dumps({"cases": [_case_dict(sid=f"h-{i}") for i in range(1, 4)]}),
        encoding="utf-8",
    )
    return p


class TestLoadShape:
    def test_list_top_level(self, list_path):
        cases = load_halumem(list_path)
        assert len(cases) == 3
        assert all(isinstance(c, HaluMemCase) for c in cases)
        assert {c.task for c in cases} == set(HaluMemTask)

    def test_dict_with_cases_key(self, dict_path):
        cases = load_halumem(dict_path)
        assert len(cases) == 3

    def test_turn_field_aliases_accepted(self, tmp_path):
        # Some HaluMem distributions use {role, content} instead of
        # {speaker, text}. The loader handles both.
        path = tmp_path / "alias.json"
        path.write_text(
            json.dumps(
                [
                    _case_dict(
                        conversation=[
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "hi"},
                        ]
                    )
                ]
            ),
            encoding="utf-8",
        )
        cases = load_halumem(path)
        assert len(cases[0].conversation) == 2
        assert cases[0].conversation[0].speaker == "user"

    def test_missing_required_field_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        bad = _case_dict()
        del bad["question"]
        path.write_text(json.dumps([bad]), encoding="utf-8")
        with pytest.raises(ValueError, match="question"):
            load_halumem(path)

    def test_unknown_task_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([_case_dict(task="not_a_task")]), encoding="utf-8")
        with pytest.raises(ValueError, match="task"):
            load_halumem(path)

    def test_duplicate_sample_id_rejected(self, tmp_path):
        path = tmp_path / "dup.json"
        path.write_text(
            json.dumps([_case_dict(sid="x"), _case_dict(sid="x")]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_halumem(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_halumem(tmp_path / "nope.json")

    def test_neither_list_nor_dict(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps("not a list"), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level"):
            load_halumem(path)


class TestEvidenceParsing:
    def test_evidence_as_list(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text(json.dumps([_case_dict(evidence=["t1", "t2"])]), encoding="utf-8")
        cases = load_halumem(path)
        assert cases[0].evidence == ("t1", "t2")

    def test_evidence_string_to_tuple(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text(json.dumps([_case_dict(evidence="solo")]), encoding="utf-8")
        cases = load_halumem(path)
        assert cases[0].evidence == ("solo",)


class TestGrouping:
    def test_cases_by_task_buckets(self, list_path):
        cases = load_halumem(list_path)
        grouped = cases_by_task(cases)
        # Each enum gets a bucket, even when empty.
        assert set(grouped.keys()) == set(HaluMemTask)
        assert len(grouped[HaluMemTask.MEMORY_INTEGRITY]) == 1
        assert len(grouped[HaluMemTask.MEMORY_ACCURACY]) == 1
        assert len(grouped[HaluMemTask.QA_ACCURACY]) == 1
