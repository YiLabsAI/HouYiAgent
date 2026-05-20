"""Tests for the raw turn log (L0 tier) append and query operations."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.types import RawTurn


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteMemoryBackend(db_path=tmp_path / "raw.db")
    yield b
    b.close()


class TestAppendRawTurn:
    def test_assigns_monotonic_index(self, backend):
        turns = [
            backend.append_raw_turn(RawTurn(session_id="s1", role="user", content=f"hi {i}"))
            for i in range(3)
        ]
        assert [t.turn_index for t in turns] == [0, 1, 2]

    def test_separate_sessions_counters(self, backend):
        a = backend.append_raw_turn(RawTurn(session_id="A", role="user", content="x"))
        b = backend.append_raw_turn(RawTurn(session_id="B", role="user", content="y"))
        assert a.turn_index == 0
        assert b.turn_index == 0

    def test_namespace_isolation(self, backend):
        a = backend.append_raw_turn(
            RawTurn(namespace="t1", session_id="s", role="user", content="hi")
        )
        b = backend.append_raw_turn(
            RawTurn(namespace="t2", session_id="s", role="user", content="hi")
        )
        assert a.turn_index == 0
        assert b.turn_index == 0

    def test_explicit_index_is_honored(self, backend):
        backend.append_raw_turn(RawTurn(session_id="s", role="user", content="0"))
        explicit = backend.append_raw_turn(
            RawTurn(session_id="s", role="user", content="x", turn_index=5)
        )
        assert explicit.turn_index == 5
        # The next auto-assigned index continues from the max.
        nxt = backend.append_raw_turn(RawTurn(session_id="s", role="user", content="next"))
        assert nxt.turn_index == 6

    def test_persists_metadata(self, backend):
        backend.append_raw_turn(
            RawTurn(
                session_id="s",
                role="user",
                content="hi",
                metadata={"trace_id": "abc-123", "lang": "en"},
            )
        )
        rows = backend.list_raw_turns("default", "s")
        assert rows[0].metadata == {"trace_id": "abc-123", "lang": "en"}


class TestListRawTurns:
    def test_returns_chronological_order(self, backend):
        for i in range(5):
            backend.append_raw_turn(RawTurn(session_id="s", role="user", content=str(i)))
        out = backend.list_raw_turns("default", "s")
        assert [t.content for t in out] == ["0", "1", "2", "3", "4"]

    def test_limit_and_offset(self, backend):
        for i in range(10):
            backend.append_raw_turn(RawTurn(session_id="s", role="user", content=str(i)))
        page = backend.list_raw_turns("default", "s", limit=3, offset=4)
        assert [t.content for t in page] == ["4", "5", "6"]

    def test_empty_session_returns_empty(self, backend):
        assert backend.list_raw_turns("default", "missing") == []


class TestRawTurnLookup:
    def test_count_raw_turns(self, backend):
        for i in range(4):
            backend.append_raw_turn(RawTurn(session_id="s", role="user", content=str(i)))
        assert backend.count_raw_turns("default", "s") == 4
        assert backend.count_raw_turns("default", "other") == 0

    def test_get_raw_turn(self, backend):
        appended = backend.append_raw_turn(
            RawTurn(session_id="s", role="assistant", content="pong")
        )
        got = backend.get_raw_turn(appended.turn_id)
        assert got is not None
        assert got.content == "pong"
        assert got.role == "assistant"

    def test_get_missing_returns_none(self, backend):
        assert backend.get_raw_turn("nope") is None


class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "persist.db"
        b1 = SQLiteMemoryBackend(db_path=path)
        b1.append_raw_turn(RawTurn(session_id="s", role="user", content="hello"))
        b1.close()

        b2 = SQLiteMemoryBackend(db_path=path)
        out = b2.list_raw_turns("default", "s")
        assert len(out) == 1
        assert out[0].content == "hello"
        b2.close()
