"""Tests for cross-turn context recovery (fetch_turn_context).

The render-side injection recovers conversation context lost by per-turn
atomization. Neighbour selection is format-agnostic:
- dia_id priority: turn_ids encoding D{session}:N gather same-prefix +/-window.
- turn_index fallback: opaque turn_ids use per-(namespace, session_id) +/-window.
"""

from __future__ import annotations

from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.turn_context import fetch_turn_context
from houyi.adapters.memory.types import RawTurn


def _append(raw_log, turn_id, ns, sid, idx, role, content):
    raw_log.append_raw_turn(
        RawTurn(
            turn_id=turn_id,
            namespace=ns,
            session_id=sid,
            turn_index=idx,
            role=role,
            content=content,
        )
    )


class TestDiaIdNeighbours:
    """dia_id priority: turn_ids with D{session}:N are parsed and gathered
    by conversation position, regardless of turn_index ordering."""

    def test_dia_id_neighbours(self, tmp_path):
        store = MemoryStore(data_dir=tmp_path)
        rl = store.backend._raw_turn_log
        ns = "ns1"
        for i, (n, role, txt) in enumerate(
            [
                (2, "user", "I got back from a family trip."),
                (4, "user", "We went to the Rockies."),
                (6, "user", "We hiked the trails."),
                (8, "user", "The views were great."),
                (10, "user", "End of session."),
            ]
        ):
            _append(rl, f"c:s1:D1:{n}:{i}", ns, "s1", i * 10, role, txt)
        ctx = fetch_turn_context(store.backend, "c:s1:D1:4:1", window=2)
        assert "family trip" in ctx
        assert "Rockies" in ctx
        assert "hiked" in ctx
        assert "End of session" not in ctx
        store.close()

    def test_dia_id_resists_distortion(self, tmp_path):
        """turn_index is ingestion order, NOT conversation order. dia_id
        parsing must gather conversation-adjacent turns even when
        turn_index gaps are large."""
        store = MemoryStore(data_dir=tmp_path)
        rl = store.backend._raw_turn_log
        ns = "ns1"
        _append(rl, "c:s1:D1:2:0", ns, "s1", 0, "user", "family trip")
        _append(rl, "c:s1:D1:3:1", ns, "s1", 50, "assistant", "where")
        _append(rl, "c:s1:D1:4:2", ns, "s1", 99, "user", "we went to Rockies")
        ctx = fetch_turn_context(store.backend, "c:s1:D1:4:2", window=2)
        assert "family trip" in ctx
        assert "Rockies" in ctx
        store.close()


class TestTurnIndexFallback:
    """turn_index fallback: opaque turn_ids (e.g. uuid) use per-session
    turn_index +/-window."""

    def test_uuid_fallback(self, tmp_path):
        store = MemoryStore(data_dir=tmp_path)
        rl = store.backend._raw_turn_log
        ns = "ns1"
        _append(rl, "uuid-aaa", ns, "s1", 0, "user", "turn zero")
        _append(rl, "uuid-bbb", ns, "s1", 1, "user", "turn one")
        _append(rl, "uuid-ccc", ns, "s1", 2, "user", "turn two target")
        _append(rl, "uuid-ddd", ns, "s1", 3, "user", "turn three")
        _append(rl, "uuid-eee", ns, "s1", 4, "user", "turn four")
        ctx = fetch_turn_context(store.backend, "uuid-ccc", window=1)
        assert "turn one" in ctx
        assert "turn two target" in ctx
        assert "turn three" in ctx
        assert "turn zero" not in ctx
        assert "turn four" not in ctx
        store.close()

    def test_empty_returns_empty(self, tmp_path):
        store = MemoryStore(data_dir=tmp_path)
        assert fetch_turn_context(store.backend, "", window=3) == ""
        assert fetch_turn_context(store.backend, None, window=3) == ""
        store.close()

    def test_missing_returns_empty(self, tmp_path):
        store = MemoryStore(data_dir=tmp_path)
        assert fetch_turn_context(store.backend, "nonexistent", window=3) == ""
        store.close()
