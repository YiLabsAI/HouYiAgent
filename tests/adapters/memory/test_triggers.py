"""Tests for L1 extraction trigger policies (min-length, role, composites)."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.triggers import (
    MinLengthTrigger,
    RegexBlocklistTrigger,
    RoleTrigger,
    all_of,
    any_of,
    default_extract_policy,
)
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import RawTurn


def _turn(content: str, *, role: str = "user", session: str = "s") -> RawTurn:
    return RawTurn(session_id=session, role=role, content=content)


class TestMinLengthTrigger:
    def test_short_rejected(self):
        t = MinLengthTrigger(min_chars=4)
        assert t.should_extract(_turn("hi")) is False
        assert t.should_extract(_turn("ok ")) is False

    def test_long_accepted(self):
        assert MinLengthTrigger(min_chars=4).should_extract(_turn("hello there"))


class TestRoleTrigger:
    def test_default_allows_user(self):
        t = RoleTrigger()
        assert t.should_extract(_turn("x", role="user"))
        assert t.should_extract(_turn("x", role="assistant"))
        assert not t.should_extract(_turn("x", role="system"))
        assert not t.should_extract(_turn("x", role="tool"))

    def test_custom_role_set(self):
        t = RoleTrigger(allowed_roles=frozenset({"system"}))
        assert t.should_extract(_turn("x", role="system"))
        assert not t.should_extract(_turn("x", role="user"))


class TestRegexBlocklistTrigger:
    def test_blocks_matching_pattern(self):
        t = RegexBlocklistTrigger([r"^/cmd"])
        assert not t.should_extract(_turn("/cmd reboot"))
        assert t.should_extract(_turn("hello"))

    def test_case_insensitive(self):
        t = RegexBlocklistTrigger([r"PASSWORD"])
        assert not t.should_extract(_turn("password is 123"))


class TestComposites:
    def test_all_of_short_circuits(self):
        t = all_of(MinLengthTrigger(min_chars=4), RoleTrigger())
        # Short user turn fails the AND.
        assert not t.should_extract(_turn("hi"))
        # Long system turn fails the AND on role.
        assert not t.should_extract(_turn("hello there", role="system"))
        # Long user turn passes both gates.
        assert t.should_extract(_turn("hello there"))

    def test_empty_all_of_accepts(self):
        # all([]) == True — useful as a "permissive" sentinel for tests.
        t = all_of()
        assert t.should_extract(_turn("anything", role="system"))

    def test_any_of_one_member(self):
        t = any_of(
            RegexBlocklistTrigger([r"^secret"]),  # blocks 'secret', else True
            RoleTrigger(allowed_roles=frozenset({"system"})),
        )
        # 'secret hi' is blocked by first, but role=system passes second.
        assert t.should_extract(_turn("secret hi", role="system"))
        # plain user msg matches the first (no regex hit, so True).
        assert t.should_extract(_turn("plain msg", role="user"))

    def test_empty_any_of_rejects(self):
        # Counter-intuitive default would be True (any([]) == False in
        # Python, but the doc explicitly states "rejects").
        assert not any_of().should_extract(_turn("hi"))


class TestDefaultPolicy:
    def test_short_user_turn_rejected(self):
        assert not default_extract_policy().should_extract(_turn("ok"))

    def test_normal_user_turn_accepted(self):
        assert default_extract_policy().should_extract(_turn("hello there"))


class TestTurnWriterIntegration:
    @pytest.fixture()
    def backend(self, tmp_path):
        b = SQLiteMemoryBackend(db_path=tmp_path / "trig.db")
        yield b
        b.close()

    def test_default_skips_short(self, backend):
        wp = TurnWriter(backend)  # default policy installed
        result = wp.fast_path(_turn("ok"))
        assert result.queue_id is None
        assert result.extract_skipped is True
        # L0 row still persisted — durability is non-negotiable.
        assert backend.list_raw_turns("default", "s") != []
        assert backend.extract_queue_stats() == {}

    def test_default_enqueues_normal(self, backend):
        wp = TurnWriter(backend)
        result = wp.fast_path(_turn("hello world"))
        assert result.queue_id is not None
        assert result.extract_skipped is False
        assert backend.extract_queue_stats() == {"pending": 1}

    def test_custom_blocklist_skips_commands(self, backend):
        wp = TurnWriter(
            backend,
            extract_trigger=all_of(
                MinLengthTrigger(min_chars=2),
                RegexBlocklistTrigger([r"^/"]),
            ),
        )
        ok = wp.fast_path(_turn("hello"))
        cmd = wp.fast_path(_turn("/reset"))
        assert ok.queue_id is not None
        assert cmd.queue_id is None
        assert cmd.extract_skipped is True
