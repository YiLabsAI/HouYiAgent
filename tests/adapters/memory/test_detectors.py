"""Tests for fast-path detectors: explicit-pin and emphasis detection."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.detectors import (
    EmphasisDetector,
    ExplicitPinDetector,
)
from houyi.adapters.memory.detectors.emphasis import EmphasisKind
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import RawTurn


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteMemoryBackend(db_path=tmp_path / "det.db")
    yield b
    b.close()


def _turn(content: str, *, session: str = "s") -> RawTurn:
    return RawTurn(session_id=session, role="user", content=content)


# ---------------------------------------------------------------------------
# ExplicitPinDetector
# ---------------------------------------------------------------------------


class TestExplicitPinDetector:
    def test_english_remember_pattern(self, backend):
        det = ExplicitPinDetector(backend)
        det.detect(_turn("Please remember that the wifi password is hunter2"))
        sig = det.last_signal
        assert sig is not None
        assert sig.payload == "the wifi password is hunter2"
        # Stored as a pending-embedding row with a 'pinned' tag.
        pending = backend.list_pending_embeddings(limit=10)
        assert len(pending) == 1
        _, record = pending[0]
        assert record.content == "the wifi password is hunter2"
        assert "pinned" in record.tags
        assert record.confidence == 0.95

    def test_chinese_remember_pattern(self, backend):
        det = ExplicitPinDetector(backend)
        # "ji-zhu, ming-tian-xia-wu-san-dian-kai-hui" (remember tomorrow 3pm meeting)
        det.detect(_turn("\u8bb0\u4f4f\uff0c\u660e\u5929\u4e0b\u5348\u4e09\u70b9\u5f00\u4f1a"))
        assert det.last_signal is not None
        assert det.last_signal.payload == ("\u660e\u5929\u4e0b\u5348\u4e09\u70b9\u5f00\u4f1a")

    def test_pin_colon_pattern(self, backend):
        det = ExplicitPinDetector(backend)
        det.detect(_turn("Pin: API key rotation is every 90 days"))
        assert det.last_signal is not None
        assert det.last_signal.payload.startswith("API key rotation")

    def test_no_match_no_write(self, backend):
        det = ExplicitPinDetector(backend)
        det.detect(_turn("just chatting about the weather"))
        assert det.last_signal is None
        assert backend.list_pending_embeddings(limit=10) == []

    def test_empty_payload_refused(self, backend):
        det = ExplicitPinDetector(backend)
        det.detect(_turn("remember"))
        # No payload after the cue — must NOT pin an empty string.
        assert det.last_signal is None
        assert backend.list_pending_embeddings(limit=10) == []

    def test_backend_failure_swallowed(self):
        class _Boom:
            def put(self, record):
                raise RuntimeError("disk full")

        det = ExplicitPinDetector(_Boom())
        # Must not raise — detector failure cannot break the L0 path.
        det.detect(_turn("remember the milk"))

    def test_construction_validation(self):
        with pytest.raises(ValueError):
            ExplicitPinDetector(None)


# ---------------------------------------------------------------------------
# EmphasisDetector
# ---------------------------------------------------------------------------


class TestEmphasisDetector:
    def test_keyword_match_in_english(self):
        det = EmphasisDetector()
        turn = _turn("This is important: don't restart the server")
        det.detect(turn)
        sig = det.last_signal
        assert sig is not None
        assert sig.kind is EmphasisKind.KEYWORD
        assert turn.metadata["emphasis"] == "keyword"
        assert "emphasis_score" in turn.metadata

    def test_keyword_match_in_chinese(self):
        det = EmphasisDetector()
        # "zhong-yao: ming-tian-zao-shang-ba-dian-ti-jiao" (important: 8am submit)
        turn = _turn("\u91cd\u8981\uff1a\u660e\u5929\u65e9\u4e0a\u516b\u70b9\u63d0\u4ea4")
        det.detect(turn)
        assert det.last_signal is not None
        assert det.last_signal.kind is EmphasisKind.KEYWORD

    def test_repeated_punctuation(self):
        det = EmphasisDetector()
        turn = _turn("hurry up!!!")
        det.detect(turn)
        sig = det.last_signal
        assert sig is not None
        assert sig.kind is EmphasisKind.REPEATED_PUNCT

    def test_all_caps_run(self):
        det = EmphasisDetector()
        turn = _turn("DO NOT MERGE THIS BRANCH")
        det.detect(turn)
        sig = det.last_signal
        assert sig is not None
        assert sig.kind is EmphasisKind.ALL_CAPS
        assert sig.score >= 0.3

    def test_normal_text_no_signal(self):
        det = EmphasisDetector()
        turn = _turn("hello, just checking in")
        det.detect(turn)
        assert det.last_signal is None
        assert "emphasis" not in turn.metadata

    def test_keyword_priority_over_punct(self):
        det = EmphasisDetector()
        turn = _turn("important!!!")
        det.detect(turn)
        # Keyword fires first per the documented priority order.
        assert det.last_signal.kind is EmphasisKind.KEYWORD

    def test_min_score_filter(self):
        # All-caps "OK GO" scores 0.3 + 0.05 * 5 = 0.55. Bumping
        # min_score above that hides the signal even though the regex
        # still matches.
        det = EmphasisDetector(min_score=0.6)
        turn = _turn("OK GO")
        det.detect(turn)
        assert det.last_signal is None

    def test_empty_text_no_signal(self):
        det = EmphasisDetector()
        turn = _turn(" ")
        det.detect(turn)
        assert det.last_signal is None

    def test_min_score_validation(self):
        with pytest.raises(ValueError):
            EmphasisDetector(min_score=-0.1)
            with pytest.raises(ValueError):
                EmphasisDetector(min_score=1.5)


# ---------------------------------------------------------------------------
# Integration with TurnWriter
# ---------------------------------------------------------------------------


class TestDetectorsThroughTurnWriter:
    def test_pin_detector_fast_path(self, backend):
        det = ExplicitPinDetector(backend)
        wp = TurnWriter(backend, detectors=[det], extract_trigger=all_of())
        result = wp.fast_path(_turn("please remember the gate code 4242"))
        assert "ExplicitPinDetector" in result.detectors_fired
        # Pin record AND a separate L0 row for the raw turn — distinct.
        assert len(backend.list_pending_embeddings(limit=10)) == 1
        assert backend.count_raw_turns("default", "s") == 1

    def test_emphasis_decorates_metadata(self, backend):
        det = EmphasisDetector()
        wp = TurnWriter(backend, detectors=[det], extract_trigger=all_of())
        # "zhong-yao: zhou-mo-jia-ban" (important: weekend overtime)
        turn = RawTurn(
            session_id="s",
            role="user",
            content="\u91cd\u8981\uff1a\u5468\u672b\u52a0\u73ed",
        )
        result = wp.fast_path(turn)
        # The persisted L0 row carries the emphasis tag.
        rows = backend.list_raw_turns("default", "s")
        assert rows[0].metadata.get("emphasis") == "keyword"
        assert "EmphasisDetector" in result.detectors_fired
