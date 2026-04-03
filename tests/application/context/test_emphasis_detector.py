"""Unit tests for EmphasisDetector."""

from __future__ import annotations

from houyi.application.context.emphasis_detector import (
    EmphasisDetector,
    EmphasisReport,
    EmphasisSignal,
    _extract_key_phrases,
)


def _msgs(*contents: str) -> list[dict]:
    return [{"role": "user", "content": c} for c in contents]


class TestAnalyze:
    def test_detects_linguistic_cues(self):
        detector = EmphasisDetector()
        report = detector.analyze(_msgs("You must always use citations."))
        types = {s.signal_type for s in report.signals}
        assert "directive" in types

    def test_detects_repetition(self):
        detector = EmphasisDetector(repetition_threshold=2)
        msgs = _msgs(
            "Please use short function names.",
            "I want short function names in the code.",
            "Remember, short function names are important.",
        )
        report = detector.analyze(msgs)
        rep_signals = [s for s in report.signals if s.signal_type == "repetition"]
        assert len(rep_signals) > 0

    def test_skips_assistant_messages(self):
        detector = EmphasisDetector()
        msgs = [
            {"role": "user", "content": "normal text"},
            {"role": "assistant", "content": "You must ALWAYS do this!"},
        ]
        report = detector.analyze(msgs)
        directive_signals = [s for s in report.signals if s.signal_type == "directive"]
        assert len(directive_signals) == 0

    def test_detects_caps(self):
        detector = EmphasisDetector()
        report = detector.analyze(_msgs("This is VERY IMPORTANT and CRITICAL"))
        formatting = [s for s in report.signals if s.signal_type == "formatting"]
        assert len(formatting) >= 1

    def test_detects_exclamation(self):
        detector = EmphasisDetector()
        report = detector.analyze(_msgs("Fix this immediately!!"))
        formatting = [s for s in report.signals if s.signal_type == "formatting"]
        assert len(formatting) >= 1

    def test_empty_messages(self):
        detector = EmphasisDetector()
        report = detector.analyze([])
        assert report.signals == []
        assert report.total_emphasis_score == 0.0

    def test_min_strength_filter(self):
        detector = EmphasisDetector(min_strength=2.0)
        report = detector.analyze(_msgs("must do this"))
        for sig in report.signals:
            assert sig.strength >= 2.0

    def test_top_emphasized_dedup(self):
        detector = EmphasisDetector()
        report = detector.analyze(
            _msgs(
                "always use citations, always add tests",
                "ALWAYS follow the spec",
            )
        )
        assert len(report.top_emphasized) <= 5

    def test_sorted_by_strength(self):
        detector = EmphasisDetector()
        report = detector.analyze(_msgs("must do this, never forget, ALWAYS REMEMBER!!"))
        strengths = [s.strength for s in report.signals]
        assert strengths == sorted(strengths, reverse=True)


class TestExtractKeyPhrases:
    def test_extracts_phrases(self):
        phrases = _extract_key_phrases("use short function names")
        assert any("function names" in p for p in phrases)

    def test_empty_input(self):
        assert _extract_key_phrases("") == []

    def test_short_text(self):
        phrases = _extract_key_phrases("hi")
        assert phrases == []


class TestModels:
    def test_signal_defaults(self):
        s = EmphasisSignal(text="test", signal_type="directive")
        assert s.strength == 1.0
        assert s.occurrences == 1

    def test_report_defaults(self):
        r = EmphasisReport()
        assert r.signals == []
        assert r.total_emphasis_score == 0.0
