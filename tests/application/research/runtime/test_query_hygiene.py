"""Tests for the planner-output hygiene pipeline used by SearchExecutor."""

from __future__ import annotations

from houyi.application.research.runtime.search_executor import (
    _apply_query_hygiene,
    _is_cjk_char,
)


class TestIsCjkChar:
    def test_basic_ideograph_true(self):
        # Chinese character for "China" (zhong).
        assert _is_cjk_char("\u4e2d") is True

    def test_extension_a_true(self):
        # Extension A range sample.
        assert _is_cjk_char("\u3400") is True

    def test_latin_false(self):
        assert _is_cjk_char("A") is False

    def test_digit_false(self):
        assert _is_cjk_char("7") is False

    def test_fullwidth_punct_false(self):
        # Full-width comma lives outside the ideograph ranges.
        assert _is_cjk_char("\uff0c") is False


class TestQueryHygieneBasics:
    def test_keeps_healthy_query(self):
        # "mckenzie 9-tier definition" (CJK) with two informative tokens.
        query = "\u9ea6\u80af\u9521 \u4e5d\u9636\u5c42\u5b9a\u4e49"
        kept, dropped = _apply_query_hygiene([query])
        assert kept == [query]
        assert dropped == 0

    def test_keeps_long_narrative(self):
        # Long analytic question that must survive hygiene untouched.
        query = (
            "\u6309\u4e0d\u540c\u5b9a\u4e49\u6807\u51c6\uff0c\u4e2d\u56fd\u4e2d"
            "\u4ea7\u9636\u7ea7\u7684\u5b9e\u9645\u4eba\u53e3\u89c4\u6a21"
        )
        kept, dropped = _apply_query_hygiene([query])
        assert kept == [query]
        assert dropped == 0

    def test_drops_blank(self):
        kept, dropped = _apply_query_hygiene(["", "   ", "\t\n"])
        assert kept == []
        assert dropped == 3

    def test_preserves_order(self):
        healthy_a = "\u9ea6\u80af\u9521 \u4e5d\u9636\u5c42"
        noise = "\u963e\u5c42\u56fa\u5316 \u963e\u5c42\u56fa\u5316"
        healthy_b = "\u5404\u9636\u5c42\u7684\u5b9a\u4e49 \u4e5d\u9636\u5c42"
        kept, dropped = _apply_query_hygiene([healthy_a, noise, healthy_b])
        assert kept == [healthy_a, healthy_b]
        assert dropped == 1


class TestQueryHygieneRepeatedTokens:
    def test_drops_cjk_repeat(self):
        # "stratum-solidification stratum-solidification" verbatim repeat.
        query = "\u9636\u5c42\u56fa\u5316 \u9636\u5c42\u56fa\u5316"
        kept, dropped = _apply_query_hygiene([query])
        assert kept == []
        assert dropped == 1

    def test_drops_ascii_repeat(self):
        kept, dropped = _apply_query_hygiene(["foo bar foo"])
        assert kept == []
        assert dropped == 1

    def test_case_insensitive(self):
        # "Alpha" vs "alpha" must still be treated as a repeat.
        kept, dropped = _apply_query_hygiene(["Alpha beta alpha"])
        assert kept == []
        assert dropped == 1

    def test_keeps_single_token(self):
        # A lone token is preserved even if short; hygiene rule 1 needs
        # at least two tokens to fire.
        kept, dropped = _apply_query_hygiene(["\u4e2d\u4ea7\u9636\u7ea7"])
        assert kept == ["\u4e2d\u4ea7\u9636\u7ea7"]
        assert dropped == 0


class TestQueryHygieneNoise:
    def test_drops_filler_only(self):
        # "caili dengdeng" - filler alone, no informative content.
        kept, dropped = _apply_query_hygiene(["\u8d22\u529b\u7b49\u7b49"])
        assert kept == []
        assert dropped == 1

    def test_drops_noise_only(self):
        # "dengdeng ruhe" - filler + interrogative stopword only.
        kept, dropped = _apply_query_hygiene(["\u7b49\u7b49 \u5982\u4f55"])
        assert kept == []
        assert dropped == 1

    def test_drops_cjk_collapse(self):
        # "caili-dengdeng income-range" - two tokens but one is pure filler,
        # leaving a single short CJK content token.
        query = "\u8d22\u529b\u7b49\u7b49 \u6536\u5165\u533a\u95f4"
        kept, dropped = _apply_query_hygiene([query])
        assert kept == []
        assert dropped == 1

    def test_drops_ruhe_noun(self):
        # "ruhe caili" - interrogative + one short noun, should drop.
        kept, dropped = _apply_query_hygiene(["\u5982\u4f55 \u8d22\u529b"])
        assert kept == []
        assert dropped == 1

    def test_keeps_wh_query(self):
        # "how middle class size" keeps three informative tokens after
        # filtering the WH-word, so the query stays.
        kept, dropped = _apply_query_hygiene(["how middle class size"])
        assert kept == ["how middle class size"]
        assert dropped == 0

    def test_keeps_rich_cjk(self):
        # "caili-dengdeng national-bureau middle-income-group 2024" keeps
        # three informative tokens after filler, must not be dropped.
        query = (
            "\u8d22\u529b\u7b49\u7b49 "
            "\u56fd\u5bb6\u7edf\u8ba1\u5c40 "
            "\u4e2d\u7b49\u6536\u5165\u7fa4\u4f53 "
            "2024"
        )
        kept, dropped = _apply_query_hygiene([query])
        assert kept == [query]
        assert dropped == 0


class TestQueryHygieneReport:
    def test_reports_drop_count(self):
        inputs = [
            "",
            "foo foo",
            "healthy query",
            "\u5982\u4f55 \u8d22\u529b",  # CJK collapse
        ]
        kept, dropped = _apply_query_hygiene(inputs)
        assert kept == ["healthy query"]
        assert dropped == 3

    def test_empty_input(self):
        kept, dropped = _apply_query_hygiene([])
        assert kept == []
        assert dropped == 0

    def test_accepts_generator(self):
        def _gen():
            yield "foo foo"
            yield "healthy query"

        kept, dropped = _apply_query_hygiene(_gen())
        assert kept == ["healthy query"]
        assert dropped == 1
