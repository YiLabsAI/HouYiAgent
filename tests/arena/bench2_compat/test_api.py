from __future__ import annotations

import json

from houyi.arena.bench2_compat import api as compat_api


def _extract_prompt(report_text: str) -> str:
    return (
        "You will be provided with a research report containing citations.\n"
        'Extract (fact, "ref_idx", "url") triplets from the report.\n'
        "Here is the main text of the research report:\n"
        f"{report_text}\n\n"
        "Please begin the extraction now. Output only the JSON list directly."
    )


def _dedup_prompt(statements: str) -> str:
    return (
        "You should return a List(int) of unique statement indices.\n"
        "Below is the list of statements you need to de-duplicate:\n"
        f"{statements}\n\n"
        "Please begin the extraction now. Output only the integer list."
    )


def test_parses_plain_array() -> None:
    normalized = compat_api._normalize_json_output('[{"fact":"A","ref_idx":1,"url":"https://a"}]')
    assert normalized is not None
    parsed = json.loads(normalized)
    assert parsed[0]["fact"] == "A"


def test_parses_fenced_json() -> None:
    normalized = compat_api._normalize_json_output('```json\n[{"idx":1,"result":"supported"}]\n```')
    assert normalized is not None
    parsed = json.loads(normalized)
    assert parsed[0]["result"] == "supported"


def test_repairs_fragment() -> None:
    raw = '[{"fact":"A","ref_idx":1,"url":"https://a"}'
    normalized = compat_api._normalize_json_output(raw)
    assert normalized is not None
    parsed = json.loads(normalized)
    assert parsed[0]["url"] == "https://a"


def test_falls_back_empty_list(monkeypatch) -> None:
    monkeypatch.setenv("HOUYI_BENCH2_FACT_CALL_RETRIES", "1")
    monkeypatch.setattr(compat_api, "_run_chat", lambda *args, **kwargs: "not json")

    result = compat_api.call_model("Please output json列表 directly")

    assert result == "[]"


def test_extracts_inline_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        compat_api,
        "_run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be used")),
    )
    prompt = _extract_prompt(
        "# Title\n\nMarket share remains concentrated [ref_a1](https://a.example/source).\n\n## References\n- [A](https://a.example/source)"
    )

    result = compat_api.call_model(prompt)

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["url"] == "https://a.example/source"
    assert "Market share remains concentrated" in parsed[0]["fact"]


def test_extracts_without_polluting_fact(monkeypatch) -> None:
    monkeypatch.setattr(
        compat_api,
        "_run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be used")),
    )
    prompt = _extract_prompt(
        "# Title\n\nMarket share remains concentrated [Source A](https://a.example/source).\n\n## References\n- [Source A](https://a.example/source)"
    )

    result = compat_api.call_model(prompt)

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["url"] == "https://a.example/source"
    assert parsed[0]["fact"] == "Market share remains concentrated"


def test_ignores_reference_links(monkeypatch) -> None:
    monkeypatch.setattr(
        compat_api,
        "_run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be used")),
    )
    prompt = _extract_prompt(
        "# Title\n\nNo inline citations here.\n\n## References\n- [A](https://a.example/source)"
    )

    result = compat_api.call_model(prompt)

    assert result == "[]"


def test_dedups_exact_statements(monkeypatch) -> None:
    monkeypatch.setattr(
        compat_api,
        "_run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be used")),
    )
    prompt = _dedup_prompt("1. A fact\n2. A fact\n3. Another fact")

    result = compat_api.call_model(prompt)

    assert result == "[1, 3]"


def test_scrape_handles_readability_error(monkeypatch) -> None:
    class _FakeJina:
        async def fetch(self, urls):
            return {urls[0]: ""}

    class _FakeReadability:
        async def fetch(self, urls):
            raise ValueError("bad html")

    monkeypatch.setattr(compat_api, "JinaContentFetcher", lambda: _FakeJina())
    monkeypatch.setattr(compat_api, "ReadabilityContentFetcher", lambda: _FakeReadability())

    result = compat_api.scrape_url("https://example.com")

    assert result["url"] == "https://example.com"
    assert result["content"] == ""
    assert "bad html" in result["error"]


def test_stops_after_timeout(monkeypatch) -> None:
    calls = {"count": 0}
    monkeypatch.setenv("HOUYI_BENCH2_FACT_CALL_RETRIES", "3")

    def _timeout(*args, **kwargs):
        calls["count"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(compat_api, "_run_chat", _timeout)

    result = compat_api.call_model("Please output json列表 directly")

    assert result == "[]"
    assert calls["count"] == 1


def test_falls_back_on_error(monkeypatch) -> None:
    monkeypatch.setenv("HOUYI_BENCH2_FACT_CALL_RETRIES", "1")

    def _raise(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(compat_api, "_run_chat", _raise)

    result = compat_api.call_model("Please output json列表 directly")

    assert result == "[]"


class TestTruncateValidateReference:
    def test_truncates_large_reference(self) -> None:
        big_ref = "x" * 20_000
        prompt = f"<reference>{big_ref}</reference>\n<statements>1. fact</statements>"
        result = compat_api._truncate_validate_reference(prompt)
        assert len(result) < len(prompt)
        assert "<reference>" in result
        assert "</reference>" in result
        assert "[... truncated ...]" in result
        # Verify the content between tags is <= limit + truncation marker
        import re

        match = re.search(r"<reference>(.*?)</reference>", result, re.DOTALL)
        assert match is not None
        assert len(match.group(1)) <= compat_api._MAX_VALIDATE_REF_CHARS + 50

    def test_preserves_short_reference(self) -> None:
        short_ref = "Short content"
        prompt = f"<reference>{short_ref}</reference>\n<statements>1. fact</statements>"
        assert compat_api._truncate_validate_reference(prompt) == prompt

    def test_no_reference_tags(self) -> None:
        prompt = "No reference tags here"
        assert compat_api._truncate_validate_reference(prompt) == prompt

    def test_extracts_validate_statements(self) -> None:
        prompt = (
            "<reference>ref</reference>\n<statements>\n1. First fact\n2. Second fact\n</statements>"
        )
        assert compat_api._extract_validate_statements(prompt) == ["First fact", "Second fact"]

    def test_keeps_relevant_segments(self) -> None:
        noisy = "catalog entry\n" * 900
        target = "Household consumption dropped by 72% in 2020 according to PPLR analysis.\n" * 20
        tail = "bibliography\n" * 900
        ref = noisy + "\n\n" + target + "\n\n" + tail
        trimmed = compat_api._select_validate_reference(
            ref, ["Household consumption dropped by 72% in 2020"]
        )
        assert len(trimmed) <= compat_api._MAX_VALIDATE_REF_CHARS
        assert "72%" in trimmed
        assert "PPLR" in trimmed

    def test_falls_back_head_tail(self) -> None:
        ref = ("lead\n" * 2500) + ("tail signal\n" * 2500)
        trimmed = compat_api._select_validate_reference(ref, ["unmatched keyword"])
        assert len(trimmed) <= compat_api._MAX_VALIDATE_REF_CHARS
        assert "[... truncated ...]" in trimmed
        assert trimmed.startswith("lead")
        assert "tail signal" in trimmed

    def test_salvages_short_unknown(self, monkeypatch) -> None:
        prompt = (
            "<reference>Evidence with 72% value and household consumption trend.</reference>\n"
            "<statements>\n1. Household consumption dropped by 72%.\n</statements>"
        )
        responses = iter(
            [
                '[{"idx":1,"result":"unknown"}]',
                '[{"idx":1,"result":"supported"}]',
            ]
        )
        monkeypatch.setattr(compat_api, "_run_chat", lambda *args, **kwargs: next(responses))

        result = compat_api.call_model(prompt)

        assert json.loads(result) == [{"idx": 1, "result": "supported"}]

    def test_skips_salvage_for_truncated(self, monkeypatch) -> None:
        ref = "x" * 20_000
        prompt = f"<reference>{ref}</reference>\n<statements>\n1. Fact one\n</statements>"
        calls = {"count": 0}

        def _run(*args, **kwargs):
            calls["count"] += 1
            return '[{"idx":1,"result":"unknown"}]'

        monkeypatch.setattr(compat_api, "_run_chat", _run)

        compat_api.call_model(prompt)

        assert calls["count"] == 1

    def test_skips_salvage_for_supported(self, monkeypatch) -> None:
        prompt = (
            "<reference>Evidence with 72% value and household consumption trend.</reference>\n"
            "<statements>\n1. Household consumption dropped by 72%.\n</statements>"
        )
        calls = {"count": 0}

        def _run(*args, **kwargs):
            calls["count"] += 1
            return '[{"idx":1,"result":"supported"}]'

        monkeypatch.setattr(compat_api, "_run_chat", _run)

        result = compat_api.call_model(prompt)

        assert json.loads(result) == [{"idx": 1, "result": "supported"}]
        assert calls["count"] == 1


class TestIsInaccessible:
    def test_short_content_flagged(self) -> None:
        assert compat_api._is_inaccessible("too short") is True

    def test_scrape_failed_signal(self) -> None:
        content = "scrape failed: HTTP Error 403: Forbidden" + " x" * 200
        assert compat_api._is_inaccessible(content) is True

    def test_captcha_signal(self) -> None:
        content = "Our systems have presented this CAPTCHA challenge" + " x" * 200
        assert compat_api._is_inaccessible(content) is True

    def test_login_wall_signal(self) -> None:
        content = "Log in or register to access precise data." + " x" * 200
        assert compat_api._is_inaccessible(content) is True

    def test_google_scholar_only(self) -> None:
        content = (
            "Glassman, Ronald M. (1997), The New Middle Class\n"
            "[Google Scholar](https://scholar.google.com/scholar_lookup?title=foo)\n"
        ) * 10
        assert compat_api._is_inaccessible(content) is True

    def test_youtube_nav_only(self) -> None:
        content = (
            "[About](https://www.youtube.com/about/)"
            "[Press](https://www.youtube.com/about/press/)"
            "[Copyright](https://www.youtube.com/about/copyright/)"
            "[Contact us](/t/contact_us/)[Creators](https://www.youtube.com/creators/)"
            "[Advertise](https://www.youtube.com/ads/)[Terms](/t/terms)"
        ) * 5
        assert compat_api._is_inaccessible(content) is True

    def test_valid_content_not_flagged(self) -> None:
        content = (
            "China's middle class has grown significantly over the past two decades. "
            "According to the National Bureau of Statistics, per capita disposable income "
            "reached 43,377 yuan in 2025, a nominal increase of 5.3 percent year-on-year. "
            "The urban-rural income ratio narrowed to 2.21, down from 2.25 the previous year."
        ) * 5
        assert compat_api._is_inaccessible(content) is False
