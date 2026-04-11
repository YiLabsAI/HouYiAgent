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


def test_extracts_title_links_without_polluting_fact(monkeypatch) -> None:
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
