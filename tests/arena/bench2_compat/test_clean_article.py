from __future__ import annotations

from houyi.arena.bench2_compat.clean_article import ArticleCleaner


class _TimeoutAgent:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        raise TimeoutError("timed out")


def test_clean_single_failfast() -> None:
    agent = _TimeoutAgent()
    cleaner = ArticleCleaner(agent)
    chunk_calls = {"count": 0}
    original_article = "A" * 1000

    cleaner._get_clean_prompt = lambda language="zh": "{article}"

    def _chunk(*args, **kwargs):
        chunk_calls["count"] += 1
        return "should not run"

    cleaner.chunk_clean_article = _chunk
    result = cleaner.clean_single(
        {
            "id": "1",
            "prompt": "Q",
            "article": original_article,
        },
        max_retries=5,
        language="zh",
    )

    assert result == {"id": "1", "prompt": "Q", "article": original_article}
    assert agent.calls == 1
    assert chunk_calls["count"] == 0
