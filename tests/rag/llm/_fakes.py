from __future__ import annotations

from typing import Any


class FakeAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0
        self.call_count = 0
        self.messages_history: list[Any] = []

    async def chat(
        self,
        messages: list[Any],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.messages_history.append(messages)

        response_content = self._responses[self._index % len(self._responses)]
        self._index += 1

        class MockResponse:
            content = response_content

        return MockResponse()
