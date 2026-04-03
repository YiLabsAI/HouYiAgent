"""ReminderInjector — inject critical instructions at context tail.

Leverages Transformer attention bias: tokens near the end of context
receive stronger attention. By placing key reminders (citation format,
language, depth constraints) as the last system message, we ensure the
agent follows them even under long-context pressure.

"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Reminder(BaseModel):
    """A single reminder entry."""

    key: str
    text: str
    priority: int = 5


class ReminderInjector:
    """Manages and injects tail-end reminders into agent message context.

    Reminders are sorted by priority (higher first) and injected as a
    single system message at the end of the message list, just before
    the final user message.
    """

    def __init__(self, reminders: list[Reminder] | None = None) -> None:
        self._reminders: list[Reminder] = list(reminders or [])

    def add(self, key: str, text: str, *, priority: int = 5) -> None:
        for r in self._reminders:
            if r.key == key:
                r.text = text
                r.priority = priority
                return
        self._reminders.append(Reminder(key=key, text=text, priority=priority))

    def remove(self, key: str) -> bool:
        before = len(self._reminders)
        self._reminders = [r for r in self._reminders if r.key != key]
        return len(self._reminders) < before

    def clear(self) -> None:
        self._reminders.clear()

    @property
    def reminders(self) -> list[Reminder]:
        return sorted(self._reminders, key=lambda r: r.priority, reverse=True)

    def inject(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject reminders into the message list.

        Inserts a system message containing all active reminders right
        before the last non-system message, maximizing recency attention.
        Returns a new list (does not mutate the input).
        """
        if not self._reminders:
            return messages

        sorted_reminders = self.reminders
        reminder_text = "IMPORTANT REMINDERS (follow strictly):\n" + "\n".join(
            f"- {r.text}" for r in sorted_reminders
        )

        result = list(messages)
        insert_idx = len(result)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") != "system":
                insert_idx = i
                break

        result.insert(insert_idx, {"role": "system", "content": reminder_text})
        return result


# -- Pre-built reminder sets ------------------------------------------------

CITATION_REMINDER = Reminder(
    key="citation_format",
    text="Every factual claim MUST have an inline citation [ref_XXX]. Never fabricate references.",
    priority=9,
)

LANGUAGE_REMINDER = Reminder(
    key="language_match",
    text="Write the report in the SAME language as the user's query. Do not mix languages.",
    priority=8,
)

DEPTH_REMINDER = Reminder(
    key="analysis_depth",
    text="Synthesize and analyze findings — do NOT simply list or summarize sources.",
    priority=7,
)

JSON_REMINDER = Reminder(
    key="json_output",
    text="Respond ONLY with valid JSON. No markdown, no extra text before or after the JSON.",
    priority=10,
)


def default_research_reminders() -> ReminderInjector:
    """Pre-configured reminders for Deep Research agents."""
    injector = ReminderInjector()
    for r in [CITATION_REMINDER, LANGUAGE_REMINDER, DEPTH_REMINDER]:
        injector.add(r.key, r.text, priority=r.priority)
    return injector
