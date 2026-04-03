"""Unit tests for ReminderInjector — context tail-end reminder injection."""

from __future__ import annotations

from houyi.application.context.reminders import (
    CITATION_REMINDER,
    DEPTH_REMINDER,
    JSON_REMINDER,
    LANGUAGE_REMINDER,
    Reminder,
    ReminderInjector,
    default_research_reminders,
)


class TestReminderManagement:
    def test_add_and_list(self):
        inj = ReminderInjector()
        inj.add("key1", "text1", priority=5)
        inj.add("key2", "text2", priority=9)
        assert len(inj.reminders) == 2
        assert inj.reminders[0].priority == 9

    def test_add_updates_existing(self):
        inj = ReminderInjector()
        inj.add("key1", "old", priority=3)
        inj.add("key1", "new", priority=7)
        assert len(inj.reminders) == 1
        assert inj.reminders[0].text == "new"
        assert inj.reminders[0].priority == 7

    def test_remove_existing(self):
        inj = ReminderInjector()
        inj.add("key1", "text1")
        assert inj.remove("key1") is True
        assert len(inj.reminders) == 0

    def test_remove_nonexistent(self):
        inj = ReminderInjector()
        assert inj.remove("nope") is False

    def test_clear(self):
        inj = ReminderInjector()
        inj.add("a", "x")
        inj.add("b", "y")
        inj.clear()
        assert len(inj.reminders) == 0

    def test_sorted_by_priority(self):
        inj = ReminderInjector(
            [
                Reminder(key="low", text="low", priority=1),
                Reminder(key="high", text="high", priority=10),
                Reminder(key="mid", text="mid", priority=5),
            ]
        )
        keys = [r.key for r in inj.reminders]
        assert keys == ["high", "mid", "low"]


class TestInject:
    def test_injects_before_last_user_message(self):
        inj = ReminderInjector()
        inj.add("r1", "Remember this!")
        messages = [
            {"role": "system", "content": "You are an AI."},
            {"role": "user", "content": "Hello"},
        ]
        result = inj.inject(messages)
        assert len(result) == 3
        assert result[1]["role"] == "system"
        assert "Remember this!" in result[1]["content"]
        assert result[2]["role"] == "user"

    def test_does_not_mutate_original(self):
        inj = ReminderInjector()
        inj.add("r1", "text")
        messages = [{"role": "user", "content": "Hi"}]
        result = inj.inject(messages)
        assert len(messages) == 1
        assert len(result) == 2

    def test_no_reminders_returns_same(self):
        inj = ReminderInjector()
        messages = [{"role": "user", "content": "Hi"}]
        result = inj.inject(messages)
        assert result is messages

    def test_multiple_reminders_joined(self):
        inj = ReminderInjector()
        inj.add("a", "First reminder", priority=9)
        inj.add("b", "Second reminder", priority=5)
        result = inj.inject([{"role": "user", "content": "Q"}])
        reminder_msg = result[0]
        assert "First reminder" in reminder_msg["content"]
        assert "Second reminder" in reminder_msg["content"]

    def test_multipart_conversation(self):
        inj = ReminderInjector()
        inj.add("r1", "Important!")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
        ]
        result = inj.inject(messages)
        assert len(result) == 5
        assert result[-1]["role"] == "user"
        assert result[-2]["role"] == "system"
        assert "Important!" in result[-2]["content"]


class TestDefaultReminders:
    def test_research_reminders_present(self):
        inj = default_research_reminders()
        keys = {r.key for r in inj.reminders}
        assert "citation_format" in keys
        assert "language_match" in keys
        assert "analysis_depth" in keys

    def test_prebuilt_constants(self):
        assert CITATION_REMINDER.priority > LANGUAGE_REMINDER.priority
        assert JSON_REMINDER.priority > CITATION_REMINDER.priority
        assert DEPTH_REMINDER.key == "analysis_depth"
