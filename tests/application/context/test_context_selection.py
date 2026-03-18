from houyi.application.context.context_selection import build_default_context_selection_policy


class TestContextSelectionPolicy:
    def test_policy_defaults(self):
        policy = build_default_context_selection_policy()

        assert policy.policy_name == "chat_default"
        assert policy.allow_memory is True
        assert policy.allow_summaries is True
        assert policy.allow_tool_summaries is True
        assert policy.allow_pinned is True
