from __future__ import annotations

from houyi.application.context.token_budget_policy import TokenBudgetPolicy


class TestTokenBudgetPolicy:
    def test_defaults(self):
        policy = TokenBudgetPolicy(default_output_budget=2048)
        decision = policy.decide(context_window=8192)
        assert decision.context_window == 8192
        assert decision.output_budget == 2048
        assert decision.input_budget == 6144
        assert decision.should_set_max_tokens is False
        assert decision.max_tokens_to_send is None

    def test_reasoning(self):
        policy = TokenBudgetPolicy(default_output_budget=2048, default_answer_reserve=512)
        decision = policy.decide(
            context_window=8192,
            enable_reasoning=True,
            provider_supports_reasoning_budget=False,
        )
        assert decision.answer_reserve == 512
        assert decision.reasoning_budget == 768
        assert decision.input_budget == 6144

    def test_user_max_tokens(self):
        policy = TokenBudgetPolicy(default_output_budget=2048)
        decision = policy.decide(
            context_window=8192,
            requested_max_tokens=1024,
            max_output_limit=4096,
        )
        assert decision.output_budget == 1024
        assert decision.should_set_max_tokens is True
        assert decision.max_tokens_to_send == 1024
        assert decision.max_tokens_source == "user"
