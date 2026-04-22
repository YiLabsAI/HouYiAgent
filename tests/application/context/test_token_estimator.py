"""Unit tests for houyi.application.context.token_estimator.TokenEstimator."""

from __future__ import annotations

from houyi.adapters.llm.models import (
    DEFAULT_MODEL,
    MODEL_CONTEXT_WINDOWS,
    normalize_model_id,
    resolve_model_context_window,
)
from houyi.application.context.token_estimator import TokenEstimator


class TestTokenEstimatorInit:
    """Test TokenEstimator initialization and configuration."""

    def test_default_model(self):
        est = TokenEstimator()
        assert est.model == DEFAULT_MODEL
        assert est.context_window == MODEL_CONTEXT_WINDOWS[DEFAULT_MODEL]

    def test_custom_model(self):
        est = TokenEstimator(model="gpt-4o")
        assert est.model == "gpt-4o"
        assert est.context_window == 128000

    def test_context_window_override(self):
        est = TokenEstimator(context_window_override=8192)
        assert est.context_window == 8192

    def test_output_reserve_default(self):
        est = TokenEstimator()
        assert est.output_reserve == 4096

    def test_max_input_tokens(self):
        est = TokenEstimator(context_window_override=10000, output_reserve=2000)
        assert est.max_input_tokens == 8000

    def test_unknown_model_fallback(self):
        est = TokenEstimator(model="unknown-model-xyz")
        assert est.context_window == 8192  # DEFAULT_CONTEXT_WINDOW fallback

    def test_minimax_arge_context(self):
        est = TokenEstimator(model="Pro/MiniMaxAI/MiniMax-M2.5")
        assert est.context_window == 1_000_000

    def test_gemini_large_context(self):
        est = TokenEstimator(model="gemini-3.1-pro-preview")
        assert est.context_window == 1_048_576

    def test_glm_context_window(self):
        est = TokenEstimator(model="zai-org/glm-5")
        assert est.context_window == 128_000

    def test_kimi_canonical_model(self):
        assert normalize_model_id("Pro/moonshotai/Kimi-K2.5") == "Pro/moonshotai/Kimi-K2.5"
        assert resolve_model_context_window("Pro/moonshotai/Kimi-K2.5") == 131_072


class TestTokenEstimatorCounting:
    """Test token counting methods."""

    def test_count_text_empty(self):
        est = TokenEstimator()
        assert est.count_text("") == 0

    def test_count_text_nonempty(self):
        est = TokenEstimator()
        count = est.count_text("Hello, world!")
        assert count > 0

    def test_count_text_multilingual(self):
        est = TokenEstimator()
        count = est.count_text("Hello world, this is a longer sentence for token estimation.")
        assert count > 0

    def test_count_message_basic(self):
        est = TokenEstimator()
        msg = {"role": "user", "content": "Hello"}
        count = est.count_message(msg)
        # At minimum: 4 overhead + role + content
        assert count >= 5

    def test_count_message_empty_content(self):
        est = TokenEstimator()
        msg = {"role": "assistant", "content": ""}
        count = est.count_message(msg)
        # Should still have overhead + role tokens
        assert count >= 4

    def test_count_messages_list(self):
        est = TokenEstimator()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        count = est.count_messages(messages)
        # 3 base overhead + sum of individual messages
        assert count > 3

    def test_count_messages_empty(self):
        est = TokenEstimator()
        count = est.count_messages([])
        assert count == 3  # base overhead only

    def test_count_multimodal_text(self):
        est = TokenEstimator()
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
        count = est.count_message(msg)
        assert count > 4


class TestTokenEstimatorBudget:
    """Test budget-related methods."""

    def test_would_exceed_within_budget(self):
        est = TokenEstimator(context_window_override=100000, output_reserve=4096)
        messages = [{"role": "user", "content": "Short message"}]
        assert est.would_exceed(messages) is False

    def test_would_exceed_over_budget(self):
        est = TokenEstimator(context_window_override=100, output_reserve=90)
        # max_input_tokens = 10, any real message will exceed
        messages = [{"role": "user", "content": "This message is definitely longer than 10 tokens"}]
        assert est.would_exceed(messages) is True

    def test_would_exceed_with_threshold(self):
        est = TokenEstimator(context_window_override=1000, output_reserve=100)
        # max_input_tokens = 900; a short message is ~8 tokens
        messages = [{"role": "user", "content": "Short"}]
        # With threshold 0.005, budget = 900 * 0.005 = 4.5 tokens; message (~8) exceeds
        assert est.would_exceed(messages, threshold_ratio=0.005) is True


class TestTokenEstimatorFallback:
    """Test fallback counting when tiktoken is unavailable."""

    def test_fallback_count_english(self):
        count = TokenEstimator._fallback_count("Hello world, this is a test.")
        assert count > 0

    def test_fallback_count_long_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        count = TokenEstimator._fallback_count(text)
        assert count >= 10

    def test_fallback_count_mixed(self):
        # Mixed short and long words
        count = TokenEstimator._fallback_count("Hello world, testing token estimation!")
        assert count > 0

    def test_fallback_count_empty(self):
        assert TokenEstimator._fallback_count("") == 0

    def test_fallback_english_ratio(self):
        # English: ~4 chars per token
        # 100 chars of English should yield ~25 tokens
        text = "a" * 100
        count = TokenEstimator._fallback_count(text)
        assert 20 <= count <= 30, f"Expected ~25 tokens for 100 English chars, got {count}"

    def test_fallback_english_sentence_ratio(self):
        # "The quick brown fox jumps over the lazy dog." = 44 chars
        # At ~4 chars/token => ~11 tokens + 1 (min for 0 CJK) = ~12
        text = "The quick brown fox jumps over the lazy dog."
        count = TokenEstimator._fallback_count(text)
        assert 8 <= count <= 16, f"Expected ~11 tokens for 44 English chars, got {count}"
