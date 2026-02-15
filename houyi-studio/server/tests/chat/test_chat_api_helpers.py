"""Tests for provider_service helper functions: sanitize_error, _is_vertex_provider."""

from houyi_studio.server.chat.provider_service import _is_vertex_provider, sanitize_error


class TestSanitizeError:
    """Test HTML sanitization for error messages."""

    def test_plain_text_unchanged(self):
        assert sanitize_error("simple error") == "simple error"

    def test_strips_html_tags(self):
        assert sanitize_error("<b>bold</b> text") == "bold text"

    def test_strips_style_block_with_content(self):
        html = "<html><head><style>*{margin:0;padding:0}html,code{font:15px}</style></head><body>Error 404</body></html>"
        result = sanitize_error(html)
        assert "{margin" not in result
        assert "padding" not in result
        assert "404" in result

    def test_strips_script_block_with_content(self):
        html = "<html><script>alert('xss')</script><body>Error</body></html>"
        result = sanitize_error(html)
        assert "alert" not in result
        assert "Error" in result

    def test_google_cloud_404_page(self):
        """Real Google Cloud 404 HTML response should be sanitized to readable text."""
        html = (
            "<!DOCTYPE html><html lang=en><meta charset=utf-8>"
            "<meta name=viewport content='initial-scale=1, minimum-scale=1, width=device-width'>"
            "<title>Error 404 (Not Found)!!1</title>"
            "<style>*{margin:0;padding:0}html,code{font:15px/22px arial,sans-serif}"
            "html{background:#fff;color:#222;padding:15px}"
            "body{margin:7% auto 0;max-width:390px;min-height:180px;padding:30px 0 15px}"
            "</style>"
            "<p><b>404.</b> <ins>That's an error.</ins>"
            "<p>The requested URL was not found on this server. <ins>That's all we know.</ins>"
        )
        result = sanitize_error(html)
        # CSS content must not leak
        assert "{margin" not in result
        assert "padding:0}" not in result
        assert "font:15px" not in result
        # Meaningful text should remain
        assert "404" in result

    def test_truncation(self):
        long_text = "x" * 500
        result = sanitize_error(long_text, max_len=200)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    def test_collapses_whitespace(self):
        assert sanitize_error("a   b\n\nc") == "a b c"

    def test_empty_string(self):
        assert sanitize_error("") == ""


class TestIsVertexProvider:
    """Test Vertex AI provider detection."""

    def test_vertex_prefix(self):
        assert _is_vertex_provider("vertex-12345") is True

    def test_vertex_exact(self):
        assert _is_vertex_provider("vertex") is True

    def test_non_vertex_id(self):
        assert _is_vertex_provider("siliconflow-123") is False

    def test_empty_id(self):
        assert _is_vertex_provider("") is False

    def test_aiplatform_domain(self):
        assert _is_vertex_provider("custom-123", "https://aiplatform.googleapis.com") is True

    def test_generativelanguage_domain(self):
        assert (
            _is_vertex_provider(
                "google-ai-456", "https://generativelanguage.googleapis.com/v1beta/openai"
            )
            is True
        )

    def test_openai_domain_not_vertex(self):
        assert _is_vertex_provider("openai-789", "https://api.openai.com/v1") is False

    def test_vertex_id_overrides_non_vertex_url(self):
        assert _is_vertex_provider("vertex-abc", "https://api.openai.com/v1") is True

    def test_non_vertex_id_with_vertex_url(self):
        assert _is_vertex_provider("custom-xyz", "https://aiplatform.googleapis.com/v1") is True
