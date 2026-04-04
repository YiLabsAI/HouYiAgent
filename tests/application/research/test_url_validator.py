"""Unit tests for URLValidator — async URL reachability verification."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from houyi.application.research.url_validator import (
    URLValidationReport,
    URLValidationResult,
    URLValidator,
    _safe_encode_url,
)


class TestSafeEncodeUrl:
    def test_ascii_url_unchanged(self):
        url = "https://example.com/page?q=test"
        assert _safe_encode_url(url) == url

    def test_unicode_path_encoded(self):
        url = "https://example.com/搜索/结果"
        encoded = _safe_encode_url(url)
        assert "example.com" in encoded
        assert "搜索" not in encoded

    def test_unicode_query_encoded(self):
        url = "https://example.com/search?q=冯嘉"
        encoded = _safe_encode_url(url)
        assert "冯嘉" not in encoded

    def test_empty_path(self):
        url = "https://example.com"
        assert _safe_encode_url(url) == "https://example.com"


class TestURLValidationReport:
    def test_defaults(self):
        report = URLValidationReport()
        assert report.total == 0
        assert report.error_rate == 0.0

    def test_error_rate_calculation(self):
        report = URLValidationReport(total=10, reachable=7, unreachable=3, error_rate=0.3)
        assert report.error_rate == 0.3


class TestURLValidator:
    async def test_empty_urls(self):
        validator = URLValidator()
        report = await validator.validate([])
        assert report.total == 0

    async def test_deduplicates_urls(self):
        validator = URLValidator()
        with patch.object(validator, "_check_url", new_callable=AsyncMock) as mock:
            mock.return_value = URLValidationResult(
                url="https://a.com", reachable=True, status_code=200
            )
            report = await validator.validate(
                [
                    "https://a.com",
                    "https://a.com",
                    "https://a.com",
                ]
            )
            assert mock.call_count == 1
            assert report.total == 1

    async def test_aggregation(self):
        validator = URLValidator()
        results = [
            URLValidationResult(url="https://ok.com", reachable=True, status_code=200),
            URLValidationResult(url="https://bad.com", reachable=False, error="404"),
        ]
        with patch.object(validator, "_check_url", new_callable=AsyncMock, side_effect=results):
            report = await validator.validate(["https://ok.com", "https://bad.com"])
        assert report.total == 2
        assert report.reachable == 1
        assert report.unreachable == 1
        assert report.error_rate == 0.5

    async def test_timeout_handled(self):
        validator = URLValidator(timeout=0.001)

        async def _slow_check(url: str) -> URLValidationResult:
            await asyncio.sleep(10)
            return URLValidationResult(url=url, reachable=True)

        with patch.object(validator, "_head_request", side_effect=_slow_check):
            report = await validator.validate(["https://slow.com"])
        assert report.unreachable == 1
        assert report.results[0].error == "timeout"

    async def test_concurrency_limit(self):
        max_conc = 2
        validator = URLValidator(max_concurrent=max_conc)
        active = {"count": 0, "max": 0}

        async def _counting_head(url: str) -> URLValidationResult:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.02)
            active["count"] -= 1
            return URLValidationResult(url=url, reachable=True, status_code=200)

        with patch.object(validator, "_head_request", side_effect=_counting_head):
            await validator.validate([f"https://{i}.com" for i in range(5)])

        assert active["max"] <= max_conc

    async def test_check_url_generic_exception(self):
        validator = URLValidator()

        async def _explode(url: str) -> URLValidationResult:
            raise RuntimeError("unexpected failure")

        validator._head_request = _explode  # type: ignore[assignment]
        result = await validator._check_url("https://fail.com")
        assert result.reachable is False
        assert "unexpected failure" in result.error


class TestHeadRequest:
    async def test_head_request_success(self):
        from unittest.mock import MagicMock

        validator = URLValidator()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await validator._head_request("https://example.com")

        assert result.reachable is True
        assert result.status_code == 200

    async def test_head_request_fallback_to_get(self):
        from unittest.mock import MagicMock

        validator = URLValidator()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        calls: list[str] = []

        def _mock_urlopen(req, **kwargs):
            calls.append(req.get_method())
            if req.get_method() == "HEAD":
                raise OSError("Connection refused")
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
            result = await validator._head_request("https://example.com")

        assert result.reachable is True
        assert result.status_code == 200
        assert calls == ["HEAD", "GET"]


class TestLocalUrlFilter:
    async def test_localhost_marked_unreachable(self):
        v = URLValidator()
        report = await v.validate(["http://localhost:3000/page", "http://127.0.0.1/foo"])
        assert report.total == 2
        assert report.unreachable == 2
        for r in report.results:
            assert r.reachable is False
            assert r.error == "local_or_invalid_scheme"

    async def test_file_scheme_rejected(self):
        v = URLValidator()
        report = await v.validate(["file:///etc/passwd"])
        assert report.unreachable == 1
