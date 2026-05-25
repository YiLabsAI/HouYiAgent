"""URL Validator — async verification of reference URL accessibility.

After report generation, validates that cited URLs are reachable (HTTP 2xx).
Unreachable URLs are flagged so the UI can display warnings and the report
generator can annotate them.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10.0
_DEFAULT_CONCURRENCY = 10


class URLValidationResult(BaseModel):
    """Result of validating a single URL."""

    url: str
    reachable: bool = False
    status_code: int | None = None
    error: str | None = None


class URLValidationReport(BaseModel):
    """Aggregated URL validation results."""

    total: int = 0
    reachable: int = 0
    unreachable: int = 0
    error_rate: float = 0.0
    results: list[URLValidationResult] = Field(default_factory=list)


class URLValidator:
    """Validates URL accessibility with concurrent HTTP HEAD requests.

    Uses asyncio.Semaphore to limit concurrency and prevent
    overwhelming target servers.
    """

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_concurrent: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    _LOCAL_PATTERNS = ("localhost", "127.0.0.1", "0.0.0.0", "file://", "file:///")

    async def validate(self, urls: list[str]) -> URLValidationReport:
        """Validate all URLs concurrently and return an aggregated report."""
        if not urls:
            return URLValidationReport()

        unique = list(dict.fromkeys(urls))
        tasks = []
        for url in unique:
            low = url.lower()
            if any(p in low for p in self._LOCAL_PATTERNS) or not low.startswith(
                ("http://", "https://")
            ):
                tasks.append(self._bogus(url))
            else:
                tasks.append(self._check_url(url))
        results = await asyncio.gather(*tasks)

        reachable = sum(1 for r in results if r.reachable)
        total = len(results)
        return URLValidationReport(
            total=total,
            reachable=reachable,
            unreachable=total - reachable,
            error_rate=round((total - reachable) / max(total, 1), 3),
            results=list(results),
        )

    @staticmethod
    async def _bogus(url: str) -> URLValidationResult:
        return URLValidationResult(url=url, reachable=False, error="local_or_invalid_scheme")

    async def _check_url(self, url: str) -> URLValidationResult:
        """Check a single URL via HTTP HEAD, falling back to GET."""
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._head_request(url),
                    timeout=self._timeout,
                )
            except TimeoutError:
                return URLValidationResult(url=url, reachable=False, error="timeout")
            except Exception as exc:
                return URLValidationResult(url=url, reachable=False, error=str(exc)[:200])

    async def _head_request(self, url: str) -> URLValidationResult:
        """Perform async HTTP HEAD using urllib (thread pool)."""
        import urllib.request

        encoded_url = _safe_encode_url(url)

        def _do_head() -> tuple[bool, int]:
            req = urllib.request.Request(encoded_url, method="HEAD")
            req.add_header("User-Agent", "HouYi-URLValidator/1.0")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return True, resp.status
            except urllib.error.HTTPError as e:
                return e.code < 400, e.code
            except Exception:
                req2 = urllib.request.Request(encoded_url, method="GET")
                req2.add_header("User-Agent", "HouYi-URLValidator/1.0")
                with urllib.request.urlopen(req2, timeout=self._timeout) as resp:
                    return True, resp.status

        ok, code = await asyncio.to_thread(_do_head)
        return URLValidationResult(url=url, reachable=ok, status_code=code)


def _safe_encode_url(url: str) -> str:
    """Encode non-ASCII characters in URL path/query while preserving structure."""
    from urllib.parse import quote, urlparse, urlunparse

    parsed = urlparse(url)
    safe_path = quote(parsed.path, safe="/:@!$&'()*+,;=-._~")
    safe_query = quote(parsed.query, safe="/:@!$&'()*+,;=-._~?=")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            safe_path,
            parsed.params,
            safe_query,
            parsed.fragment,
        )
    )
