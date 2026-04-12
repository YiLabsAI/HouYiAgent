from __future__ import annotations

import asyncio
import json
import os
import re
from functools import lru_cache
from typing import Any

from houyi.adapters.llm.factory import LLMAdapterFactory
from houyi.rag.llm.json_utils import parse_embedded_json
from houyi.skills.web_search.content_fetchers import JinaContentFetcher, ReadabilityContentFetcher

_DEFAULT_PROVIDER = "siliconflow"
_BENCH2_PROVIDER_ENV = "HOUYI_BENCH2_PROVIDER"
_RACE_MODEL_ENV = "HOUYI_BENCH2_RACE_MODEL"
_FACT_MODEL_ENV = "HOUYI_BENCH2_FACT_MODEL"
_FACT_MAX_TOKENS_ENV = "HOUYI_BENCH2_FACT_MAX_TOKENS"
_FACT_MAX_RETRIES_ENV = "HOUYI_BENCH2_FACT_CALL_RETRIES"
_FACT_TIMEOUT_ENV = "HOUYI_BENCH2_FACT_TIMEOUT_SECONDS"
_DEFAULT_FACT_MAX_TOKENS = 6000
_DEFAULT_FACT_CALL_RETRIES = 3
_DEFAULT_FACT_TIMEOUT_SECONDS = 90
_EXTRACT_PROMPT_FEATURE_A = '"ref_idx"'
_EXTRACT_PROMPT_FEATURE_B = '"url"'
_DEDUP_PROMPT_FEATURE = "List(int)"
_INACCESSIBLE_SIGNALS = (
    "scrape failed:",
    "captcha",
    "log in or register",
    "please log in",
    "sign in to",
    "please solve",
    "we apologize for the inconvenience",
    "access denied",
    "403 forbidden",
    "404 not found",
    "page not found",
    "robot",
    "subscribe to read",
    "subscription required",
    "create an account",
    "register to access",
    "google scholar](https://scholar.google.com/scholar_lookup",
)
_MIN_ACCESSIBLE_CONTENT_LEN = 300
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_SENTENCE_BREAKS = ".!?\n"
_NUMBERED_STMT_RE = re.compile(r"^\s*(\d+)\.\s+(.+\S)\s*$")


def _resolve_provider() -> str:
    return (
        os.getenv(_BENCH2_PROVIDER_ENV)
        or os.getenv("LLM_PROVIDER")
        or os.getenv("DEFAULT_LLM_PROVIDER")
        or _DEFAULT_PROVIDER
    )


@lru_cache(maxsize=4)
def _get_adapter(provider: str):
    return LLMAdapterFactory.create(provider=provider)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _run_chat(
    prompt: str,
    *,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    provider = _resolve_provider()
    adapter = _get_adapter(provider)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async def _call() -> str:
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        request = adapter.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            **kwargs,
        )
        response = (
            await asyncio.wait_for(request, timeout=timeout_seconds)
            if timeout_seconds
            else await request
        )
        return response.content or ""

    return asyncio.run(_call())


class AIClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.model = model or os.getenv(_RACE_MODEL_ENV)

    def generate(self, user_prompt: str, system_prompt: str = "", model: str | None = None) -> str:
        return _run_chat(user_prompt, system_prompt=system_prompt, model=model or self.model)


def call_model(user_prompt: str) -> str:
    extracted = _extract_inline_citations(user_prompt)
    if extracted is not None:
        return extracted

    deduped = _deduplicate_statements(user_prompt)
    if deduped is not None:
        return deduped

    model = os.getenv(_FACT_MODEL_ENV)
    max_tokens = _env_int(_FACT_MAX_TOKENS_ENV, _DEFAULT_FACT_MAX_TOKENS)
    retries = _env_int(_FACT_MAX_RETRIES_ENV, _DEFAULT_FACT_CALL_RETRIES)
    timeout_seconds = _env_int(_FACT_TIMEOUT_ENV, _DEFAULT_FACT_TIMEOUT_SECONDS)
    for _ in range(retries):
        try:
            raw = _run_chat(
                user_prompt,
                model=model,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            break
        except Exception:
            continue
        normalized = _normalize_json_output(raw)
        if normalized is not None:
            return normalized
    return _fallback_json_for_prompt(user_prompt)


def _is_extract_prompt(user_prompt: str) -> bool:
    return _EXTRACT_PROMPT_FEATURE_A in user_prompt and _EXTRACT_PROMPT_FEATURE_B in user_prompt


def _is_dedup_prompt(user_prompt: str) -> bool:
    return _DEDUP_PROMPT_FEATURE in user_prompt


def _extract_inline_citations(user_prompt: str) -> str | None:
    if not _is_extract_prompt(user_prompt):
        return None
    citations = _parse_inline_citations(user_prompt)
    return json.dumps(citations, ensure_ascii=False)


def _deduplicate_statements(user_prompt: str) -> str | None:
    if not _is_dedup_prompt(user_prompt):
        return None

    deduped_idx: list[int] = []
    seen: set[str] = set()
    for line in user_prompt.splitlines():
        match = _NUMBERED_STMT_RE.match(line)
        if match is None:
            continue
        index = int(match.group(1))
        statement = re.sub(r"\s+", " ", match.group(2)).strip()
        if statement in seen:
            continue
        seen.add(statement)
        deduped_idx.append(index)
    return json.dumps(deduped_idx, ensure_ascii=False)


def _parse_inline_citations(prompt_or_article: str) -> list[dict[str, Any]]:
    article_start = _find_article_start(prompt_or_article)
    article = prompt_or_article[article_start:]
    body = article.split("\n## References", 1)[0]
    citations: list[dict[str, Any]] = []
    for match in _LINK_PATTERN.finditer(body):
        url = _trim_citation_url(match.group(2))
        if not url.startswith("http://") and not url.startswith("https://"):
            continue
        fact = _citation_fact(body, match.start(), match.end())
        if not fact:
            continue
        citations.append({"fact": fact, "ref_idx": 0, "url": url})
    return citations


def _find_article_start(text: str) -> int:
    last_schema_pos = text.rfind('"ref_idx"')
    if last_schema_pos < 0:
        return 0
    heading_pos = text.find("\n# ", last_schema_pos)
    if heading_pos < 0:
        return 0
    return heading_pos + 1


def _citation_fact(text: str, start: int, end: int) -> str:
    left = _sentence_start(text, start)
    right = _sentence_end(text, end)
    fragment = text[left:right].strip()
    return _clean_fact(fragment)


def _sentence_start(text: str, index: int) -> int:
    cursor = index - 1
    while cursor >= 0 and text[cursor] not in _SENTENCE_BREAKS:
        cursor -= 1
    return cursor + 1


def _sentence_end(text: str, index: int) -> int:
    cursor = index
    while cursor < len(text) and text[cursor] not in _SENTENCE_BREAKS:
        cursor += 1
    return cursor


def _clean_fact(text: str) -> str:
    cleaned = _LINK_PATTERN.sub("", text)
    cleaned = re.sub(r"According to\s*,\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("根据,", "").replace("根据，", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = cleaned.strip(" \t\n-,:;，。")
    return cleaned


def _trim_citation_url(url: str) -> str:
    cut_idx = url.find("#:~:text=")
    return url[:cut_idx] if cut_idx != -1 else url


def _normalize_json_output(raw: str) -> str | None:
    text = _strip_code_fence(raw.strip())
    if not text:
        return None

    for candidate in (text, _recover_json_fragment(text)):
        if not candidate:
            continue
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False)

    try:
        parsed = parse_embedded_json(text)
    except Exception:
        return None
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False)
    return None


def _try_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    for candidate in (text, _remove_trailing_commas(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    body = [line for line in lines if not line.startswith("```")]
    return "\n".join(body).strip()


def _find_first_json_start(text: str) -> int:
    object_start = text.find("{")
    array_start = text.find("[")
    if object_start < 0:
        return array_start
    if array_start < 0:
        return object_start
    return min(object_start, array_start)


def _advance_json_scan_state(
    ch: str,
    *,
    in_string: bool,
    escaped: bool,
    brace_open: int,
    bracket_open: int,
) -> tuple[bool, bool, int, int]:
    if in_string:
        if escaped:
            return True, False, brace_open, bracket_open
        if ch == "\\":
            return True, True, brace_open, bracket_open
        if ch == '"':
            return False, False, brace_open, bracket_open
        return True, False, brace_open, bracket_open

    if ch == '"':
        return True, False, brace_open, bracket_open
    if ch == "{":
        return False, False, brace_open + 1, bracket_open
    if ch == "}":
        return False, False, max(0, brace_open - 1), bracket_open
    if ch == "[":
        return False, False, brace_open, bracket_open + 1
    if ch == "]":
        return False, False, brace_open, max(0, bracket_open - 1)
    return False, False, brace_open, bracket_open


def _repair_json_suffix(
    fragment: str,
    *,
    in_string: bool,
    escaped: bool,
    brace_open: int,
    bracket_open: int,
) -> str:
    repaired = fragment
    if in_string:
        if escaped:
            repaired += "\\"
        repaired += '"'
    if bracket_open > 0:
        repaired += "]" * bracket_open
    if brace_open > 0:
        repaired += "}" * brace_open
    return _remove_trailing_commas(repaired)


def _recover_json_fragment(text: str) -> str | None:
    start = _find_first_json_start(text)
    if start < 0:
        return None

    fragment = text[start:]
    in_string = False
    escaped = False
    brace_open = 0
    bracket_open = 0

    for ch in fragment:
        in_string, escaped, brace_open, bracket_open = _advance_json_scan_state(
            ch,
            in_string=in_string,
            escaped=escaped,
            brace_open=brace_open,
            bracket_open=bracket_open,
        )

    return _repair_json_suffix(
        fragment,
        in_string=in_string,
        escaped=escaped,
        brace_open=brace_open,
        bracket_open=bracket_open,
    )


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fallback_json_for_prompt(user_prompt: str) -> str:
    normalized = user_prompt.lower()
    if "json list" in normalized or "json列表" in normalized or "list(int)" in normalized:
        return "[]"
    return "{}"


def _clean_jina_payload(payload: str) -> str:
    return payload.split("\n\n---\n", 1)[0].strip()


def _strip_control_chars(payload: str) -> str:
    return "".join(ch for ch in payload if ch in "\n\r\t" or ord(ch) >= 32).strip()


async def _fetch_content(url: str) -> tuple[str, str | None]:
    jina_error: str | None = None
    try:
        jina_results = await JinaContentFetcher().fetch([url])
        jina_content = _strip_control_chars(_clean_jina_payload(jina_results.get(url, "")))
        if jina_content:
            return jina_content, None
    except Exception as exc:
        jina_error = str(exc)

    readability_error: str | None = None
    try:
        readability_results = await ReadabilityContentFetcher().fetch([url])
        readability_content = _strip_control_chars(readability_results.get(url, ""))
        if readability_content:
            return readability_content, None
    except Exception as exc:
        readability_error = str(exc)

    if jina_error and readability_error:
        return "", f"{jina_error}; {readability_error}"
    return "", jina_error or readability_error or "empty content"


def _is_nav_only(content: str) -> bool:
    link_chars = sum(len(m.group(0)) for m in _LINK_PATTERN.finditer(content[:2000]))
    stripped = content[:2000].strip()
    return len(stripped) > 0 and link_chars / len(stripped) > 0.6


def _is_inaccessible(content: str) -> bool:
    if len(content) < _MIN_ACCESSIBLE_CONTENT_LEN:
        return True
    sample = content[:2000].lower()
    if any(sig in sample for sig in _INACCESSIBLE_SIGNALS):
        return True
    return _is_nav_only(content)


def scrape_url(url: str) -> dict[str, Any]:
    async def _call() -> dict[str, Any]:
        content, error = await _fetch_content(url)
        if error and not content:
            return {"url": url, "content": "", "error": error}
        if _is_inaccessible(content):
            return {
                "url": url,
                "content": "",
                "error": "inaccessible: content quality check failed",
            }
        return {"url": url, "title": "", "description": "", "content": content}

    return asyncio.run(_call())
