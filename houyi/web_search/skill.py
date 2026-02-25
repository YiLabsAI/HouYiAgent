from __future__ import annotations

import dataclasses
import inspect

from pydantic import BaseModel, Field

from houyi.core.skill import ExecutionMode, SkillSpec
from houyi.core.skill.policy import InvocationPolicy, NetworkPerm, Permissions, SideEffect
from houyi.web_search.service import WebSearchService


class WebSearchInput(BaseModel):
    query: str = Field(...)
    max_results: int = Field(default=3, ge=1)
    provider: str | None = Field(default=None)
    use_cache: bool | None = Field(default=None)
    mode: str | None = Field(default=None)


class WebSearchOutput(BaseModel):
    provider: str
    results: list[dict]
    metadata: dict


async def _web_search_executor(
    *,
    query: str,
    max_results: int = 3,
    provider: str | None = None,
    mode: str | None = None,
    use_cache: bool | None = None,
) -> dict:
    service = WebSearchService.from_env(provider=provider)
    include_content = (mode or "").strip().lower() == "browse"

    search_kwargs = {
        "max_results": max_results,
        "include_content": include_content,
    }
    if "use_cache" in inspect.signature(service.search).parameters:
        search_kwargs["use_cache"] = True if use_cache is None else bool(use_cache)

    response = await service.search(query, **search_kwargs)  # type: ignore[arg-type]
    return {
        "provider": response.provider,
        "results": [dataclasses.asdict(result) for result in response.results],
        "metadata": {
            "cached": response.metadata.cached,
            "cache_hit": response.metadata.cache_hit,
            "latency_ms": response.metadata.latency_ms,
            "provider": response.metadata.provider,
            "provider_latency_ms": response.metadata.provider_latency_ms,
            "extraction_latency_ms": response.metadata.extraction_latency_ms,
            "extraction_provider": response.metadata.extraction_provider,
            "error_count": response.metadata.error_count,
            "rate_limit_count": response.metadata.rate_limit_count,
            "errors": response.metadata.errors,
        },
        "raw": response.raw,
    }


def build_web_search_skill() -> SkillSpec:
    return SkillSpec(
        name="web_search",
        description="Search the web and return results.",
        input_schema=WebSearchInput,
        output_schema=WebSearchOutput,
        executor=_web_search_executor,
        execution_mode=ExecutionMode.PLUGIN,
        invocation_policy=InvocationPolicy.default_for_side_effect(SideEffect.NETWORK),
        permissions=Permissions(network=NetworkPerm(enabled=True)),
    )
