from __future__ import annotations

import logging
from typing import Any

from houyi.rag.types import RetrievalStrategy, SearchResult

logger = logging.getLogger(__name__)


def process_retrieval_results(
    *,
    task_results: list[Any],
    strategies_used: list[RetrievalStrategy],
    all_results: list[tuple[str, list[Any]]],
    fallback_on_timeout: bool,
) -> dict[str, Any]:
    """Aggregate per-strategy execution outcomes into retrieval metadata.

    Successful strategies are appended into strategies_used and all_results for
    later fusion. When timeout fallback is disabled, any timeout causes previously
    collected retrieval results to be cleared so callers do not answer from a partial
    retrieval set.
    """

    metadata: dict[str, Any] = {
        "total_strategies": len(task_results),
        "successful_count": 0,
        "failed_count": 0,
        "timed_out_count": 0,
        "strategy_details": [],
    }

    for result in task_results:
        detail = {
            "strategy": result.strategy.value,
            "name": result.strategy_name,
            "success": result.success,
            "timed_out": result.timed_out,
            "result_count": len(result.results),
            "duration_ms": result.duration_ms,
        }
        if result.error:
            detail["error"] = result.error
        metadata["strategy_details"].append(detail)

        if result.success:
            metadata["successful_count"] += 1
            strategies_used.append(result.strategy)
            all_results.append((result.strategy_name, result.results))
            continue

        metadata["failed_count"] += 1
        if result.timed_out:
            metadata["timed_out_count"] += 1

    if not fallback_on_timeout and metadata["timed_out_count"] > 0:
        logger.warning(
            "Fallback disabled and %d strategies timed out, clearing results",
            metadata["timed_out_count"],
        )
        strategies_used.clear()
        all_results.clear()

    return metadata


async def apply_crag(
    *,
    validator: Any,
    query: str,
    fused_results: list[SearchResult],
    enable_crag: bool,
    metadata: dict[str, Any],
) -> tuple[list[SearchResult], str | None]:
    """Apply CRAG validation when available and fall back on validation failure.

    The original fused results are preserved whenever CRAG is disabled, unavailable, or
    fails at runtime. Successful validation enriches metadata even when no result list is
    replaced.
    """

    if not enable_crag or not validator or not fused_results:
        return fused_results, None
    try:
        crag_result = await validator.validate(query, fused_results)
        crag_quality = crag_result.quality.value
        metadata["crag_quality"] = crag_quality
        metadata["crag_confidence"] = crag_result.confidence
        metadata["crag_reasoning"] = crag_result.reasoning
        if crag_result.relevant_results:
            logger.debug(
                "CRAG filtered results: %d -> %d (quality: %s)",
                len(fused_results),
                len(crag_result.relevant_results),
                crag_quality,
            )
            return crag_result.relevant_results, crag_quality
        return fused_results, crag_quality
    except Exception as exc:
        logger.warning("CRAG validation failed: %s", exc)
        return fused_results, None


def adjust_confidence(
    *,
    confidence: float,
    crag_quality: str | None,
    retrieval_metadata: dict[str, Any],
) -> float:
    """Cap confidence based on CRAG quality and retrieval degradation signals."""

    if crag_quality == "incorrect":
        confidence = min(confidence, 0.3)
    elif crag_quality == "ambiguous":
        confidence = min(confidence, 0.6)
    if retrieval_metadata.get("timed_out_count", 0) > 0:
        confidence = min(confidence, 0.7)
    return confidence


def collect_sources(fused_results: list[SearchResult]) -> list[str]:
    """Collect up to ten non-empty sources from fused retrieval results."""

    return [result.source for result in fused_results if result.source][:10]


async def generate_answer(
    *,
    answer_generator: Any,
    query: str,
    results: list[SearchResult],
) -> tuple[str, float]:
    """Generate the final answer with LLM fallback to a simple extractive summary."""

    if not results:
        return "No relevant information found.", 0.0

    if answer_generator:
        try:
            answer, confidence = await answer_generator.generate(
                query=query,
                results=results,
                include_sources=True,
            )
            return answer, confidence
        except Exception as exc:
            logger.warning("LLM answer generation failed: %s", exc)

    return build_answer_simple(results), min(len(results) * 0.1, 0.7)


def build_answer_simple(results: list[SearchResult]) -> str:
    """Build a minimal extractive answer from the highest-ranked result contents."""

    contents: list[str] = []
    for result in results[:5]:
        if result.content:
            contents.append(result.content.strip())
    if not contents:
        return "No relevant information found."
    return "\n\n---\n\n".join(contents)
