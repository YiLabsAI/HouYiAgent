from __future__ import annotations

from typing import Any


def fusion_result_key(result: Any) -> str:
    return str(result.chunk_id or hash(result.content))


def accumulate_fusion_score(
    *,
    result: Any,
    delta: float,
    scores: dict[str, float],
    result_map: dict[str, Any],
) -> None:
    key = fusion_result_key(result)
    if key not in scores:
        scores[key] = 0.0
        result_map[key] = result
    scores[key] += delta


def rrf_fusion(
    *,
    strategy_results: list[tuple[str, list[Any]]],
    top_k: int,
    rrf_k: int,
) -> list[Any]:
    if not strategy_results:
        return []
    if len(strategy_results) == 1:
        return strategy_results[0][1][:top_k]

    scores: dict[str, float] = {}
    result_map: dict[str, Any] = {}

    for _, results in strategy_results:
        for rank, result in enumerate(results, 1):
            accumulate_fusion_score(
                result=result,
                delta=1.0 / (rrf_k + rank),
                scores=scores,
                result_map=result_map,
            )

    sorted_keys = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
    fused: list[Any] = []
    for key in sorted_keys[:top_k]:
        result = result_map[key]
        result.score = scores[key]
        fused.append(result)
    return fused
