from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalExecutionRequest:
    """Stable input boundary for indexed retrieval execution paths."""

    strategies: list[Any]
    timeout: float
    create_retrieval_task: Any
    result_factory: Any


async def execute_parallel_retrieval(
    request: RetrievalExecutionRequest,
) -> list[Any]:
    """Execute retrieval strategies concurrently and normalize their outcomes.

    Each strategy may yield a task or be skipped by returning `None`. Task failures are
    captured into `result_factory` records instead of surfacing to the facade. Pending
    tasks are cancelled after the timeout window and converted into timed-out results.
    """

    tasks, task_info = build_retrieval_plan(
        strategies=request.strategies,
        create_retrieval_task=request.create_retrieval_task,
    )
    if not tasks:
        return []

    results: list[Any] = []
    start_time = time.time()
    try:
        async_tasks = [asyncio.create_task(task) for task in tasks]
        done, pending = await asyncio.wait(
            async_tasks,
            timeout=request.timeout,
            return_when=asyncio.ALL_COMPLETED,
        )
        elapsed = (time.time() - start_time) * 1000
        completed_indices = collect_parallel_results(
            done=done,
            task_info=task_info,
            results=results,
            elapsed=elapsed,
            result_factory=request.result_factory,
        )
        await cancel_pending_tasks(pending)
        append_timed_out_results(
            results=results,
            task_info=task_info,
            completed_indices=completed_indices,
            timeout=request.timeout,
            result_factory=request.result_factory,
        )
    except Exception as exc:
        logger.error("Parallel retrieval failed: %s", exc)
        append_failed_results(
            results=results,
            task_info=task_info,
            error=str(exc),
            result_factory=request.result_factory,
        )
    return results


async def execute_sequential_retrieval(
    request: RetrievalExecutionRequest,
) -> list[Any]:
    """Execute retrieval strategies one by one with per-strategy timeout handling.

    Sequential execution preserves strategy order and records timeout / failure state per
    strategy so callers can still build retrieval metadata from partial success.
    """

    results: list[Any] = []

    for strategy in request.strategies:
        task = request.create_retrieval_task(strategy)
        if task is None:
            continue
        name, coro = task
        start_time = time.time()
        try:
            task_results = await asyncio.wait_for(coro, timeout=request.timeout)
            elapsed = (time.time() - start_time) * 1000
            results.append(
                request.result_factory(
                    strategy=strategy,
                    strategy_name=name,
                    results=task_results,
                    success=True,
                    duration_ms=elapsed,
                )
            )
        except TimeoutError:
            elapsed = (time.time() - start_time) * 1000
            results.append(
                request.result_factory(
                    strategy=strategy,
                    strategy_name=name,
                    success=False,
                    timed_out=True,
                    error=f"Timeout after {request.timeout}s",
                    duration_ms=elapsed,
                )
            )
            logger.warning("Strategy %s timed out after %s seconds", name, request.timeout)
        except Exception as exc:
            elapsed = (time.time() - start_time) * 1000
            results.append(
                request.result_factory(
                    strategy=strategy,
                    strategy_name=name,
                    success=False,
                    error=str(exc),
                    duration_ms=elapsed,
                )
            )
            logger.warning("Strategy %s failed: %s", name, exc)

    return results


def build_retrieval_plan(
    *,
    strategies: list[Any],
    create_retrieval_task,
) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Build the executable task list and aligned strategy metadata.

    Strategies that do not produce a task are filtered out here so downstream execution
    paths only operate on runnable coroutines.
    """

    tasks: list[Any] = []
    task_info: list[tuple[Any, str]] = []
    for strategy in strategies:
        task = create_retrieval_task(strategy)
        if task is None:
            continue
        name, coro = task
        tasks.append(coro)
        task_info.append((strategy, name))
    return tasks, task_info


def collect_parallel_results(
    *,
    done: set[asyncio.Task[Any]],
    task_info: list[tuple[Any, str]],
    results: list[Any],
    elapsed: float,
    result_factory,
) -> set[int]:
    """Collect completed parallel tasks into normalized retrieval result records."""

    completed_indices: set[int] = set()
    for index, task in enumerate(done):
        strategy, name = task_info[index]
        try:
            task_results = task.result()
            results.append(
                result_factory(
                    strategy=strategy,
                    strategy_name=name,
                    results=task_results,
                    success=True,
                    duration_ms=elapsed,
                )
            )
        except Exception as exc:
            results.append(
                result_factory(
                    strategy=strategy,
                    strategy_name=name,
                    success=False,
                    error=str(exc),
                    duration_ms=elapsed,
                )
            )
        completed_indices.add(index)
    return completed_indices


async def cancel_pending_tasks(pending: set[asyncio.Task[Any]]) -> None:
    """Cancel pending tasks and suppress expected cancellation noise."""

    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def append_timed_out_results(
    *,
    results: list[Any],
    task_info: list[tuple[Any, str]],
    completed_indices: set[int],
    timeout: float,
    result_factory,
) -> None:
    """Append timeout records for strategies that did not finish within the window."""

    for index, (strategy, name) in enumerate(task_info):
        if index in completed_indices:
            continue
        results.append(
            result_factory(
                strategy=strategy,
                strategy_name=name,
                success=False,
                timed_out=True,
                error=f"Timeout after {timeout}s",
                duration_ms=timeout * 1000,
            )
        )
        logger.warning("Strategy %s timed out after %s seconds", name, timeout)


def append_failed_results(
    *,
    results: list[Any],
    task_info: list[tuple[Any, str]],
    error: str,
    result_factory,
) -> None:
    """Append failure records when the parallel execution path aborts as a whole."""

    for strategy, name in task_info:
        results.append(
            result_factory(
                strategy=strategy,
                strategy_name=name,
                success=False,
                error=error,
            )
        )
