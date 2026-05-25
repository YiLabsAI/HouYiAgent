"""Indexed-mode retrieval orchestration collaborators.

This package contains retrieval dispatch and execution helpers that are owned by
IndexedMode and are specific to indexed retrieval orchestration rather than the
cross-mode retrieval domain.
"""

from houyi.rag.indexed.retrieval.dispatch import (
    RetrievalDispatchContext,
    create_retrieval_task,
)
from houyi.rag.indexed.retrieval.execution import (
    RetrievalExecutionRequest,
    append_failed_results,
    append_timed_out_results,
    build_retrieval_plan,
    cancel_pending_tasks,
    collect_parallel_results,
    execute_parallel_retrieval,
    execute_sequential_retrieval,
)

__all__ = [
    "RetrievalDispatchContext",
    "RetrievalExecutionRequest",
    "append_failed_results",
    "append_timed_out_results",
    "build_retrieval_plan",
    "cancel_pending_tasks",
    "collect_parallel_results",
    "create_retrieval_task",
    "execute_parallel_retrieval",
    "execute_sequential_retrieval",
]
