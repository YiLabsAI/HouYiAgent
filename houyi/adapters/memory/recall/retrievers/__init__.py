"""Per-query-type retrievers for the memory recall pipeline.

Each retriever implements Retriever and is responsible for a
specific retrieval strategy. The orchestrator chooses retrievers based
on the router's classification; retrievers themselves are stateless
and never inspect QueryType directly.
"""

from __future__ import annotations

from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.iterative import IterativeMultiHopRetriever
from houyi.adapters.memory.recall.retrievers.raw_turn import RawTurnLogRetriever
from houyi.adapters.memory.recall.retrievers.timeline import TimelineRetriever
from houyi.adapters.memory.recall.retrievers.vector import VectorRecallRetriever

__all__ = [
    "EntityStateRetriever",
    "IterativeMultiHopRetriever",
    "RawTurnLogRetriever",
    "Retriever",
    "TimelineRetriever",
    "VectorRecallRetriever",
]
