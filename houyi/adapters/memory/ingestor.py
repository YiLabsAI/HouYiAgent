"""Memory ingestor: absorb one conversation turn into the store.

MemoryIngestor is the single async entry point that turns a
user utterance into persisted memory. It is intentionally a thin
coordinator with no state of its own, composing four collaborators:

1. RetractionOrchestrator - if the speaker just took back a
 prior claim, close the active rows the caller flagged as recent and
 skip extraction entirely. This must run before extraction so a
 pure retraction utterance ("I was wrong about the address") cannot
 accidentally produce contradictory new facts.
2. AtomicFactExtractor - turn the utterance into 6-tuple
 AtomicFact instances. This stage is async (LLM HTTP call).
3. Sourceless routing - if the caller could not supply a source anchor,
 every extracted item lands in the candidate inbox tagged
 reason=sourceless instead of the main entity_state view.
4. MemoryWriterTools.ingest_fact - for facts that have a real
 anchor, route them through the ADD/UPDATE/vague decision logic.

Concurrency model: stages 3 and 4 issue blocking SQLite calls. To
keep the asyncio event loop responsive under concurrent ingest load,
those calls are dispatched through asyncio.to_thread rather
than executed inline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from houyi.adapters.memory.backends.base import CandidateInbox
from houyi.adapters.memory.entity_resolver import (
    EntityResolver,
    TurnContext,
    get_default_resolver,
)
from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.adapters.memory.extractor import AtomicFactExtractor, ExtractionResult
from houyi.adapters.memory.resolver import IngestDecision, MemoryWriterTools
from houyi.adapters.memory.retraction import (
    RetractionOrchestrator,
    RetractionOutcome,
    RetractionTarget,
)
from houyi.adapters.memory.types import Certainty
from houyi.application.evolution.events import EvolutionEventType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestTurnResult:
    """Aggregate outcome of one ingest_turn call.

    The fields mirror the order in which each stage runs so callers can
    read the result top-to-bottom and reconstruct what happened:

    - retraction is populated only if the retraction stage fired
    (None otherwise) so a downstream UI can render a "you took
    back X" badge precisely when one is needed.
    - decisions lists the per-fact IngestDecision returned by
    the writer tools. decision == "admitted" means the fact
    reached entity_state; "deferred_vague" means the inbox.
    - sourceless_candidates are inbox ids generated when the call
    had no anchor. Empty when source_anchor was supplied.
    - invalid_dropped matches ExtractionResult.invalid_dropped
    and is surfaced for telemetry.
    """

    retraction: RetractionOutcome | None = None
    decisions: list[IngestDecision] = field(default_factory=list)
    sourceless_candidates: list[str] = field(default_factory=list)
    invalid_dropped: int = 0


class MemoryIngestor:
    """Async coordinator that absorbs one conversation turn into memory.

    Runs retraction -> extraction -> sourceless routing -> writer routing
    for a single utterance and aggregates the per-stage outcomes into
    one IngestTurnResult.
    """

    def __init__(
        self,
        extractor: AtomicFactExtractor,
        retraction: RetractionOrchestrator,
        writer_tools: MemoryWriterTools,
        inbox: CandidateInbox,
        *,
        emitter: MemoryEventEmitter | None = None,
    ) -> None:
        self._extractor = extractor
        self._retraction = retraction
        self._writer = writer_tools
        self._inbox = inbox
        # Optional hot-path event emitter. The ingestor publishes
        # EXTRACTOR_LOW_CERTAINTY when an extraction yields zero certain
        # facts (entirely vague, sourceless, or schema-invalid output) so
        # the evolution control plane can target prompt/extractor tuning
        # at the actual failure modes seen in production turns.
        self._emitter = emitter or MemoryEventEmitter()

    @property
    def namespace(self) -> str:
        return self._writer.namespace

    async def ingest_turn(
        self,
        text: str,
        *,
        source_anchor: str | None,
        recent_targets: list[RetractionTarget] | tuple[RetractionTarget, ...] = (),
        observation_date: str | None = None,
        turn_context: TurnContext | None = None,
        entity_resolver: EntityResolver | None = None,
    ) -> IngestTurnResult:
        """Process one user utterance through the ingestor.

        recent_targets is the caller's bookkeeping of which
        (entity, attribute) pairs the speaker most recently
        committed; it is the only state the ingestor cannot derive on
        its own and is used exclusively by the retraction stage.
        source_anchor is the provenance handle for the utterance
        (chunk id, message id, etc.). Pass None or an empty string
        to force every extracted item into the sourceless inbox.
        observation_date is the date when the conversation occurred,
        used to resolve relative time references (e.g., "yesterday").
        If not provided, the current system date is used.
        turn_context provides speaker and session metadata for entity
        resolution. If omitted, a default context is created.
        entity_resolver determines how the speaker maps to the entity
        subject in extracted facts. If omitted, uses the default "user"
        resolver for single-user scenarios.
        """
        # Stage 1 - retraction first so a pure "I was wrong" turn does
        # not enter extraction at all (which would happily produce
        # whatever facts the LLM hallucinates from negative phrasing).
        retraction_outcome = self._retraction.process(text, recent_targets)
        if retraction_outcome.signal is not None:
            return IngestTurnResult(retraction=retraction_outcome)

        # Stage 2 - extract (async LLM call).
        # Resolve entity and format input with temporal context.
        resolver = entity_resolver or get_default_resolver()
        ctx = turn_context or TurnContext(text=text)
        entity_id = resolver.resolve(ctx)
        extract_input = self._format_extract_input(text, observation_date, entity_id)
        extraction = await self._extractor.extract(extract_input, source_anchor)

        # Side-channel: tell the evolution control plane when extraction
        # did not yield any certain fact for a turn. This is a strong
        # signal for prompt or LLM tuning targets.
        self._emit_low_certainty(text, extraction)

        # Stage 3 - sourceless degradation. Synchronous SQLite INSERTs
        # are dispatched off the event loop so concurrent ingestors do
        # not serialize on the writer's I/O.
        sourceless_ids = await asyncio.to_thread(self._park_sourceless, extraction)

        # Stage 4 - route validated facts through the writer tools.
        # Each ingest_fact issues SELECT + INSERT/UPDATE under SQLite;
        # again wrapped in to_thread to keep the loop free.
        decisions: list[IngestDecision] = []
        for fact in extraction.facts:
            try:
                decisions.append(await asyncio.to_thread(self._writer.ingest_fact, fact))
            except Exception:
                # Writer-level errors (conflict / missing-active) are
                # already mapped to recoverable exception classes; any
                # leak past that is genuinely unexpected and should be
                # logged but must not abort the whole turn.
                logger.exception("writer_tools.ingest_fact crashed on fact=%r", fact)

        return IngestTurnResult(
            retraction=None,
            decisions=decisions,
            sourceless_candidates=sourceless_ids,
            invalid_dropped=extraction.invalid_dropped,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _park_sourceless(self, extraction: ExtractionResult) -> list[str]:
        """Persist sourceless extractions to the inbox; return their ids."""
        if not extraction.raw_sourceless:
            return []
        ids: list[str] = []
        for raw in extraction.raw_sourceless:
            ids.append(self._inbox.add_sourceless(self.namespace, raw))
        return ids

    @staticmethod
    def _normalize_date(raw: str) -> str:
        """Convert a free-form date string to YYYY-MM-DD where possible.

        Handles formats like:
          "4:15 pm on 20 April, 2023"  -> "2023-04-20"
          "March 5, 2023"              -> "2023-03-05"
          "2023-04-20"                 -> "2023-04-20"  (pass-through)
        Returns the original string unchanged if no known pattern matches.
        """
        if not raw:
            return raw
        # Already ISO format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", raw.strip()):
            return raw.strip()
        # Try dateutil if available (best effort)
        try:
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                _du: Any = None
            else:
                from dateutil import parser as _du

            return _du.parse(raw, fuzzy=True).strftime("%Y-%m-%d")
        except Exception:
            pass
        # Fallback regex: "D Month, YYYY" or "D Month YYYY"
        m = re.search(
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
            r"September|October|November|December),?\s+(\d{4})",
            raw,
            re.IGNORECASE,
        )
        if m:
            months = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }
            day = int(m.group(1))
            month = months[m.group(2).lower()]
            year = int(m.group(3))
            return f"{year:04d}-{month:02d}-{day:02d}"
        return raw

    def _format_extract_input(self, text: str, observation_date: str | None, entity_id: str) -> str:
        """Format text with temporal context and entity identity for extraction.

        Returns JSON format expected by the few-shot prompt:
        {"observation_date": "YYYY-MM-DD", "system_date": "YYYY-MM-DD", "text": "...", "speaker_name": "..."}

        The entity_id is passed as speaker_name in the prompt to guide the LLM
        to use this identifier as the subject in extracted facts.
        """
        obs_date = (
            self._normalize_date(observation_date)
            if observation_date
            else datetime.now().strftime("%Y-%m-%d")
        )
        sys_date = datetime.now().strftime("%Y-%m-%d")
        data: dict[str, Any] = {
            "observation_date": obs_date,
            "system_date": sys_date,
            "text": text,
            "speaker_name": entity_id,
        }
        return json.dumps(data, ensure_ascii=False)

    def _emit_low_certainty(self, text: str, extraction: ExtractionResult) -> None:
        """Publish EXTRACTOR_LOW_CERTAINTY when the turn yielded no certain fact.

        Triggers when either: the extractor produced zero schema-valid
        facts (entirely vague / sourceless / dropped), or every produced
        fact has Certainty != CERTAIN. We also record auxiliary counters
        so the control plane can stratify by failure mode.
        """
        certain_count = sum(1 for fact in extraction.facts if fact.certainty == Certainty.CERTAIN)
        if certain_count > 0:
            return
        self._emitter.emit(
            EvolutionEventType.EXTRACTOR_LOW_CERTAINTY,
            target="memory_ingestor",
            payload={"text_preview": text[:200]},
            metrics={
                "facts_total": float(len(extraction.facts)),
                "vague_or_probable": float(
                    sum(1 for fact in extraction.facts if fact.certainty != Certainty.CERTAIN)
                ),
                "sourceless": float(len(extraction.raw_sourceless)),
                "invalid_dropped": float(extraction.invalid_dropped),
            },
            namespace=self.namespace,
        )
