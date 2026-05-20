"""SQLite candidate inbox implementation."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from typing import Any

from houyi.adapters.memory.backends.base import CandidateInbox
from houyi.adapters.memory.types import AtomicFact, Certainty

logger = logging.getLogger(__name__)


class SQLiteCandidateInbox(CandidateInbox):
    """SQLite-backed implementation of CandidateInbox.

    Reuses the host backend's connection so writes to the inbox stay in
    the same transactional boundary as writes to entity_state.
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    def add(self, namespace: str, fact: AtomicFact) -> str:
        if fact.certainty is not Certainty.VAGUE:
            raise ValueError("CandidateInbox.add only accepts vague facts")

        candidate_id = uuid.uuid4().hex[:12]
        conn = self._backend._conn()
        conn.execute(
            """
 INSERT INTO vague_candidates
 (candidate_id, namespace, entity, attribute, value,
 qualifiers, source_unit_id, fact_payload, reason, created_at)
 VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'vague', ?)
 """,
            (
                candidate_id,
                namespace,
                fact.subject,
                fact.predicate,
                fact.object,
                json.dumps(fact.qualifiers, ensure_ascii=False) if fact.qualifiers else None,
                fact.source_anchor,
                fact.model_dump_json(),
                time.time(),
            ),
        )
        conn.commit()
        return candidate_id

    def add_sourceless(
        self,
        namespace: str,
        raw_payload: dict[str, Any],
    ) -> str:
        candidate_id = uuid.uuid4().hex[:12]
        entity = str(raw_payload.get("subject", "") or "")
        attribute = str(raw_payload.get("predicate", "") or "")
        value = str(raw_payload.get("object", "") or "")
        conn = self._backend._conn()
        conn.execute(
            """
 INSERT INTO vague_candidates
 (candidate_id, namespace, entity, attribute, value,
 qualifiers, source_unit_id, fact_payload, reason, created_at)
 VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, 'sourceless', ?)
 """,
            (
                candidate_id,
                namespace,
                entity,
                attribute,
                value,
                json.dumps(raw_payload, ensure_ascii=False),
                time.time(),
            ),
        )
        conn.commit()
        return candidate_id

    def list_for(
        self,
        namespace: str,
        entity: str | None = None,
        attribute: str | None = None,
        reason: str | None = None,
    ) -> list[AtomicFact]:
        conn = self._backend._conn()
        sql = "SELECT fact_payload FROM vague_candidates WHERE namespace=?"
        params: list[object] = [namespace]
        sql += " AND reason=?"
        params.append(reason or "vague")
        if entity is not None:
            sql += " AND entity=?"
            params.append(entity)
        if attribute is not None:
            sql += " AND attribute=?"
            params.append(attribute)
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, tuple(params)).fetchall()

        out: list[AtomicFact] = []
        for row in rows:
            try:
                out.append(AtomicFact.model_validate_json(row["fact_payload"]))
            except (ValueError, sqlite3.Error):
                logger.warning("skipping malformed candidate payload")
        return out

    def list_sourceless(self, namespace: str) -> list[dict[str, Any]]:
        conn = self._backend._conn()
        rows = conn.execute(
            """
 SELECT fact_payload FROM vague_candidates
 WHERE namespace=? AND reason='sourceless'
 ORDER BY created_at DESC
 """,
            (namespace,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["fact_payload"])
                if isinstance(payload, dict):
                    out.append(payload)
            except json.JSONDecodeError:
                logger.warning("skipping malformed sourceless payload")
                continue
        return out
