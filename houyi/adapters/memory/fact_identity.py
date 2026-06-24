"""Identity hashing for AtomicFact-derived records.

The memories table and the entity_state view are linked only by a shared
SHA-256 digest derived from the fact's (subject, predicate, object, anchor)
quadruple. Both tables compute the same digest independently at write time:
fact_promoter builds the memories.record_id / memories.key, and the
extractor worker re-derives the same record_id for edge resolution. The
dreamer's entity-state consolidator must re-derive it again to propagate
valid_to to the exact backing memories row when superseding an
entity_state row.

Centralising the formula here makes it a single source of truth: a future
change to the identity scheme lands in one place and the promoter, the
extractor, and the consolidator all follow. It also removes the pre-existing
duplication between fact_promoter and the extractor worker.
"""

from __future__ import annotations

import hashlib


def fact_digest(subject: str, predicate: str, obj: object, anchor: str) -> str:
    """Stable 24-hex-char identity digest for a fact quadruple.

    The quadruple is (subject, predicate, object, source_anchor). anchor
    is the empty string when the fact carries no source anchor, matching the
    historical behaviour at the two prior call sites.
    """
    plain = f"{subject}|{predicate}|{obj}|{anchor}"
    return hashlib.sha256(plain.encode()).hexdigest()[:24]


def fact_record_id(subject: str, predicate: str, obj: object, anchor: str) -> str:
    """The memories.record_id for a fact (primary key of the backing row)."""
    return f"fact:{fact_digest(subject, predicate, obj, anchor)}"


def fact_key(subject: str, predicate: str, obj: object, anchor: str) -> str:
    """The memories.key for a fact (the UNIQUE(scope, key) business key)."""
    return f"{subject}.{predicate}.{fact_digest(subject, predicate, obj, anchor)}"


__all__ = ["fact_digest", "fact_key", "fact_record_id"]
