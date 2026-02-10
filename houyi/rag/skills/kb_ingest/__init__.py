"""Knowledge base ingest skill."""

from houyi.rag.skills.kb_ingest.hooks import (
    get_ingest_state,
    post_ingest_hook,
    pre_ingest_hook,
    reset_ingest_state,
    stop_hook,
)
from houyi.rag.skills.kb_ingest.skill import (
    KBIngestInput,
    KBIngestOutput,
    execute_kb_ingest,
    kb_ingest_skill,
)

__all__ = [
    "KBIngestInput",
    "KBIngestOutput",
    "execute_kb_ingest",
    "kb_ingest_skill",
    "pre_ingest_hook",
    "post_ingest_hook",
    "stop_hook",
    "get_ingest_state",
    "reset_ingest_state",
]
