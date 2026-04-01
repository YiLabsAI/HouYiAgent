"""MemoryStore unit tests.

Validates:
- Core CRUD API (put, get, list_by_scope, delete, clear)
- Typed memory fields (memory_type, tags, confidence, provenance, embedding)
- Convenience methods (put_record, list_by_type, all_records)
- Enriched as_context_text rendering
- Persistence round-trip
"""

from __future__ import annotations

import time

from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)


class TestCoreApi:
    """Core CRUD operations."""

    def test_put_positional_args(self):
        store = MemoryStore()
        r = store.put("k", "v", MemoryScope.SESSION)
        assert r.key == "k"
        assert r.memory_type == MemoryType.FACT

    def test_put_keyword_args(self):
        store = MemoryStore()
        r = store.put("k", "v", scope=MemoryScope.USER, metadata={"a": 1}, ttl=60.0)
        assert r.scope == MemoryScope.USER
        assert r.metadata == {"a": 1}
        assert r.ttl == 60.0

    def test_get_returns_defaults(self):
        store = MemoryStore()
        store.put("k", "v")
        r = store.get("k")
        assert r is not None
        assert r.memory_type == MemoryType.FACT
        assert r.confidence == 1.0
        assert r.decay == 1.0
        assert r.tags == []
        assert r.provenance is None
        assert r.embedding is None


class TestTypedFields:
    """Typed memory fields (memory_type, tags, provenance, embedding)."""

    def test_put_with_type_and_tags(self):
        store = MemoryStore()
        r = store.put(
            "lang",
            "Python",
            memory_type=MemoryType.PREFERENCE,
            tags=["coding", "language"],
        )
        assert r.memory_type == MemoryType.PREFERENCE
        assert r.tags == ["coding", "language"]

    def test_put_with_provenance(self):
        store = MemoryStore()
        prov = MemoryProvenance(
            source_type="conversation",
            source_ids=["msg_001"],
            extracted_by="rule_extractor",
        )
        r = store.put("fact1", "Earth orbits Sun", provenance=prov)
        assert r.provenance is not None
        assert r.provenance.source_type == "conversation"

    def test_put_with_embedding(self):
        store = MemoryStore()
        emb = [0.1, 0.2, 0.3]
        r = store.put("k", "v", embedding=emb)
        assert r.embedding == emb

    def test_put_with_confidence(self):
        store = MemoryStore()
        r = store.put("k", "v", confidence=0.85)
        assert r.confidence == 0.85

    def test_update_preserves_typed(self):
        store = MemoryStore()
        store.put("k", "v1", memory_type=MemoryType.PROFILE, tags=["bio"])
        r = store.put("k", "v2", memory_type=MemoryType.PROFILE, tags=["bio", "name"])
        assert r.content == "v2"
        assert r.tags == ["bio", "name"]
        assert r.memory_type == MemoryType.PROFILE


class TestPutRecord:
    """Test put_record for pipeline-assembled records."""

    def test_store_prebuilt_record(self):
        store = MemoryStore()
        rec = MemoryRecord(
            key="pref_lang",
            content="prefers Rust",
            scope=MemoryScope.USER,
            memory_type=MemoryType.PREFERENCE,
            tags=["language"],
            confidence=0.9,
        )
        result = store.put_record(rec)
        assert result.key == "pref_lang"
        retrieved = store.get("pref_lang", MemoryScope.USER)
        assert retrieved is not None
        assert retrieved.memory_type == MemoryType.PREFERENCE


class TestListByType:
    """Test list_by_type filtering."""

    def test_filters_by_type(self):
        store = MemoryStore()
        store.put("name", "Alice", memory_type=MemoryType.PROFILE)
        store.put("lang", "Python", memory_type=MemoryType.PREFERENCE)
        store.put("role", "Engineer", memory_type=MemoryType.PROFILE)

        profiles = store.list_by_type(MemoryType.PROFILE)
        assert len(profiles) == 2
        assert all(r.memory_type == MemoryType.PROFILE for r in profiles)

    def test_filters_by_type_and_scope(self):
        store = MemoryStore()
        store.put("a", "v1", scope=MemoryScope.USER, memory_type=MemoryType.FACT)
        store.put("b", "v2", scope=MemoryScope.SESSION, memory_type=MemoryType.FACT)

        user_facts = store.list_by_type(MemoryType.FACT, scope=MemoryScope.USER)
        assert len(user_facts) == 1
        assert user_facts[0].key == "a"


class TestAllRecords:
    """Test all_records across scopes."""

    def test_returns_all_non_expired(self):
        store = MemoryStore()
        store.put("a", "1", scope=MemoryScope.SESSION)
        store.put("b", "2", scope=MemoryScope.USER)
        store.put("c", "3", scope=MemoryScope.WORKSPACE)
        assert len(store.all_records()) == 3

    def test_excludes_expired(self):
        store = MemoryStore()
        store.put("live", "val")
        store.put("dead", "val", ttl=0.001)
        time.sleep(0.01)
        assert len(store.all_records()) == 1


class TestContextTextEnriched:
    """Test enriched as_context_text rendering."""

    def test_includes_type_prefix(self):
        store = MemoryStore()
        store.put("name", "Alice", memory_type=MemoryType.PROFILE)
        text = store.as_context_text(MemoryScope.SESSION)
        assert "[profile]" in text
        assert "name: Alice" in text

    def test_fact_type_has_no_prefix(self):
        store = MemoryStore()
        store.put("sky", "blue", memory_type=MemoryType.FACT)
        text = store.as_context_text(MemoryScope.SESSION)
        assert text == "- sky: blue"

    def test_includes_tags(self):
        store = MemoryStore()
        store.put("lang", "Python", tags=["coding"])
        text = store.as_context_text(MemoryScope.SESSION)
        assert "[coding]" in text

    def test_default_format(self):
        store = MemoryStore()
        store.put("k", "v")
        text = store.as_context_text(MemoryScope.SESSION)
        assert text == "- k: v"


class TestPersistence:
    """Typed fields survive persist/reload cycle."""

    def test_typed_fields_roundtrip(self, tmp_path):
        store1 = MemoryStore(data_dir=tmp_path / "mem")
        prov = MemoryProvenance(source_type="tool_result", source_ids=["t1"])
        store1.put(
            "fact1",
            "Water boils at 100C",
            scope=MemoryScope.WORKSPACE,
            memory_type=MemoryType.FACT,
            tags=["science", "chemistry"],
            confidence=0.95,
            decay=0.98,
            provenance=prov,
            embedding=[0.1, 0.2, 0.3],
        )

        store2 = MemoryStore(data_dir=tmp_path / "mem")
        r = store2.get("fact1", MemoryScope.WORKSPACE)
        assert r is not None
        assert r.memory_type == MemoryType.FACT
        assert r.tags == ["science", "chemistry"]
        assert r.confidence == 0.95
        assert r.decay == 0.98
        assert r.provenance is not None
        assert r.provenance.source_type == "tool_result"
        assert r.embedding is not None
        assert len(r.embedding) == 3
        assert abs(r.embedding[0] - 0.1) < 1e-4

    def test_defaults_on_minimal_record(self):
        """A record with only required fields gets sensible defaults."""
        store = MemoryStore()
        store.put("minimal", "val")
        r = store.get("minimal", MemoryScope.SESSION)
        assert r is not None
        assert r.memory_type == MemoryType.FACT
        assert r.confidence == 1.0
        assert r.tags == []
        assert r.provenance is None
