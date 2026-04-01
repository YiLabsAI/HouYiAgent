"""SQLiteMemoryBackend unit tests.

Covers CRUD, FTS5, embedding cache, expiry, and edge cases.
"""

from __future__ import annotations

import time

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.types import (
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)


@pytest.fixture()
def backend(tmp_path) -> SQLiteMemoryBackend:
    return SQLiteMemoryBackend(db_path=tmp_path / "test.db")


@pytest.fixture()
def mem_backend(tmp_path) -> SQLiteMemoryBackend:
    """In-memory backend for fast tests."""
    return SQLiteMemoryBackend(db_path=":memory:")


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


class TestCrud:
    def test_put_and_get(self, mem_backend):
        r = MemoryRecord(key="name", content="Alice", scope=MemoryScope.USER)
        mem_backend.put(r)
        got = mem_backend.get("name", MemoryScope.USER)
        assert got is not None
        assert got.content == "Alice"
        assert got.record_id == r.record_id

    def test_put_upsert(self, mem_backend):
        r1 = MemoryRecord(key="k", content="v1", scope=MemoryScope.SESSION)
        mem_backend.put(r1)
        r2 = MemoryRecord(key="k", content="v2", scope=MemoryScope.SESSION)
        mem_backend.put(r2)
        got = mem_backend.get("k", MemoryScope.SESSION)
        assert got is not None
        assert got.content == "v2"

    def test_get_missing(self, mem_backend):
        assert mem_backend.get("nope", MemoryScope.SESSION) is None

    def test_delete(self, mem_backend):
        r = MemoryRecord(key="rm", content="bye", scope=MemoryScope.SESSION)
        mem_backend.put(r)
        assert mem_backend.delete("rm", MemoryScope.SESSION) is True
        assert mem_backend.get("rm", MemoryScope.SESSION) is None

    def test_delete_missing(self, mem_backend):
        assert mem_backend.delete("nope", MemoryScope.SESSION) is False

    def test_list_by_scope(self, mem_backend):
        mem_backend.put(MemoryRecord(key="a", content="1", scope=MemoryScope.USER))
        mem_backend.put(MemoryRecord(key="b", content="2", scope=MemoryScope.SESSION))
        mem_backend.put(MemoryRecord(key="c", content="3", scope=MemoryScope.USER))
        users = mem_backend.list_by_scope(MemoryScope.USER)
        assert len(users) == 2
        assert all(r.scope == MemoryScope.USER for r in users)

    def test_list_by_type(self, mem_backend):
        mem_backend.put(MemoryRecord(key="n", content="A", memory_type=MemoryType.PROFILE))
        mem_backend.put(MemoryRecord(key="l", content="Py", memory_type=MemoryType.PREFERENCE))
        profiles = mem_backend.list_by_type(MemoryType.PROFILE)
        assert len(profiles) == 1
        assert profiles[0].key == "n"

    def test_all_records(self, mem_backend):
        for i in range(5):
            mem_backend.put(MemoryRecord(key=f"k{i}", content=f"v{i}"))
        assert len(mem_backend.all_records()) == 5

    def test_clear_all(self, mem_backend):
        for i in range(3):
            mem_backend.put(MemoryRecord(key=f"k{i}", content=f"v{i}"))
        count = mem_backend.clear()
        assert count == 3
        assert len(mem_backend.all_records()) == 0

    def test_clear_by_scope(self, mem_backend):
        mem_backend.put(MemoryRecord(key="a", content="1", scope=MemoryScope.USER))
        mem_backend.put(MemoryRecord(key="b", content="2", scope=MemoryScope.SESSION))
        count = mem_backend.clear(MemoryScope.USER)
        assert count == 1
        assert len(mem_backend.all_records()) == 1


# ------------------------------------------------------------------
# Expiry
# ------------------------------------------------------------------


class TestExpiry:
    def test_expired_not_returned(self, mem_backend):
        r = MemoryRecord(key="exp", content="old", ttl=0.001)
        mem_backend.put(r)
        time.sleep(0.01)
        assert mem_backend.get("exp", MemoryScope.SESSION) is None

    def test_expired_excluded_from_list(self, mem_backend):
        mem_backend.put(MemoryRecord(key="live", content="ok"))
        mem_backend.put(MemoryRecord(key="dead", content="stale", ttl=0.001))
        time.sleep(0.01)
        records = mem_backend.list_by_scope(MemoryScope.SESSION)
        assert len(records) == 1
        assert records[0].key == "live"


# ------------------------------------------------------------------
# Typed fields round-trip
# ------------------------------------------------------------------


class TestFieldRoundTrip:
    def test_tags_roundtrip(self, mem_backend):
        r = MemoryRecord(key="t", content="v", tags=["a", "b"])
        mem_backend.put(r)
        got = mem_backend.get("t", MemoryScope.SESSION)
        assert got is not None
        assert got.tags == ["a", "b"]

    def test_provenance_roundtrip(self, mem_backend):
        prov = MemoryProvenance(source_type="tool", source_ids=["t1"])
        r = MemoryRecord(key="p", content="v", provenance=prov)
        mem_backend.put(r)
        got = mem_backend.get("p", MemoryScope.SESSION)
        assert got is not None
        assert got.provenance is not None
        assert got.provenance.source_type == "tool"

    def test_embedding_roundtrip(self, mem_backend):
        emb = [0.1, 0.2, 0.3, 0.4]
        r = MemoryRecord(key="e", content="v", embedding=emb)
        mem_backend.put(r)
        got = mem_backend.get("e", MemoryScope.SESSION)
        assert got is not None
        assert got.embedding is not None
        assert len(got.embedding) == 4
        assert abs(got.embedding[0] - 0.1) < 1e-5

    def test_metadata_roundtrip(self, mem_backend):
        r = MemoryRecord(key="m", content="v", metadata={"x": 42})
        mem_backend.put(r)
        got = mem_backend.get("m", MemoryScope.SESSION)
        assert got is not None
        assert got.metadata == {"x": 42}


# ------------------------------------------------------------------
# FTS5
# ------------------------------------------------------------------


class TestFts:
    def test_basic_search(self, mem_backend):
        mem_backend.put(MemoryRecord(key="lang", content="Python is great for ML"))
        mem_backend.put(MemoryRecord(key="food", content="Pizza is delicious"))
        results = mem_backend.search_fts("Python ML")
        assert len(results) >= 1
        assert results[0][0].key == "lang"
        assert results[0][1] > 0

    def test_search_empty_query(self, mem_backend):
        mem_backend.put(MemoryRecord(key="k", content="something"))
        assert mem_backend.search_fts("") == []

    def test_search_scoped(self, mem_backend):
        mem_backend.put(MemoryRecord(key="a", content="rust lang", scope=MemoryScope.USER))
        mem_backend.put(MemoryRecord(key="b", content="rust lang", scope=MemoryScope.SESSION))
        results = mem_backend.search_fts("rust", scope=MemoryScope.USER)
        assert len(results) == 1
        assert results[0][0].scope == MemoryScope.USER

    def test_fts_synced_on_update(self, mem_backend):
        r = MemoryRecord(key="k", content="old content")
        mem_backend.put(r)
        r2 = MemoryRecord(key="k", content="brand new topic")
        mem_backend.put(r2)
        assert mem_backend.search_fts("old") == []
        results = mem_backend.search_fts("brand new")
        assert len(results) == 1

    def test_fts_synced_on_delete(self, mem_backend):
        mem_backend.put(MemoryRecord(key="del", content="ephemeral data"))
        mem_backend.delete("del", MemoryScope.SESSION)
        assert mem_backend.search_fts("ephemeral") == []


# ------------------------------------------------------------------
# Embedding cache
# ------------------------------------------------------------------


class TestEmbeddingCache:
    def test_put_and_get(self, mem_backend):
        r = MemoryRecord(key="ec", content="v")
        mem_backend.put(r)
        emb = [0.5, 0.6, 0.7]
        mem_backend.put_embedding(r.record_id, "test_provider", "dim3", emb)
        got = mem_backend.get_embedding(r.record_id, "test_provider", "dim3")
        assert got is not None
        assert len(got) == 3
        assert abs(got[0] - 0.5) < 1e-5

    def test_get_missing(self, mem_backend):
        assert mem_backend.get_embedding("none", "p", "m") is None

    def test_different_providers(self, mem_backend):
        r = MemoryRecord(key="mp", content="v")
        mem_backend.put(r)
        mem_backend.put_embedding(r.record_id, "openai", "3-small", [0.1, 0.2])
        mem_backend.put_embedding(r.record_id, "local", "minilm", [0.3, 0.4])
        e1 = mem_backend.get_embedding(r.record_id, "openai", "3-small")
        e2 = mem_backend.get_embedding(r.record_id, "local", "minilm")
        assert e1 is not None and abs(e1[0] - 0.1) < 1e-5
        assert e2 is not None and abs(e2[0] - 0.3) < 1e-5

    def test_cascade_delete(self, mem_backend):
        r = MemoryRecord(key="cd", content="v")
        mem_backend.put(r)
        mem_backend.put_embedding(r.record_id, "p", "m", [1.0])
        mem_backend.delete("cd", MemoryScope.SESSION)
        assert mem_backend.get_embedding(r.record_id, "p", "m") is None


# ------------------------------------------------------------------
# Embedding search
# ------------------------------------------------------------------


class TestEmbeddingSearch:
    def test_cosine_search(self, mem_backend):
        mem_backend.put(MemoryRecord(key="a", content="x", embedding=[1.0, 0.0, 0.0]))
        mem_backend.put(MemoryRecord(key="b", content="y", embedding=[0.0, 1.0, 0.0]))
        results = mem_backend.search_embedding([0.9, 0.1, 0.0], limit=2)
        assert len(results) == 2
        assert results[0][0].key == "a"
        assert results[0][1] > results[1][1]


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


class TestPersistence:
    def test_survives_reopen(self, backend):
        backend.put(MemoryRecord(key="persist", content="data", scope=MemoryScope.USER))
        backend.close()
        backend2 = SQLiteMemoryBackend(db_path=backend._db_path)
        got = backend2.get("persist", MemoryScope.USER)
        assert got is not None
        assert got.content == "data"
        backend2.close()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


class TestFactory:
    def test_create_sqlite(self):
        from houyi.adapters.memory.backends import create_backend

        b = create_backend("sqlite", db_path=":memory:")
        assert isinstance(b, SQLiteMemoryBackend)
        b.close()

    def test_create_unknown_raises(self):
        from houyi.adapters.memory.backends import create_backend

        with pytest.raises(KeyError, match="Unknown memory backend"):
            create_backend("nosuchbackend")
