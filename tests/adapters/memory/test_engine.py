"""MemoryEngine unit tests.

Covers the write pipeline (extract → classify → dedup → store),
read pipeline (recall → score → explain), manual approval flow,
and degradation scenarios (no embedding provider).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from houyi.adapters.embedding import NoOpEmbeddingProvider
from houyi.adapters.memory.engine import (
    _INTERNAL_QUALIFIER_KEYS,
    MemoryEngine,
    _build_temporal_turns,
)
from houyi.adapters.memory.reasoner import (
    MemoryReasoningInput,
    TurnEvidenceReasoningPolicy,
)
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.turn_context import reformat_recall_content
from houyi.adapters.memory.types import (
    CandidateStatus,
    MemoryPolicy,
    MemoryType,
    SessionContext,
)


@pytest.fixture()
def engine(tmp_path):
    """Engine with auto-approve and NoOp embeddings."""
    store = MemoryStore(data_dir=tmp_path)
    emb = NoOpEmbeddingProvider(dim=32)
    policy = MemoryPolicy(auto_approve=True)
    yield MemoryEngine(store, embedding_provider=emb, policy=policy)
    store.close()


@pytest.fixture()
def engine_no_emb(tmp_path):
    """Engine without embedding provider (lexical-only fallback)."""
    store = MemoryStore(data_dir=tmp_path)
    policy = MemoryPolicy(auto_approve=True)
    yield MemoryEngine(store, policy=policy)
    store.close()


@pytest.fixture()
def engine_manual_approve(tmp_path):
    """Engine requiring manual approval (auto_approve=False)."""
    store = MemoryStore(data_dir=tmp_path)
    yield MemoryEngine(store, policy=MemoryPolicy(auto_approve=False))
    store.close()


class TestWritePipeline:
    """Test extract → classify → dedup → store flow."""

    async def test_explicit_memory_stored(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that the deadline is Friday."},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) >= 1
        approved = [c for c in candidates if c.status == CandidateStatus.APPROVED]
        assert len(approved) >= 1
        assert approved[0].memory_type == MemoryType.FACT

    async def test_identity_extracted(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "My name is Alice and I work at ACME."},
        ]
        candidates = await engine.process_messages(messages)
        profile_cands = [c for c in candidates if c.memory_type == MemoryType.PROFILE]
        assert len(profile_cands) >= 1
        assert "Alice" in profile_cands[0].content

    async def test_preference_extracted(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "I prefer dark mode for all my editors."},
        ]
        candidates = await engine.process_messages(messages)
        pref_cands = [c for c in candidates if c.memory_type == MemoryType.PREFERENCE]
        assert len(pref_cands) >= 1

    async def test_constraint_extracted(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "Don't use any Java examples."},
        ]
        candidates = await engine.process_messages(messages)
        constraint_cands = [c for c in candidates if c.memory_type == MemoryType.CONSTRAINT]
        assert len(constraint_cands) >= 1

    async def test_assistant_messages_ignored(self, engine: MemoryEngine):
        messages = [
            {"role": "assistant", "content": "Remember that I am helpful."},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) == 0

    async def test_no_extraction_from_empty(self, engine: MemoryEngine):
        candidates = await engine.process_messages([])
        assert candidates == []

    async def test_duplicate_merged(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that the server is on port 8080."},
        ]
        await engine.process_messages(messages)
        candidates = await engine.process_messages(messages)
        merged = [c for c in candidates if c.status == CandidateStatus.MERGED]
        assert len(merged) >= 1


class TestManualApproval:
    """Test the manual approval flow."""

    async def test_pending_not_stored(self, engine_manual_approve: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that tests are important."},
        ]
        candidates = await engine_manual_approve.process_messages(messages)
        assert all(
            c.status in (CandidateStatus.PENDING, CandidateStatus.MERGED) for c in candidates
        )
        records = engine_manual_approve.store.all_records()
        assert len(records) == 0

    async def test_approve_stores_record(self, engine_manual_approve: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that the API key is 12345."},
        ]
        candidates = await engine_manual_approve.process_messages(messages)
        pending = [c for c in candidates if c.status == CandidateStatus.PENDING]
        assert len(pending) >= 1

        record = await engine_manual_approve.approve_candidate(pending[0])
        assert record.content == pending[0].content
        assert engine_manual_approve.store.get(record.key, record.scope) is not None


class TestRecallPipeline:
    """Test retrieval scoring and ranking."""

    async def test_recall_returns_results(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that Python is great for ML."},
        ]
        await engine.process_messages(messages)
        recalls = await engine.recall("What language for machine learning?")
        assert len(recalls) >= 1
        assert recalls[0].score > 0

    async def test_recall_empty_store(self, engine: MemoryEngine):
        recalls = await engine.recall("anything")
        assert recalls == []

    async def test_recall_context_text(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that the deploy target is k8s."},
        ]
        await engine.process_messages(messages)
        recalls = await engine.recall("deploy target")
        text = engine.recall_as_context_text(recalls)
        assert "deploy" in text.lower() or "k8s" in text.lower()

    async def test_recall_respects_top_k(self, engine: MemoryEngine):
        for i in range(10):
            await engine.process_messages(
                [
                    {"role": "user", "content": f"Remember fact number {i}."},
                ]
            )
        recalls = await engine.recall("facts", top_k=3)
        assert len(recalls) <= 3


class TestNoEmbeddingFallback:
    """Engine works without embedding provider (lexical-only)."""

    async def test_write_without_embedding(self, engine_no_emb: MemoryEngine):
        messages = [
            {"role": "user", "content": "Remember that Redis runs on port 6379."},
        ]
        candidates = await engine_no_emb.process_messages(messages)
        assert len(candidates) >= 1
        records = engine_no_emb.store.all_records()
        assert len(records) >= 1
        assert records[0].embedding is None

    async def test_recall_without_embedding(self, engine_no_emb: MemoryEngine):
        await engine_no_emb.process_messages(
            [
                {"role": "user", "content": "Remember that Redis runs on port 6379."},
            ]
        )
        recalls = await engine_no_emb.recall("Redis port")
        assert len(recalls) >= 1


class TestBuildContext:
    """Test build_context recall + formatting."""

    async def test_returns_context(self, engine: MemoryEngine):
        await engine.process_messages(
            [{"role": "user", "content": "Remember that Python is great for ML."}]
        )
        ctx = await engine.build_context("Python machine learning")
        assert ctx is not None
        assert "python" in ctx.lower() or "ml" in ctx.lower()

    async def test_returns_none_empty(self, engine: MemoryEngine):
        ctx = await engine.build_context("anything")
        assert ctx is None

    async def test_respects_top_k(self, engine: MemoryEngine):
        for i in range(5):
            await engine.process_messages(
                [{"role": "user", "content": f"Remember fact number {i}."}]
            )
        ctx = await engine.build_context("fact", top_k=2)
        if ctx:
            assert ctx.count("\n") <= 1


class TestAnswer:
    async def test_answer_uses_reasoner(self, engine: MemoryEngine):
        await engine.process_messages(
            [{"role": "user", "content": "Remember that preferred editor is neovim."}]
        )
        result = await engine.answer("what is my preferred editor")
        assert result.abstained is False
        assert "editor" in result.answer.lower() or "neovim" in result.answer.lower()

    async def test_answer_abstains_empty(self, engine: MemoryEngine):
        result = await engine.answer("what is my preference")
        assert result.abstained is True


class TestPrepareReasonerRecords:
    """_prepare_reasoner_records dedups and demotes content-free relations."""

    def _records(self, *contents: str):
        from houyi.adapters.memory.types import MemoryRecord

        return [MemoryRecord(key=f"k{i}", content=c) for i, c in enumerate(contents)]

    def test_duplicates_dropped(self):
        """Duplicate contents collapse to the highest-ranked copy."""
        records = self._records(
            "John has attribute has kids",
            "John shares interest with Maria",
            "John shares interest with Maria",
            "John  shares INTEREST with Maria",
        )
        out = MemoryEngine._prepare_reasoner_records(records)
        assert [r.content for r in out] == [
            "John has attribute has kids",
            "John shares interest with Maria",
        ]

    def test_relations_demoted(self):
        """Content-free relational facts move behind substantive facts."""
        records = self._records(
            "John shares interest with Maria",
            "John has value looking out for others",
            "Maria related to John",
            "John believes others don't have enough",
        )
        out = MemoryEngine._prepare_reasoner_records(records)
        assert [r.content for r in out] == [
            "John has value looking out for others",
            "John believes others don't have enough",
            "John shares interest with Maria",
            "Maria related to John",
        ]

    def test_detailed_relation_kept(self):
        """A relational fact carrying extra detail is not demoted."""
        records = self._records(
            "John shares interest with Maria in making desserts",
            "John shares interest with Maria",
        )
        out = MemoryEngine._prepare_reasoner_records(records)
        assert out[0].content == "John shares interest with Maria in making desserts"

    def test_order_preserved(self):
        """Distinct substantive facts keep their ranked order."""
        records = self._records("fact one", "fact two", "fact three")
        out = MemoryEngine._prepare_reasoner_records(records)
        assert [r.content for r in out] == ["fact one", "fact two", "fact three"]


class TestRecallContextText:
    """Test recall_as_context_text formatting."""

    async def test_format_includes_type(self, engine: MemoryEngine):
        await engine.process_messages(
            [{"role": "user", "content": "Remember that the API uses v2."}]
        )
        recalls = await engine.recall("API version")
        text = engine.recall_as_context_text(recalls)
        assert "score=" in text

    async def test_empty_recalls(self, engine: MemoryEngine):
        text = engine.recall_as_context_text([])
        assert text == ""


class TestRecordIndex:
    """_build_record_index resolves exact record_ids and fact-id aliases."""

    async def test_exact_ids(self, engine: MemoryEngine):
        engine.store.put("editor.preferred", "neovim", memory_type=MemoryType.PREFERENCE)
        engine.store.put("server.port", "8080", memory_type=MemoryType.FACT)
        index = engine._build_record_index()
        for record in engine.store.all_records():
            found = index.get(record.record_id)
            assert found is not None
            assert found.record_id == record.record_id

    async def test_fact_id_aliases(self, engine: MemoryEngine):
        engine.store.put("editor.preferred", "neovim", memory_type=MemoryType.PREFERENCE)
        engine.store.put("server.port", "8080", memory_type=MemoryType.FACT)
        index = engine._build_record_index()
        for record in engine.store.all_records():
            for strategy in ("A", "B"):
                fid = engine._record_to_fact_id(record, strategy)
                indexed = index.get(fid)
                assert indexed is not None
                assert indexed.record_id == record.record_id

    async def test_unknown_id_none(self, engine: MemoryEngine):
        engine.store.put("x.y", "z", memory_type=MemoryType.FACT)
        index = engine._build_record_index()
        assert index.get("fact:doesnotexist") is None


class TestAnswerTopK:
    """engine.answer must honor the caller's top_k verbatim (no forced floor)."""

    async def test_top_k_passthrough(self, engine: MemoryEngine, monkeypatch):
        await engine.process_messages(
            [{"role": "user", "content": "Remember that the API uses v2."}]
        )
        captured: dict[str, int] = {}
        original_recall = engine.recall

        async def spy(query, session_context=None, top_k=5):
            captured["top_k"] = top_k
            return await original_recall(query, session_context, top_k)

        monkeypatch.setattr(engine, "recall", spy)
        await engine.answer("API version", top_k=7)
        assert captured["top_k"] == 7


class TestEmbeddingWritePath:
    """Synchronous approve path does not perform embedding writes."""

    async def test_approve_embedding_empty(self, engine: MemoryEngine):
        await engine.process_messages(
            [{"role": "user", "content": "Remember that Redis port is 6379."}]
        )
        records = engine.store.all_records()
        assert len(records) >= 1
        record = records[0]
        assert record.embedding is None

    async def test_no_embedding_without_provider(self, engine_no_emb: MemoryEngine):
        await engine_no_emb.process_messages(
            [{"role": "user", "content": "Remember that port is 443."}]
        )
        records = engine_no_emb.store.all_records()
        assert len(records) >= 1
        assert records[0].embedding is None


class TestDeriveKey:
    """Test key derivation from candidate content."""

    async def test_key_from_content(self, engine: MemoryEngine):
        msgs = [{"role": "user", "content": "Remember that the sky is blue."}]
        await engine.process_messages(msgs)
        records = engine.store.all_records()
        assert len(records) >= 1
        assert records[0].key != ""
        assert "_" in records[0].key or len(records[0].key) > 0


class TestForgetting:
    """Test forgetting maintenance."""

    async def test_forgetting_evicts_stale(self, engine: MemoryEngine):
        engine.store.put("old", "stale data", memory_type=MemoryType.FACT)
        record = engine.store.get("old")
        assert record is not None
        record.decay = 0.05
        record.updated_at = 1.0
        engine.store.put_record(record)

        evicted = await engine.run_forgetting()
        assert evicted >= 1

    async def test_forgetting_no_eviction(self, engine: MemoryEngine):
        engine.store.put("fresh", "new data", memory_type=MemoryType.FACT)
        evicted = await engine.run_forgetting()
        assert evicted == 0
        assert len(engine.store.all_records()) == 1

    async def test_forgetting_empty_store(self, engine: MemoryEngine):
        evicted = await engine.run_forgetting()
        assert evicted == 0


class TestReformatRecallContent:
    """_reformat_recall_content renders only answer-relevant qualifiers."""

    def test_strips_internal_qualifiers(self):
        content = "Calvin took his Ferrari for a ride (time: 2023-03-25)"
        quals = {
            "date": "2023-03-25",
            "compound_type": "emotional_transition",
            "original_time": "yesterday",
            "fact_object": "Calvin took his Ferrari",
        }
        out = reformat_recall_content(content, quals, _INTERNAL_QUALIFIER_KEYS)
        assert "compound_type" not in out
        assert "original_time" not in out
        assert "fact_object" not in out
        assert "time: 2023-03-25" in out

    def test_keeps_meaningful_qualifiers(self):
        content = "Maria had dinner"
        quals = {"location": "Rome", "co_agent": "her mother"}
        out = reformat_recall_content(content, quals, _INTERNAL_QUALIFIER_KEYS)
        assert "location: Rome" in out
        assert "co_agent: her mother" in out

    def test_approximate_baked(self):
        # vague-recency report date must not look exact.
        content = "Calvin went to Tokyo (time: 2023-04-20)"
        quals = {"date": "2023-04-20", "date_certainty": "approximate", "original_time": "just"}
        out = reformat_recall_content(content, quals, _INTERNAL_QUALIFIER_KEYS)
        assert "reported on 2023-04-20, exact date earlier/uncertain" in out
        assert "(time: 2023-04-20)" not in out
        assert "date_certainty" not in out

    def test_approximate_qualifier(self):
        # Date present only in qualifiers (no baked "(time: ...)").
        content = "Calvin went to Tokyo"
        quals = {"date": "2023-04-20", "date_certainty": "approximate"}
        out = reformat_recall_content(content, quals, _INTERNAL_QUALIFIER_KEYS)
        assert "reported on 2023-04-20, exact date earlier/uncertain" in out
        assert "time: 2023-04-20" not in out


class TestBuildTemporalTurns:
    """_build_temporal_turns feeds the turn-evidence policy from the raw
    turn log. It must sort oldest-first (the resolver walks back for the
    previous session anchor) and tolerate malformed turns without bubbling
    into MemoryEngine.answer() (it runs before every query's policy chain).
    """

    @staticmethod
    def _turn(turn_id, speaker, content, obs_date, created_at, role="user"):
        extract_blob = json.dumps(
            {
                "observation_date": obs_date,
                "system_date": obs_date,
                "text": content,
                "speaker_name": speaker,
            }
        )
        return SimpleNamespace(
            turn_id=turn_id,
            role=role,
            content=content,
            metadata={"speaker": speaker, "extract_text": extract_blob},
            created_at=created_at,
        )

    @staticmethod
    def _backend(turns):
        return SimpleNamespace(
            list_raw_turns_by_namespace=lambda namespace, limit=2000: list(turns)
        )

    def test_no_backend(self):
        assert _build_temporal_turns(None, SessionContext(session_id="ns")) is None

    def test_no_method(self):
        backend = SimpleNamespace()
        assert _build_temporal_turns(backend, SessionContext(session_id="ns")) is None

    def test_empty_namespace(self):
        backend = self._backend([])
        assert _build_temporal_turns(backend, SessionContext(session_id="ns")) is None

    def test_sorts_oldest_first(self):
        # Backend returns newest-first (D3 tokyo before D2), matching the
        # real list_raw_turns_by_namespace ORDER BY created_at DESC. The
        # resolver needs oldest-first or prev_iso comes back empty.
        turns = [
            self._turn("D3:1", "Calvin", "I just went to Tokyo.", "2023-04-20", 2000.0),
            self._turn("D2:1", "Dave", "Hi Calvin, new car?", "2023-03-26", 1000.0),
        ]
        result = _build_temporal_turns(
            self._backend(turns), SessionContext(session_id="locomo:conv-50")
        )
        assert result is not None
        assert [t.occurred_at for t in result] == ["2023-03-26", "2023-04-20"]
        assert [t.turn_id for t in result] == ["D2:1", "D3:1"]

    async def test_resolves_range(self):
        turns = [
            self._turn("D3:1", "Calvin", "I just went to Tokyo.", "2023-04-20", 2000.0),
            self._turn("D2:1", "Dave", "Hi Calvin, new car?", "2023-03-26", 1000.0),
        ]
        built = _build_temporal_turns(
            self._backend(turns), SessionContext(session_id="locomo:conv-50")
        )
        assert built is not None
        request = MemoryReasoningInput(
            query="When did Calvin first travel to Tokyo?",
            recalls=[],
            records=[],
            turns=built,
        )
        result = await TurnEvidenceReasoningPolicy().answer(request)
        assert result is not None
        assert result.answer == "between 26 March and 20 April 2023"
        assert result.reason == "turn_evidence"

    def test_malformed_turn(self):
        # A turn with a non-numeric created_at must not bubble; the whole
        # build is wrapped so any malformed turn degrades to None.
        turns = [
            self._turn("D3:1", "Calvin", "I just went to Tokyo.", "2023-04-20", "not-a-number"),
        ]
        result = _build_temporal_turns(self._backend(turns), SessionContext(session_id="ns"))
        assert result is None

    def test_list_error(self):
        backend = SimpleNamespace(
            list_raw_turns_by_namespace=lambda namespace, limit=2000: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        )
        assert _build_temporal_turns(backend, SessionContext(session_id="ns")) is None
