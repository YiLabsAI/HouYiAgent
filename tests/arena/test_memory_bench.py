"""Unit tests for the memory benchmark harness.

These tests exercise the pure modules (types / metrics / judge / dataset
fixture / runner with stubs) without any LLM, dataset, or network
dependency, so they live under tests/arena/ (joins make check's unit +
coverage gate). The end-to-end integration smoke that drives a real
MemoryIngestor lives separately under
tests/integration/benchmark/test_memory_halumem.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from houyi.arena.memory_bench.__main__ import (
    BenchRunConfig,
    _report_to_dict,
    _run_isolated_sessions,
)
from houyi.arena.memory_bench.cells import (
    CellCheckResult,
    CellDataset,
    CellRunner,
    cell_matrix,
    cell_pass_rate,
    override_status,
    write_cells_report,
)
from houyi.arena.memory_bench.dataset import (
    _session_from_hf_row,
    load_synthetic_fixture,
)
from houyi.arena.memory_bench.judge import (
    JudgeVerdict,
    LLMMemoryJudge,
    StubMemoryJudge,
    _parse_verdict,
)
from houyi.arena.memory_bench.metrics import (
    BenchMetrics,
    ExtractionMetrics,
    QAMetrics,
    UpdateMetrics,
)
from houyi.arena.memory_bench.runner import (
    MemoryBenchRunner,
    SubstringAnswerer,
)
from houyi.arena.memory_bench.timing import (
    PATH_KIND_SYNC_INLINE,
    PATH_KIND_TIERED_ASYNC,
    BenchTimings,
    _percentile,
)
from houyi.arena.memory_bench.types import (
    BenchSession,
    DialogueTurn,
    MemoryPoint,
    QAItem,
    UpdatePair,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetricsArithmetic:
    """Pure formulas; one assertion per branch."""

    def test_extraction_recall(self) -> None:
        m = ExtractionMetrics(gold_total=4, predicted_total=3, recalled=3, accurate=2)
        assert m.memory_recall == 0.75
        assert m.memory_accuracy == pytest.approx(2 / 3)
        assert m.f1 > 0

    def test_extraction_zero_safe(self) -> None:
        m = ExtractionMetrics(gold_total=0, predicted_total=0, recalled=0, accurate=0)
        assert m.memory_recall == 0.0
        assert m.memory_accuracy == 0.0
        assert m.f1 == 0.0
        # When salience is also zero, weighted recall falls back to plain.
        assert m.weighted_memory_recall == 0.0

    def test_weighted_recall(self) -> None:
        m = ExtractionMetrics(
            gold_total=2,
            predicted_total=2,
            recalled=1,
            accurate=2,
            weighted_recalled=3.0,
            salience_total=4.0,
        )
        assert m.weighted_memory_recall == 0.75

    def test_update_partition(self) -> None:
        m = UpdateMetrics(target_total=4, correct=2, wrong=1, missed=1)
        assert m.upd_acc == 0.5
        assert m.upd_hall == 0.25
        assert m.upd_omit == 0.25

    def test_qa_partition(self) -> None:
        m = QAMetrics(total=10, correct=7, hallucinated=2, omitted=1)
        assert m.qa_acc == 0.7
        assert m.qa_hall == 0.2
        assert m.qa_omit == 0.1

    def test_merge_aggregates(self) -> None:
        a = BenchMetrics(
            extraction=ExtractionMetrics(2, 2, 1, 1, 1.0, 2.0),
            update=UpdateMetrics(1, 1, 0, 0),
            qa=QAMetrics(2, 1, 0, 1),
        )
        b = BenchMetrics(
            extraction=ExtractionMetrics(2, 1, 0, 1, 0.0, 2.0),
            update=UpdateMetrics(1, 0, 1, 0),
            qa=QAMetrics(1, 0, 1, 0),
        )
        merged = a.merge(b)
        assert merged.extraction.gold_total == 4
        assert merged.update.correct == 1
        assert merged.update.wrong == 1
        assert merged.qa.total == 3
        # Error-propagation derived values reflect merged accuracy.
        assert merged.extraction_error == pytest.approx(1 - 2 / 3)
        assert merged.update_error == pytest.approx(0.5)


class TestCellMatrix:
    def test_matrix_size(self) -> None:
        cells = cell_matrix()

        assert len(cells) == 17
        assert cell_pass_rate() >= 0.8

    def test_dataset_mapping_covers(self) -> None:
        # Every dataset family that the closeout doc names must appear
        # at least once in the matrix.
        seen = {c.dataset for c in cell_matrix()}
        assert CellDataset.LOCOMO in seen
        assert CellDataset.HALUMEM in seen
        assert CellDataset.ADVERSARIAL in seen
        assert CellDataset.DREAMER in seen


class TestCellRunner:
    def test_run_static_default(self) -> None:
        report = CellRunner().run(run_id="r1")
        assert report.run_id == "r1"
        assert report.total == 17
        # static check mirrors each cell's bundled status verbatim
        passed = sum(1 for o in report.outcomes if o.status == "passed")
        assert passed == report.passed >= 13

    def test_run_check_override(self) -> None:
        def force_passed(_cell):  # type: ignore[no-untyped-def]
            return CellCheckResult(status="passed", detail="forced")

        runner = CellRunner(checks={"Q1": force_passed, "R6": force_passed, "U3": force_passed})
        report = runner.run(run_id="r2")
        assert report.pass_rate == 1.0
        q1 = next(o for o in report.outcomes if o.cell.cell_id == "Q1")
        assert q1.status == "passed"
        assert q1.detail == "forced"

    def test_run_check_raises(self) -> None:
        def boom(_cell):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

        report = CellRunner(checks={"E1": boom}).run(run_id="r3")
        e1 = next(o for o in report.outcomes if o.cell.cell_id == "E1")
        assert e1.status == "failed"
        assert "kaboom" in e1.detail

    def test_override_helper(self) -> None:
        cell = cell_matrix()[0]
        bumped = override_status(cell, status="failed", evidence="ext run")
        assert bumped.status == "failed"
        assert bumped.evidence == "ext run"
        # original unchanged
        assert cell.status == "passed"

    def test_write_report_files(self, tmp_path) -> None:
        report = CellRunner().run(run_id="rid-x")
        out_dir = write_cells_report(report, tmp_path)
        assert out_dir == tmp_path / "rid-x"
        cells_json = json.loads((out_dir / "cells.json").read_text("utf-8"))
        assert cells_json["run_id"] == "rid-x"
        assert cells_json["total"] == 17
        assert "by_dataset" in cells_json
        md = (out_dir / "summary.md").read_text("utf-8")
        assert "17-cell report" in md
        assert "| Cell |" in md


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class TestStubJudge:
    """Deterministic stub judge — verifies the matching rule."""

    def test_extraction_substring(self) -> None:
        j = StubMemoryJudge()
        assert j.judge_extraction("user lives in Beijing", "lives_in: Beijing").kind == "correct"

    def test_extraction_empty_is_missing(self) -> None:
        assert StubMemoryJudge().judge_extraction("anything", "").kind == "missing"

    def test_extraction_disjoint_is_wrong(self) -> None:
        v = StubMemoryJudge().judge_extraction("user lives in Beijing", "favourite color: blue")
        assert v.kind == "wrong"

    def test_update_keeps_old_missing(self) -> None:
        v = StubMemoryJudge().judge_update("Beijing", "Shanghai", "Beijing")
        assert v.kind == "missing"

    def test_update_correct_supersession(self) -> None:
        v = StubMemoryJudge().judge_update("Beijing", "Shanghai", "lives_in: Shanghai")
        assert v.kind == "correct"

    def test_qa_delegates_to_extraction(self) -> None:
        v = StubMemoryJudge().judge_qa("Where?", "Beijing", "Beijing")
        assert v.kind == "correct"


class TestVerdictParser:
    """The LLM judge's reply parser handles formatting variance."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("CORRECT", "correct"),
            ("WRONG\nthe candidate is fabricated", "wrong"),
            ("MISSING", "missing"),
            ("correct.", "correct"),  # trailing punctuation tolerated
            (" wrong ", "wrong"),
            ("", "missing"),  # empty falls back to missing
            ("???", "missing"),  # unparseable falls back to missing
        ],
    )
    def test_parse_branches(self, text: str, expected: str) -> None:
        assert _parse_verdict(text).kind == expected


class TestLLMJudgeRequiresAdapter:
    def test_none_adapter_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_adapter"):
            LLMMemoryJudge(None)

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            LLMMemoryJudge(_StubAdapter(["CORRECT"]), max_retries=-1)


@dataclass
class _StubAdapterResponse:
    content: str


class _StubAdapter:
    """Async adapter that returns a scripted sequence of responses.

    Each entry is either a string (success: emitted as content) or an
    Exception instance (failure: raised on the call). Run-time behaviour
    mirrors the adapter contract LLMMemoryJudge expects.
    """

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    async def chat(self, messages, *, model, temperature, max_tokens):
        self.calls += 1
        if not self._script:
            raise AssertionError("StubAdapter received unexpected extra call")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _StubAdapterResponse(content=item)


class TestLLMJudgeRetry:
    """LLMMemoryJudge retries transient parse / transport failures."""

    def test_retries_on_exception(self) -> None:
        adapter = _StubAdapter([ConnectionError("network blip"), "CORRECT"])
        judge = LLMMemoryJudge(adapter, max_retries=2)
        verdict = judge.judge_extraction("alice lives in beijing", "lives in beijing")
        assert verdict.kind == "correct"
        assert adapter.calls == 2

    def test_retries_on_unparseable(self) -> None:
        adapter = _StubAdapter(["not a verdict at all", "WRONG\nmismatch"])
        judge = LLMMemoryJudge(adapter, max_retries=2)
        verdict = judge.judge_qa("q?", "gold", "candidate")
        assert verdict.kind == "wrong"
        assert adapter.calls == 2

    def test_retries_on_empty_content(self) -> None:
        adapter = _StubAdapter(["", "MISSING\nout of scope"])
        judge = LLMMemoryJudge(adapter, max_retries=2)
        verdict = judge.judge_qa("q?", "gold", "candidate")
        # Legitimate MISSING from the model is preserved (not retried).
        assert verdict.kind == "missing"
        assert verdict.reason == "out of scope"
        assert adapter.calls == 2

    def test_legitimate_missing_not_retried(self) -> None:
        adapter = _StubAdapter(["MISSING\ndoes not match"])
        judge = LLMMemoryJudge(adapter, max_retries=2)
        verdict = judge.judge_extraction("gold", "predicted")
        assert verdict.kind == "missing"
        assert adapter.calls == 1  # No retry on legitimate MISSING.

    def test_exhausts_retries_then_degrades(self) -> None:
        adapter = _StubAdapter([ConnectionError("x"), ConnectionError("y"), ConnectionError("z")])
        judge = LLMMemoryJudge(adapter, max_retries=2)
        verdict = judge.judge_extraction("gold", "predicted")
        assert verdict.kind == "missing"
        assert "judge call failed" in verdict.reason
        assert adapter.calls == 3  # 1 + 2 retries.

    def test_short_circuits(self) -> None:
        adapter = _StubAdapter([])  # Should never be invoked.
        judge = LLMMemoryJudge(adapter, max_retries=2)
        verdict = judge.judge_extraction("gold", "")
        assert verdict.kind == "missing"
        assert verdict.reason == "predicted is empty"
        assert adapter.calls == 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestSyntheticFixture:
    def test_three_sessions(self) -> None:
        sessions = load_synthetic_fixture()
        ids = {s.session_id for s in sessions}
        assert ids == {
            "fixture-extract-001",
            "fixture-update-001",
            "fixture-vague-001",
        }

    def test_update_fixture_gold(self) -> None:
        update = next(s for s in load_synthetic_fixture() if s.session_id == "fixture-update-001")
        assert len(update.gold_updates) == 1
        assert update.gold_updates[0].new_text == "Shanghai"


class TestHFRowTranslation:
    """The defensive HF row reader handles partial / aliased schemas."""

    def test_full_row(self) -> None:
        row = {
            "session_id": "s-001",
            "dialogue": [
                {"role": "user", "content": "I live in Beijing."},
                {"role": "assistant", "content": "OK."},
            ],
            "memory_points": [{"id": "m1", "text": "user lives in Beijing", "salience": 1.5}],
            "memory_updates": [
                {"id": "u1", "old_id": "m1", "new_text": "Shanghai"},
            ],
            "qa_pairs": [
                {"id": "q1", "question": "Where?", "answer": "Beijing"},
            ],
            "metadata": {"persona": "Alice"},
        }
        session = _session_from_hf_row(row, fallback_id="halumem-fallback")
        assert session.session_id == "s-001"
        assert len(session.dialogue) == 2
        assert session.gold_memories[0].salience == 1.5
        assert session.gold_updates[0].new_text == "Shanghai"
        assert session.qa_items[0].question == "Where?"
        assert session.metadata == {"persona": "Alice"}

    def test_aliased_columns(self) -> None:
        # ``conversation``/``memories`` are accepted as aliases.
        row = {
            "conversation": [{"speaker": "user", "text": "hi"}],
            "memories": [{"text": "user said hi"}],
            "questions": [{"q": "what?", "a": "hi"}],
        }
        session = _session_from_hf_row(row, fallback_id="fallback-007")
        assert session.session_id == "fallback-007"
        assert session.dialogue[0].role == "user"
        assert session.qa_items[0].answer == "hi"

    def test_missing_fields_degrade_gracefully(self) -> None:
        session = _session_from_hf_row({}, fallback_id="empty-1")
        assert session.session_id == "empty-1"
        assert session.dialogue == ()
        assert session.gold_memories == ()


# ---------------------------------------------------------------------------
# Runner with stub ingestor
# ---------------------------------------------------------------------------


@dataclass
class _FakeIngestor:
    """Records calls so tests can assert on the ingest path."""

    calls: list[tuple[str, str | None]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    async def ingest_turn(self, text, *, source_anchor, recent_targets=()):
        self.calls.append((text, source_anchor))
        return None


class _FixedReader:
    """Returns a fixed set of active memories regardless of namespace."""

    def __init__(self, memories: list[str]) -> None:
        self._memories = memories

    def list_active_memories(self, namespace: str) -> list[str]:
        return list(self._memories)


class TestRunnerScoring:
    @pytest.mark.asyncio
    async def test_perfect_extraction(self) -> None:
        session = BenchSession(
            session_id="t1",
            dialogue=(DialogueTurn("user", "hello"),),
            gold_memories=(MemoryPoint(point_id="m1", text="user lives in Beijing"),),
            # SubstringAnswerer matches question tokens (len >= 3) against
            # the memory string; "lives" appears in both so the answerer
            # surfaces the Beijing memory and the stub judge marks it correct.
            qa_items=(QAItem(qa_id="q1", question="Where lives the user?", answer="Beijing"),),
        )
        runner = MemoryBenchRunner(
            ingestor=_FakeIngestor(),
            reader=_FixedReader(["lives_in: Beijing"]),
        )
        report = await runner.run([session])
        assert report.aggregate.extraction.memory_recall == 1.0
        assert report.aggregate.extraction.memory_accuracy == 1.0
        assert report.aggregate.qa.qa_acc == 1.0

    @pytest.mark.asyncio
    async def test_fabrication_lowers_accuracy(self) -> None:
        session = BenchSession(
            session_id="t2",
            dialogue=(DialogueTurn("user", "hi"),),
            gold_memories=(MemoryPoint(point_id="m1", text="user lives in Beijing"),),
        )
        runner = MemoryBenchRunner(
            ingestor=_FakeIngestor(),
            reader=_FixedReader(
                [
                    "lives_in: Beijing",
                    "favourite_color: purple",  # fabricated; no gold support
                ]
            ),
        )
        report = await runner.run([session])
        # Recall stays at 1.0 (Beijing is covered) but accuracy drops to 0.5.
        assert report.aggregate.extraction.memory_recall == 1.0
        assert report.aggregate.extraction.memory_accuracy == 0.5

    @pytest.mark.asyncio
    async def test_update_correct(self) -> None:
        session = BenchSession(
            session_id="t3",
            dialogue=(DialogueTurn("user", "moved"),),
            gold_memories=(MemoryPoint(point_id="m1", text="user lives in Beijing"),),
            gold_updates=(UpdatePair(update_id="u1", old_point_id="m1", new_text="Shanghai"),),
        )
        runner = MemoryBenchRunner(
            ingestor=_FakeIngestor(),
            reader=_FixedReader(["lives_in: Shanghai"]),
        )
        report = await runner.run([session])
        assert report.aggregate.update.upd_acc == 1.0

    @pytest.mark.asyncio
    async def test_dialogue_drives_ingestor(self) -> None:
        ingestor = _FakeIngestor()
        session = BenchSession(
            session_id="t4",
            dialogue=(
                DialogueTurn("user", "first"),
                DialogueTurn("assistant", "ack"),  # assistant turns skipped
                DialogueTurn("user", "second"),
            ),
        )
        runner = MemoryBenchRunner(ingestor=ingestor, reader=_FixedReader([]))
        await runner.run([session])
        assert [text for text, _ in ingestor.calls] == ["first", "second"]
        # Source anchors are unique per turn for provenance traceability.
        anchors = [a for _, a in ingestor.calls]
        assert anchors == ["t4:turn-0", "t4:turn-2"]

    @pytest.mark.asyncio
    async def test_report_has_config(self) -> None:
        runner = MemoryBenchRunner(ingestor=_FakeIngestor(), reader=_FixedReader([]))
        report = await runner.run([BenchSession(session_id="t5", dialogue=())])
        payload = _report_to_dict(
            report,
            run_config=BenchRunConfig(
                extractor_model="Qwen/Qwen2.5-7B-Instruct",
                extractor_retries=2,
                extractor_json_mode=True,
            ),
        )
        assert payload["config"] == {
            "extractor_model": "Qwen/Qwen2.5-7B-Instruct",
            "extractor_retries": 2,
            "extractor_json_mode": True,
            "shared_memory_across_sessions": False,
        }

    @pytest.mark.asyncio
    async def test_prediction_and_qa_details(self) -> None:
        session = BenchSession(
            session_id="t6",
            dialogue=(),
            gold_memories=(MemoryPoint(point_id="m1", text="user lives in Beijing"),),
            qa_items=(QAItem(qa_id="q1", question="Where?", answer="Beijing"),),
        )
        runner = MemoryBenchRunner(
            ingestor=_FakeIngestor(),
            reader=_FixedReader(["lives_in: Beijing", "favorite_food: pizza"]),
        )
        report = await runner.run([session])
        payload = _report_to_dict(
            report,
            run_config=BenchRunConfig(
                extractor_model="model",
                extractor_retries=1,
                extractor_json_mode=True,
                debug_predictions=True,
            ),
        )
        details = payload["sessions"][0]
        assert details["active_memories"] == ["lives_in: Beijing", "favorite_food: pizza"]
        assert details["gold_recall_judgments"][0]["gold"] == "user lives in Beijing"
        assert details["missing_gold"] == []
        assert details["wrong_predictions"][0]["predicted"] == "favorite_food: pizza"
        assert details["qa_judgments"][0]["question"] == "Where?"

    @pytest.mark.asyncio
    async def test_sessions_isolate_namespaces(self, tmp_path) -> None:
        class _Args:
            namespace = "bench"
            extractor_model = None
            extractor_retries = 1
            disable_extractor_json_mode = False

        sessions = [
            BenchSession(session_id="s1", dialogue=(), gold_memories=()),
            BenchSession(session_id="s2", dialogue=(), gold_memories=()),
        ]
        report = await _run_isolated_sessions(
            _Args(),
            sessions=sessions,
            db_path=tmp_path / "bench.db",
            judge=StubMemoryJudge(),
            answerer=SubstringAnswerer(),
        )
        assert [session.session_id for session in report.sessions] == ["s1", "s2"]


class TestSubstringAnswerer:
    def test_returns_first_matching_memory(self) -> None:
        ans = SubstringAnswerer()
        out = ans("Where does the user live?", ["job: engineer", "lives_in: Beijing"])
        assert out == "lives_in: Beijing"

    def test_no_match_empty(self) -> None:
        out = SubstringAnswerer()("any?", [])
        assert out == ""


# ---------------------------------------------------------------------------
# Smoke: types are JSON-friendly enough for report serialisation
# ---------------------------------------------------------------------------


class TestTypesSerialisation:
    def test_session_round_trip(self) -> None:
        session = BenchSession(
            session_id="s",
            dialogue=(DialogueTurn("user", "hi"),),
            gold_memories=(MemoryPoint(point_id="m", text="x"),),
            metadata={"k": "v"},
        )
        # We don't ship a serializer; just make sure the fields are
        # JSON-encodable when shallowly converted.
        payload = json.dumps(
            {
                "session_id": session.session_id,
                "dialogue": [{"role": t.role, "content": t.content} for t in session.dialogue],
                "metadata": session.metadata,
            }
        )
        assert "user" in payload

    def test_judge_verdict_kind_field(self) -> None:
        v = JudgeVerdict("correct", "ok")
        assert v.kind == "correct"
        assert v.reason == "ok"


# ---------------------------------------------------------------------------
# Bench timings (sync vs tiered comparison primitives)
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_zero(self) -> None:
        assert _percentile([], 50.0) == 0.0

    def test_single_returns_value(self) -> None:
        assert _percentile([7.5], 95.0) == 7.5

    def test_p50_interpolates(self) -> None:
        # Median of [1, 2, 3, 4] interpolates to 2.5.
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5

    def test_p95_near_max(self) -> None:
        samples = [float(i) for i in range(1, 101)]
        # p95 of 1..100 lies between 95 and 96; tolerate either side.
        assert 95.0 <= _percentile(samples, 95.0) <= 96.0


class TestBenchTimings:
    def test_empty_path_kind(self) -> None:
        t = BenchTimings.empty(PATH_KIND_TIERED_ASYNC)
        assert t.path_kind == PATH_KIND_TIERED_ASYNC
        assert t.ingest_calls == 0
        assert t.ingest_total_ms == 0.0

    def test_from_samples_summarises(self) -> None:
        t = BenchTimings.from_samples(
            PATH_KIND_SYNC_INLINE,
            [10.0, 20.0, 30.0, 40.0],
            drain_total_ms=5.5,
            extractor_calls=4,
        )
        assert t.path_kind == PATH_KIND_SYNC_INLINE
        assert t.ingest_calls == 4
        assert t.ingest_total_ms == 100.0
        assert t.ingest_max_ms == 40.0
        assert t.drain_total_ms == 5.5
        assert t.extractor_calls == 4

    def test_from_samples_empty(self) -> None:
        t = BenchTimings.from_samples(
            PATH_KIND_TIERED_ASYNC, [], drain_total_ms=12.0, extractor_calls=2
        )
        assert t.ingest_calls == 0
        assert t.drain_total_ms == 12.0
        assert t.extractor_calls == 2

    def test_merge_combines_counts(self) -> None:
        a = BenchTimings.from_samples(PATH_KIND_TIERED_ASYNC, [10.0, 20.0], drain_total_ms=5.0)
        b = BenchTimings.from_samples(PATH_KIND_TIERED_ASYNC, [30.0, 40.0], extractor_calls=3)
        merged = a.merge(b)
        assert merged.ingest_calls == 4
        assert merged.ingest_total_ms == 100.0
        assert merged.ingest_max_ms == 40.0
        assert merged.drain_total_ms == 5.0
        assert merged.extractor_calls == 3
        assert merged.path_kind == PATH_KIND_TIERED_ASYNC

    def test_merge_two_empty(self) -> None:
        a = BenchTimings.empty(PATH_KIND_SYNC_INLINE)
        b = BenchTimings.empty(PATH_KIND_SYNC_INLINE)
        merged = a.merge(b)
        assert merged.ingest_calls == 0
        assert merged.path_kind == PATH_KIND_SYNC_INLINE


# ---------------------------------------------------------------------------
# Runner timing wiring (drain hook + cost probe)
# ---------------------------------------------------------------------------


class _NoopReader:
    def list_active_memories(self, namespace: str) -> list[str]:
        return []


class _RecordingIngestor:
    """Trivial ingest target for runner timing tests; no real memory work."""

    def __init__(self) -> None:
        self.calls = 0

    async def ingest_turn(self, text, *, source_anchor, recent_targets=()):
        self.calls += 1
        return None


class TestRunnerTimings:
    @pytest.mark.asyncio
    async def test_records_ingest_count(self) -> None:
        ing = _RecordingIngestor()
        runner = MemoryBenchRunner(
            ing, _NoopReader(), path_kind=PATH_KIND_SYNC_INLINE, namespace="bench"
        )
        sessions = [
            BenchSession(
                session_id="s1",
                dialogue=(DialogueTurn("user", "hi"), DialogueTurn("user", "again")),
                gold_memories=(),
            )
        ]
        report = await runner.run(sessions)
        assert ing.calls == 2
        assert report.timings.ingest_calls == 2
        assert report.timings.path_kind == PATH_KIND_SYNC_INLINE
        assert report.timings.drain_total_ms == 0.0

    @pytest.mark.asyncio
    async def test_drain_cost_hook(self) -> None:
        ing = _RecordingIngestor()
        cost = {"n": 0}

        async def drain() -> None:
            # Simulate worker bumping the extractor count during drain.
            cost["n"] += 3

        runner = MemoryBenchRunner(
            ing,
            _NoopReader(),
            path_kind=PATH_KIND_TIERED_ASYNC,
            drain_callback=drain,
            cost_probe=lambda: cost["n"],
            namespace="bench",
        )
        sessions = [
            BenchSession(
                session_id="s1",
                dialogue=(DialogueTurn("user", "hello"),),
                gold_memories=(),
            )
        ]
        report = await runner.run(sessions)
        assert report.timings.path_kind == PATH_KIND_TIERED_ASYNC
        assert report.timings.extractor_calls == 3
        assert report.sessions[0].timing_samples is not None
        assert report.sessions[0].timing_samples.drain_total_ms >= 0.0
