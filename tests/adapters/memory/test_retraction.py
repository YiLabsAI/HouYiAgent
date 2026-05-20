from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.resolver import MemoryWriterTools
from houyi.adapters.memory.retraction import (
    RetractionDetector,
    RetractionOrchestrator,
    RetractionTarget,
)

# Curated corpus exercising each pattern label. Sized at 30 positive +
# 20 negative cases so the >=90% recall gate requires at most
# 3 misses on positives and zero false positives on the obvious negatives.
POSITIVE_CASES: tuple[tuple[str, str], ...] = (
    # explicit_wrong (5)
    ("I was wrong about the meeting time", "explicit_wrong"),
    ("I am wrong, the deadline is Friday", "explicit_wrong"),
    ("I made a mistake on the address", "explicit_wrong"),
    ("My bad, the report is due tomorrow", "explicit_wrong"),
    ("I misspoke earlier about the budget", "explicit_wrong"),
    # actually_correction (5)
    ("Actually, that's not right", "actually_correction"),
    ("Wait, the file isn't in /tmp", "actually_correction"),
    ("Hold on, my email is not the gmail one", "actually_correction"),
    ("On second thought, that wasn't the correct branch", "actually_correction"),
    ("Come to think of it, the server isn't us-east", "actually_correction"),
    # scratch_that (5)
    ("Scratch that, use the staging URL instead", "scratch_that"),
    ("Forget what I said about the password", "scratch_that"),
    ("Forget that I said anything about the deal", "scratch_that"),
    ("Ignore what I just said about the price", "scratch_that"),
    ("Never mind what I said about the launch date", "scratch_that"),
    # let_me_correct (5)
    ("Let me correct myself: the answer is 42", "let_me_correct"),
    ("Let me fix that: the team has 6 members", "let_me_correct"),
    ("Correction: the server is in eu-west", "let_me_correct"),
    ("To correct myself, the build is green now", "let_me_correct"),
    ("I take that back, the test is actually passing", "let_me_correct"),
    # earlier_statement_void (5)
    ("What I said earlier was wrong", "earlier_statement_void"),
    ("What I told you before was incorrect", "earlier_statement_void"),
    ("What I said a moment ago was not right", "earlier_statement_void"),
    ("What I said a second ago is inaccurate", "earlier_statement_void"),
    ("What I said before is wrong", "earlier_statement_void"),
    # zh_retraction (5)
    (
        "\u521a\u624d\u8bf4\u9519\u4e86\uff0c\u4e0d\u662f\u5468\u4e00\u662f\u5468\u4e8c",
        "zh_retraction",
    ),
    ("\u6211\u641e\u9519\u4e86\uff0c\u662f Python 3.11", "zh_retraction"),
    ("\u4e0d\u5bf9\u4e0d\u5bf9\uff0c\u662f 8080 \u7aef\u53e3", "zh_retraction"),
    ("\u7b97\u6211\u6ca1\u8bf4\uff0c\u91cd\u6765", "zh_retraction"),
    ("\u53d6\u6d88\u4e4b\u524d\u7684\u6307\u4ee4", "zh_retraction"),
)

NEGATIVE_CASES: tuple[str, ...] = (
    "The meeting is at 3pm on Friday",
    "Please update the README",
    "I prefer dark mode",
    "We deployed the new version yesterday",
    "Can you review the PR?",
    "The server is in us-east-1",
    "I work at a startup",
    "My favorite language is Python",
    "Let me know when you're done",
    "The build passed all checks",
    "Actually deploying now",  # 'actually' alone, no negation, no retraction
    "Wait for the CI to finish",  # 'wait' alone, no negation
    "I will fix it tomorrow",  # 'fix' but not 'let me fix'
    "She corrected the typo",  # third-person, not self-correction
    "Never use sudo without thinking",  # 'never' but not 'never mind'
    "\u4eca\u5929\u5929\u6c14\u4e0d\u9519",  # today's weather is good (not retraction)
    "\u6211\u53bb\u8fc7\u5317\u4eac",  # I have been to Beijing
    "\u8bf7\u6c42\u5df2\u53d1\u9001",  # request has been sent
    "\u8bf7\u4f60\u68c0\u67e5\u4ee3\u7801",  # please review the code
    "\u6211\u540c\u610f\u8fd9\u4e2a\u65b9\u6848",  # I agree with the plan
)


@pytest.fixture
def detector() -> RetractionDetector:
    return RetractionDetector()


class TestRetractionPositives:
    """Each catalogued pattern must fire on its example sentences."""

    @pytest.mark.parametrize("text,expected_label", POSITIVE_CASES)
    def test_pattern_fires(
        self,
        detector: RetractionDetector,
        text: str,
        expected_label: str,
    ) -> None:
        signal = detector.detect(text)
        assert signal is not None, f"no match for: {text!r}"
        assert signal.label == expected_label


class TestRetractionNegatives:
    """Plain statements without retraction cues must not trigger."""

    @pytest.mark.parametrize("text", NEGATIVE_CASES)
    def test_no_false_positive(
        self,
        detector: RetractionDetector,
        text: str,
    ) -> None:
        assert detector.detect(text) is None


class TestRetractionGate:
    """Aggregate recall gate requires >=90% on the positive corpus."""

    def test_recall_above_threshold(self, detector: RetractionDetector) -> None:
        hits = sum(1 for text, _ in POSITIVE_CASES if detector.is_retraction(text))
        recall = hits / len(POSITIVE_CASES)
        assert recall >= 0.9, f"recall={recall:.2%} < 90%"

    def test_no_match_on_empty(self, detector: RetractionDetector) -> None:
        assert detector.detect("") is None
        assert detector.is_retraction("") is False


# ---------------------------------------------------------------------------
# RetractionOrchestrator
# ---------------------------------------------------------------------------


@pytest.fixture
def writer_setup(tmp_path) -> Iterator[tuple[MemoryWriterTools, SQLiteEntityStateView]]:
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    try:
        view = SQLiteEntityStateView(backend)
        inbox = SQLiteCandidateInbox(backend)
        yield MemoryWriterTools(view, inbox, namespace="ws"), view
    finally:
        backend.close()


def _seed_fact(view: SQLiteEntityStateView, attribute: str, value: str) -> None:
    view.upsert("ws", "user", attribute, value, valid_from=100.0)


class TestOrchestratorNoSignal:
    def test_no_signal_no_invalidation(self, writer_setup) -> None:
        tools, view = writer_setup
        _seed_fact(view, "city", "Beijing")
        orch = RetractionOrchestrator(RetractionDetector(), tools)
        result = orch.process("I love Beijing food", [RetractionTarget("user", "city")])
        assert result.signal is None
        assert result.invalidated == ()
        assert view.get_active("ws", "user", "city")  # untouched


class TestOrchestratorWithSignal:
    def test_invalidates_target(self, writer_setup) -> None:
        tools, view = writer_setup
        _seed_fact(view, "city", "Beijing")
        orch = RetractionOrchestrator(RetractionDetector(), tools)
        result = orch.process(
            "Actually, that's not right",
            [RetractionTarget("user", "city")],
        )
        assert result.signal is not None
        assert result.signal.label == "actually_correction"
        assert result.invalidated == (RetractionTarget("user", "city"),)
        assert view.get_active("ws", "user", "city") == []

    def test_skips_unknown_target(self, writer_setup) -> None:
        tools, _view = writer_setup
        orch = RetractionOrchestrator(RetractionDetector(), tools)
        result = orch.process(
            "I was wrong about everything",
            [RetractionTarget("user", "ghost_attr")],
        )
        assert result.signal is not None
        assert result.invalidated == ()  # no active row to close

    def test_partial_invalidation(self, writer_setup) -> None:
        tools, view = writer_setup
        _seed_fact(view, "city", "Beijing")
        # No fact seeded for 'job' yet, only city.
        orch = RetractionOrchestrator(RetractionDetector(), tools)
        result = orch.process(
            "Scratch that",
            [
                RetractionTarget("user", "city"),
                RetractionTarget("user", "job"),
            ],
        )
        assert result.signal is not None
        assert result.invalidated == (RetractionTarget("user", "city"),)
        assert view.get_active("ws", "user", "city") == []
