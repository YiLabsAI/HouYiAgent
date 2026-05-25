"""LoCoMo loader tests.

The real LoCoMo corpus is ~50MB and lives outside the repo (cloned to
/Users/von/workspace/locomo during dev). Tests therefore drive a
self-contained mini-payload to verify the parser, and add an opt-in
smoke test that runs against the real file when present.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from houyi.adapters.memory.bench import (
    DEFAULT_LOCOMO_PATH,
    LoCoMoCase,
    load_locomo_all,
    load_locomo_balanced,
)


def _mini_corpus(num_samples: int = 3, qa_per_sample: int = 5) -> list[dict]:
    """Build a minimal LoCoMo-shaped JSON payload."""
    out: list[dict] = []
    for s in range(num_samples):
        sid = f"conv-{s:02d}"
        sample = {
            "sample_id": sid,
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1 Jan 2023",
                "session_1": [
                    {
                        "speaker": "Alice",
                        "dia_id": f"D1:{i + 1}",
                        "text": f"alice line {i} in sample {s}",
                    }
                    for i in range(3)
                ],
                "session_2_date_time": "2 Jan 2023",
                "session_2": [
                    {
                        "speaker": "Bob",
                        "dia_id": f"D2:{i + 1}",
                        "text": f"bob line {i} in sample {s}",
                    }
                    for i in range(2)
                ],
            },
            "qa": [
                {
                    "question": f"q{i} for sample {s}?",
                    "answer": f"a{i}-{s}",
                    "evidence": [f"D1:{i + 1}"],
                    "category": (i % 4) + 1,
                }
                for i in range(qa_per_sample)
            ],
        }
        out.append(sample)
    return out


@pytest.fixture()
def mini_path(tmp_path) -> Path:
    p = tmp_path / "locomo_mini.json"
    p.write_text(json.dumps(_mini_corpus()), encoding="utf-8")
    return p


class TestLoadAll:
    def test_parses_every_qa_pair(self, mini_path):
        cases = load_locomo_all(mini_path)
        assert len(cases) == 3 * 5  # 3 samples × 5 qa each

    def test_case_carries_sample_turns(self, mini_path):
        cases = load_locomo_all(mini_path)
        first = cases[0]
        # Each sample has 3 + 2 = 5 turns.
        assert len(first.sample.turns) == 5
        assert first.sample.speaker_a == "Alice"
        # Turns are sorted by session number.
        assert first.sample.turns[0].session_id == "session_1"
        assert first.sample.turns[-1].session_id == "session_2"

    def test_evidence_preserved(self, mini_path):
        cases = load_locomo_all(mini_path)
        assert all(c.evidence for c in cases)
        assert cases[0].evidence == ("D1:1",)

    def test_session_datetime_carried(self, mini_path):
        cases = load_locomo_all(mini_path)
        first_turn = cases[0].sample.turns[0]
        assert first_turn.session_datetime == "1 Jan 2023"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_locomo_all(tmp_path / "nope.json")

    def test_non_list_top_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_locomo_all(bad)


class TestLoadBalanced:
    def test_returns_at_most_n(self, mini_path):
        # mini corpus has 15 cases; n=10 returns exactly 10.
        cases = load_locomo_balanced(mini_path, n=10)
        assert len(cases) == 10

    def test_returns_all_small_corpus(self, mini_path):
        cases = load_locomo_balanced(mini_path, n=200)
        # mini has 15; loader returns all without padding.
        assert len(cases) == 15

    def test_round_robin_across_samples(self, mini_path):
        cases = load_locomo_balanced(mini_path, n=6)
        # 3 samples × 2 cases each → first 6 cases must include every sample.
        sample_ids = {c.sample_id for c in cases}
        assert sample_ids == {"conv-00", "conv-01", "conv-02"}
        # And the order should be RR: 00, 01, 02, 00, 01, 02
        assert [c.sample_id for c in cases] == [
            "conv-00",
            "conv-01",
            "conv-02",
            "conv-00",
            "conv-01",
            "conv-02",
        ]

    def test_deterministic(self, mini_path):
        a = load_locomo_balanced(mini_path, n=10)
        b = load_locomo_balanced(mini_path, n=10)
        assert [c.question for c in a] == [c.question for c in b]

    def test_n_must_be_positive(self, mini_path):
        with pytest.raises(ValueError):
            load_locomo_balanced(mini_path, n=0)


@pytest.mark.skipif(
    not DEFAULT_LOCOMO_PATH.exists(),
    reason="real LoCoMo corpus not present (clone snap-research/locomo)",
)
class TestRealCorpusSmoke:
    """Opt-in: only runs when the local clone is present."""

    def test_load_all_yields_many(self):
        cases = load_locomo_all()
        assert len(cases) >= 1500

    def test_default_slice_balanced(self):
        cases = load_locomo_balanced(n=200)
        assert len(cases) == 200
        # Round-robin across 10 samples → ~20 each.
        counts = Counter(c.sample_id for c in cases)
        assert min(counts.values()) >= 15
        assert max(counts.values()) <= 25

    def test_every_case_evidence_question(self):
        cases = load_locomo_balanced(n=200)
        for c in cases:
            assert isinstance(c, LoCoMoCase)
            assert c.question.strip()
            assert c.answer is not None
