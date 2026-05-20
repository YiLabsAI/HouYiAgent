"""Tests for the adversarial fixture schema validation and coverage checks."""

from __future__ import annotations

from collections import Counter

import pytest

from houyi.adapters.memory.bench import (
    AdversarialCase,
    AdversarialKind,
    load_adversarial_fixture,
)


@pytest.fixture(scope="module")
def cases() -> list[AdversarialCase]:
    return load_adversarial_fixture()


class TestFixtureShape:
    def test_at_least_50_cases(self, cases):
        assert len(cases) >= 50, f" requires ≥50 cases; got {len(cases)}"

    def test_unique_ids(self, cases):
        ids = [c.id for c in cases]
        assert len(set(ids)) == len(ids)

    def test_every_kind_represented(self, cases):
        kinds = {c.kind for c in cases}
        # All twelve enum members must appear at least once so the bench
        # harness has coverage of every documented failure mode.
        assert kinds == set(AdversarialKind), f"missing kinds: {set(AdversarialKind) - kinds}"

    def test_kind_minimum_density(self, cases):
        # Each kind needs ≥3 cases so per-kind aggregates are not too
        # noisy in the bench report.
        counts = Counter(c.kind for c in cases)
        for kind, n in counts.items():
            assert n >= 3, f"{kind.value} has only {n} cases (min 3)"

    def test_answer_cases_have_assertions(self, cases):
        for c in cases:
            if c.expected.mode == "answer":
                assert c.expected.contains or c.expected.forbid, c.id

    def test_abstain_cases_valid_reason(self, cases):
        for c in cases:
            if c.expected.mode == "abstain":
                assert c.expected.reason is not None, c.id

    def test_query_strings_non_empty(self, cases):
        for c in cases:
            assert c.query.strip(), c.id


class TestSchemaRejection:
    def test_extra_top_key_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "cases:\n"
            "  - id: x1\n"
            "    kind: empty_memory\n"
            "    query: q\n"
            "    seed_facts: []\n"
            "    expected: {mode: abstain, reason: no_candidates}\n"
            "bogus: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_adversarial_fixture(bad)

    def test_answer_without_assertions_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "cases:\n"
            "  - id: x1\n"
            "    kind: paraphrase_recall\n"
            "    query: q\n"
            "    seed_facts: []\n"
            "    expected: {mode: answer}\n",  # no contains, no forbid
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_adversarial_fixture(bad)

    def test_abstain_unknown_reason_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "cases:\n"
            "  - id: x1\n"
            "    kind: empty_memory\n"
            "    query: q\n"
            "    seed_facts: []\n"
            "    expected: {mode: abstain, reason: not_a_real_reason}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_adversarial_fixture(bad)

    def test_duplicate_ids_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "cases:\n"
            "  - id: dup\n"
            "    kind: empty_memory\n"
            "    query: q\n"
            "    seed_facts: []\n"
            "    expected: {mode: abstain, reason: no_candidates}\n"
            "  - id: dup\n"
            "    kind: empty_memory\n"
            "    query: q2\n"
            "    seed_facts: []\n"
            "    expected: {mode: abstain, reason: no_candidates}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_adversarial_fixture(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_adversarial_fixture(tmp_path / "nope.yaml")
