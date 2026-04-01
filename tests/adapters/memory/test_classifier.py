"""MemoryClassifier unit tests.

Covers rule-based classification for all MemoryType categories,
priority ordering, and fallback to FACT.
"""

from __future__ import annotations

import pytest

from houyi.adapters.memory.classifier import MemoryClassifier
from houyi.adapters.memory.types import MemoryCandidate, MemoryType


@pytest.fixture()
def classifier() -> MemoryClassifier:
    return MemoryClassifier()


def _candidate(content: str) -> MemoryCandidate:
    return MemoryCandidate(content=content)


class TestConstraintClassification:
    async def test_dont_pattern(self, classifier):
        assert await classifier.classify(_candidate("don't use eval")) == MemoryType.CONSTRAINT

    async def test_never_pattern(self, classifier):
        assert (
            await classifier.classify(_candidate("never deploy on Friday")) == MemoryType.CONSTRAINT
        )

    async def test_avoid_pattern(self, classifier):
        assert (
            await classifier.classify(_candidate("avoid global variables")) == MemoryType.CONSTRAINT
        )

    async def test_must_not_pattern(self, classifier):
        assert (
            await classifier.classify(_candidate("must not use print in production"))
            == MemoryType.CONSTRAINT
        )


class TestProfileClassification:
    async def test_my_name_is(self, classifier):
        assert await classifier.classify(_candidate("my name is Alice")) == MemoryType.PROFILE

    async def test_i_work(self, classifier):
        assert await classifier.classify(_candidate("i work at Google")) == MemoryType.PROFILE

    async def test_my_role(self, classifier):
        assert await classifier.classify(_candidate("my role is team lead")) == MemoryType.PROFILE

    async def test_user_name(self, classifier):
        assert await classifier.classify(_candidate("User name: Bob")) == MemoryType.PROFILE


class TestPreferenceClassification:
    async def test_i_prefer(self, classifier):
        assert await classifier.classify(_candidate("i prefer dark mode")) == MemoryType.PREFERENCE

    async def test_i_like(self, classifier):
        assert await classifier.classify(_candidate("i like Python")) == MemoryType.PREFERENCE

    async def test_my_favorite(self, classifier):
        assert (
            await classifier.classify(_candidate("my favorite editor is Vim"))
            == MemoryType.PREFERENCE
        )

    async def test_i_hate(self, classifier):
        assert await classifier.classify(_candidate("i hate verbose code")) == MemoryType.PREFERENCE


class TestProcedureClassification:
    async def test_step_pattern(self, classifier):
        assert await classifier.classify(_candidate("step 1 install deps")) == MemoryType.PROCEDURE

    async def test_how_to(self, classifier):
        assert (
            await classifier.classify(_candidate("how to deploy the service"))
            == MemoryType.PROCEDURE
        )

    async def test_workflow(self, classifier):
        assert (
            await classifier.classify(_candidate("the CI workflow runs nightly"))
            == MemoryType.PROCEDURE
        )


class TestProjectClassification:
    async def test_project_keyword(self, classifier):
        assert await classifier.classify(_candidate("the project uses React")) == MemoryType.PROJECT

    async def test_codebase(self, classifier):
        assert (
            await classifier.classify(_candidate("the codebase is in Python")) == MemoryType.PROJECT
        )

    async def test_tech_stack(self, classifier):
        assert (
            await classifier.classify(_candidate("our tech stack includes Redis"))
            == MemoryType.PROJECT
        )


class TestFactFallback:
    async def test_generic_fact(self, classifier):
        assert (
            await classifier.classify(_candidate("water boils at 100 degrees")) == MemoryType.FACT
        )

    async def test_empty_content(self, classifier):
        assert await classifier.classify(_candidate("")) == MemoryType.FACT

    async def test_numeric_only(self, classifier):
        assert await classifier.classify(_candidate("42")) == MemoryType.FACT


class TestPriorityOrder:
    """CONSTRAINT > PROFILE > PREFERENCE > PROCEDURE > PROJECT > FACT."""

    async def test_constraint_beats_profile(self, classifier):
        result = await classifier.classify(_candidate("don't share my name is Alice"))
        assert result == MemoryType.CONSTRAINT

    async def test_profile_beats_preference(self, classifier):
        result = await classifier.classify(_candidate("my name is Alice and i prefer Python"))
        assert result == MemoryType.PROFILE

    async def test_preference_beats_procedure(self, classifier):
        result = await classifier.classify(_candidate("i prefer this workflow over that one"))
        assert result == MemoryType.PREFERENCE
