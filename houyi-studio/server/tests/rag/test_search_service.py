"""Tests for SearchService — query dispatch and fallback keyword search."""

from __future__ import annotations

from pathlib import Path

import pytest
from houyi_studio.server.rag.library_repository import LibraryRepository
from houyi_studio.server.rag.search_service import SearchService


@pytest.fixture
def repo(tmp_path: Path) -> LibraryRepository:
    """Isolated repository for search tests."""
    return LibraryRepository(storage_dir=tmp_path)


@pytest.fixture
def svc(repo: LibraryRepository) -> SearchService:
    """SearchService wired to the test repository."""
    return SearchService(repo)


@pytest.fixture
def library_id(repo: LibraryRepository) -> str:
    """A library pre-created in the repository."""
    lib = repo.create_library(name="SearchLib", knowledge_dir="/nonexistent")
    return lib["library_id"]


# ── search_knowledge edge cases ──────────────────────────────────


class TestSearchKnowledge:
    """High-level search_knowledge behaviour (RAG engine not required)."""

    @pytest.mark.asyncio
    async def test_nonexistent_library_returns_error(self, svc: SearchService):
        result = await svc.search_knowledge("hello", library_id="lib_nope")
        assert result["total_results"] == 0
        assert "error" in result
        assert result["mode_used"] == "none"

    @pytest.mark.asyncio
    async def test_nonexistent_library_preserves_query(self, svc: SearchService):
        result = await svc.search_knowledge("my query", library_id="lib_nope")
        assert result["query"] == "my query"
        assert result["library_id"] == "lib_nope"


# ── _fallback_search ─────────────────────────────────────────────


class TestFallbackSearch:
    """The keyword-grep fallback used when the RAG engine is unavailable."""

    @pytest.mark.asyncio
    async def test_matching_files(self, svc: SearchService, tmp_path: Path):
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / "notes.md").write_text("Python is a programming language")
        (kdir / "other.txt").write_text("Java is also a language")

        result = await svc._fallback_search("python programming", str(kdir), top_k=5)

        assert result["mode_used"] == "fallback"
        assert result["total_results"] >= 1
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_directory(self, svc: SearchService, tmp_path: Path):
        kdir = tmp_path / "empty"
        kdir.mkdir()

        result = await svc._fallback_search("anything", str(kdir), top_k=5)

        assert result["mode_used"] == "fallback"
        assert result["total_results"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, svc: SearchService, tmp_path: Path):
        result = await svc._fallback_search("query", str(tmp_path / "nope"), top_k=5)

        assert result["total_results"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, svc: SearchService, tmp_path: Path):
        kdir = tmp_path / "many"
        kdir.mkdir()
        for i in range(10):
            (kdir / f"file{i}.md").write_text(f"keyword content {i}")

        result = await svc._fallback_search("keyword", str(kdir), top_k=3)

        assert result["total_results"] <= 3

    @pytest.mark.asyncio
    async def test_skips_unsupported_extensions(self, svc: SearchService, tmp_path: Path):
        kdir = tmp_path / "mixed"
        kdir.mkdir()
        (kdir / "good.md").write_text("findme content")
        (kdir / "bad.py").write_text("findme content")
        (kdir / "bad.csv").write_text("findme content")

        result = await svc._fallback_search("findme", str(kdir), top_k=10)

        sources = [r["source"]["file_path"] for r in result["results"]]
        assert any("good.md" in s for s in sources)
        assert not any(".py" in s for s in sources)

    @pytest.mark.asyncio
    async def test_result_contains_snippet(self, svc: SearchService, tmp_path: Path):
        kdir = tmp_path / "snip"
        kdir.mkdir()
        (kdir / "doc.txt").write_text("The quick brown fox jumps over the lazy dog")

        result = await svc._fallback_search("fox", str(kdir), top_k=5)

        assert result["total_results"] == 1
        assert result["results"][0]["source"]["snippet"] != ""
