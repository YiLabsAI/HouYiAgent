"""Knowledge-base search (RAG query) service.

Responsibilities:
    - Execute RAG queries against a library's index using
      ``houyi.rag.RAG.query()``.
    - Fall back to a simple keyword-grep search when the RAG engine or
      embedding provider is unavailable.

Dependencies:
    - :class:`~.library_repository.LibraryRepository` for library lookup
      and path resolution.
    - :func:`~.embedding_config.resolve_embedding_config` for embedding
      provider resolution.

Thread Safety:
    All methods are stateless beyond the shared repository reference and
    are safe to call concurrently from multiple async tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .embedding_config import resolve_embedding_config
from .library_repository import LibraryRepository

logger = logging.getLogger(__name__)


class SearchService:
    """Executes search queries against knowledge libraries.

    Args:
        repo: The shared :class:`LibraryRepository` instance.
    """

    def __init__(self, repo: LibraryRepository) -> None:
        self._repo = repo

    # ── Public entry point ────────────────────────────────────

    async def search_knowledge(
        self,
        query: str,
        library_id: str | None = None,
        mode: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search the knowledge base.

        Args:
            query: Free-text search query.
            library_id: Restrict search to a single library (optional).
            mode: Override the library's default RAG mode.
            top_k: Maximum number of results to return.

        Returns:
            A result dict with ``query``, ``results``, ``mode_used``,
            ``total_results``, and optionally ``answer``, ``quality``,
            ``error``, etc.
        """
        library = None
        if library_id:
            library = self._repo.get_library(library_id)
            if not library:
                return {
                    "query": query,
                    "library_id": library_id,
                    "results": [],
                    "mode_used": "none",
                    "total_results": 0,
                    "error": f"Library not found: {library_id}",
                }

        library_mode = library.get("mode") if library else None
        logger.info("search_knowledge: mode=%s, library_id=%s", library_mode, library_id)
        effective_mode = mode or library_mode or "agentic"

        if library_id:
            knowledge_dir = str(self._repo.library_upload_dir(library_id))
            index_dir = str(self._repo.library_index_dir(library_id))
        else:
            knowledge_dir = "./knowledge"
            index_dir = None

        try:
            return await self._search_with_rag(
                query,
                library,
                library_id,
                effective_mode,
                knowledge_dir,
                index_dir,
                top_k,
            )
        except ImportError:
            logger.warning("RAG service not available, using fallback search")
            return await self._fallback_search(query, knowledge_dir, top_k)
        except Exception as e:
            logger.error("RAG search failed: %s", e)
            return {
                "query": query,
                "library_id": library_id or "",
                "results": [],
                "mode_used": "error",
                "total_results": 0,
                "error": str(e),
            }

    # ── RAG-backed search ─────────────────────────────────────

    async def _search_with_rag(
        self,
        query: str,
        library: dict[str, Any] | None,
        library_id: str | None,
        effective_mode: str,
        knowledge_dir: str,
        index_dir: str | None,
        top_k: int,
    ) -> dict[str, Any]:
        from houyi.rag import RAG as HouyiRAG
        from houyi.rag.config import GraphConfig, IndexedConfig, RAGConfig
        from houyi.rag.types import RetrievalStrategy

        embedding_config = None
        if effective_mode in ("indexed", "auto"):
            logger.debug("Search mode=%s, checking embedding providers...", effective_mode)

            lib_meta = library.get("metadata", {}) if library else {}
            embedding_config, provider_name = resolve_embedding_config(
                preferred_provider=lib_meta.get("embedding_provider"),
                preferred_model=lib_meta.get("embedding_model"),
                preferred_dimension=lib_meta.get("embedding_dimension"),
            )
            if embedding_config:
                logger.debug(
                    "Embedding for search: %s/%s (dim=%d)",
                    embedding_config.provider,
                    embedding_config.model,
                    embedding_config.dimension,
                )
            else:
                logger.warning("No embedding provider for indexed mode, falling back to agentic")
                effective_mode = "agentic"

            logger.debug(
                "Final effective_mode=%s, embedding_config=%s",
                effective_mode,
                embedding_config,
            )

        indexed_config = None
        graph_config = None
        if library and effective_mode == "indexed":
            lib_metadata = library.get("metadata", {})
            strategies_raw = lib_metadata.get("strategies", ["bm25", "vector"])
            strategies = []
            has_graph = False
            for s in strategies_raw:
                s_lower = s.lower() if isinstance(s, str) else s
                if s_lower == "bm25":
                    strategies.append(RetrievalStrategy.BM25)
                elif s_lower == "vector":
                    strategies.append(RetrievalStrategy.VECTOR)
                elif s_lower == "graph":
                    strategies.append(RetrievalStrategy.GRAPH)
                    has_graph = True
            if strategies:
                indexed_config = IndexedConfig(strategies=strategies)
                logger.debug(
                    "Using strategies from library metadata: %s",
                    [s.value for s in strategies],
                )
            if has_graph:
                graph_config = GraphConfig(enabled=True)
                logger.debug("Graph retrieval enabled")

        config_kwargs: dict[str, Any] = {
            "mode": effective_mode,
            "knowledge_dir": knowledge_dir,
            "index_dir": index_dir,
        }
        if embedding_config is not None:
            config_kwargs["embedding"] = embedding_config
        if indexed_config is not None:
            config_kwargs["indexed"] = indexed_config
        if graph_config is not None:
            config_kwargs["graph"] = graph_config

        config = RAGConfig(**config_kwargs)
        rag_service = HouyiRAG(config)

        result = await rag_service.query(query, top_k=top_k)

        search_results = []
        for sr in result.search_results:
            search_results.append(
                {
                    "chunk_id": sr.chunk_id,
                    "content": sr.content,
                    "score": sr.score,
                    "source": {
                        "file_path": sr.source.file_path if sr.source else "",
                        "location": sr.source.location if sr.source else "",
                        "snippet": sr.source.snippet if sr.source else "",
                    }
                    if sr.source
                    else None,
                    "metadata": sr.metadata,
                }
            )

        quality_data = None
        if result.quality:
            quality_data = {
                "min_score": result.quality.min_score,
                "max_score": result.quality.max_score,
                "avg_score": result.quality.avg_score,
                "above_threshold_count": result.quality.above_threshold_count,
                "total_count": result.quality.total_count,
                "relevance": result.quality.relevance,
                "coverage": result.quality.coverage,
                "confidence_level": result.quality.confidence_level,
                "suggestion": result.quality.suggestion,
                "score_distribution": result.quality.score_distribution,
            }

        return {
            "query": query,
            "library_id": library_id or "",
            "answer": result.answer,
            "results": search_results,
            "mode_used": result.mode_used.value
            if hasattr(result.mode_used, "value")
            else str(result.mode_used),
            "strategies_used": [
                s.value if hasattr(s, "value") else str(s) for s in result.strategies_used
            ],
            "confidence": result.confidence,
            "total_results": len(search_results),
            "metadata": result.metadata,
            "quality": quality_data,
        }

    # ── Fallback search ───────────────────────────────────────

    async def _fallback_search(
        self,
        query: str,
        knowledge_dir: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Simple keyword-grep fallback when the RAG engine is unavailable.

        Args:
            query: Free-text search query.
            knowledge_dir: Directory to scan.
            top_k: Maximum number of results.

        Returns:
            A result dict in the same shape as :meth:`search_knowledge`.
        """
        results: list[dict[str, Any]] = []
        dir_path = Path(knowledge_dir)

        if not dir_path.exists():
            return {
                "query": query,
                "library_id": "",
                "results": [],
                "mode_used": "fallback",
                "total_results": 0,
                "error": f"Directory not found: {knowledge_dir}",
            }

        try:
            keywords = query.lower().split()

            for file_path in dir_path.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in [".md", ".txt", ".json"]:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    content_lower = content.lower()

                    score = sum(1 for kw in keywords if kw in content_lower) / len(keywords)

                    if score > 0:
                        snippet = ""
                        for kw in keywords:
                            idx = content_lower.find(kw)
                            if idx >= 0:
                                start = max(0, idx - 100)
                                end = min(len(content), idx + 200)
                                snippet = content[start:end].strip()
                                break

                        results.append(
                            {
                                "chunk_id": f"chunk_{file_path.stem}_{len(results)}",
                                "content": content[:500] if len(content) > 500 else content,
                                "score": score,
                                "source": {
                                    "file_path": str(file_path),
                                    "location": "",
                                    "snippet": snippet,
                                },
                                "metadata": {"file_name": file_path.name},
                            }
                        )

                except Exception as e:
                    logger.debug("Failed to read file %s: %s", file_path, e)

            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

            return {
                "query": query,
                "library_id": "",
                "results": results,
                "mode_used": "fallback",
                "total_results": len(results),
            }

        except Exception as e:
            logger.error("Fallback search failed: %s", e)
            return {
                "query": query,
                "library_id": "",
                "results": [],
                "mode_used": "error",
                "total_results": 0,
                "error": str(e),
            }
