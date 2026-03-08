"""Agentic searcher using grep and read_file operations."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from houyi.rag.types import SearchResult, Source


class AgenticSearcher:
    """Search files using local grep and direct file reads."""

    def __init__(
        self,
        knowledge_dir: str,
        chunk_limit: int = 500,
    ) -> None:
        """Initialize searcher.

        Args:
            knowledge_dir: Knowledge base root directory
            chunk_limit: Max lines to read per chunk
        """
        self._knowledge_dir = Path(knowledge_dir)
        self._chunk_limit = chunk_limit

    async def search_files(
        self,
        paths: list[str],
        keywords: list[str],
        exclude_files: set[str] | None = None,
    ) -> list[SearchResult]:
        """Search files for keywords.

        Args:
            paths: File paths to search
            keywords: Keywords to search for
            exclude_files: Files already searched (to skip)

        Returns:
            List of search results
        """
        if not paths or not keywords:
            return []

        exclude = exclude_files or set()
        results: list[SearchResult] = []

        for path in paths:
            if path in exclude:
                continue

            file_results = await self._search_file(path, keywords)
            results.extend(file_results)

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def _search_file(
        self,
        path: str,
        keywords: list[str],
    ) -> list[SearchResult]:
        """Search a single file for keywords."""
        file_path = Path(path)
        if not file_path.exists():
            return []

        # Use grep to find matching lines
        matches = await self._grep_file(path, keywords)
        if not matches:
            return []

        # Read context around matches
        results = []
        for match in matches:
            line_num = match["line_num"]
            content = await self._read_context(path, line_num)

            if content:
                results.append(
                    SearchResult(
                        content=content,
                        score=match["score"],
                        source=Source(
                            file_path=path,
                            location=f"line {line_num}",
                            snippet=match["line"][:200],
                            score=match["score"],
                        ),
                    )
                )

        return results

    async def _grep_file(
        self,
        path: str,
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        """Search file for keywords using grep."""
        matches: list[dict[str, Any]] = []

        # Build grep pattern
        pattern = "|".join(re.escape(kw) for kw in keywords)

        try:
            result = subprocess.run(
                ["grep", "-n", "-i", "-E", pattern, path],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                # Parse grep output: line_num:content
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        line_num = int(parts[0])
                        line_content = parts[1]

                        # Score based on keyword density
                        score = sum(
                            1 for kw in keywords if kw.lower() in line_content.lower()
                        ) / len(keywords)

                        matches.append(
                            {
                                "line_num": line_num,
                                "line": line_content,
                                "score": score,
                            }
                        )
                    except ValueError:
                        continue

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # Fallback to Python-based search
            matches = await self._python_grep(path, keywords)

        return matches

    async def _python_grep(
        self,
        path: str,
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        """Python-based grep fallback."""
        matches: list[dict[str, Any]] = []

        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line_lower = line.lower()
                    matching_kws = [kw for kw in keywords if kw.lower() in line_lower]

                    if matching_kws:
                        score = len(matching_kws) / len(keywords)
                        matches.append(
                            {
                                "line_num": line_num,
                                "line": line.strip(),
                                "score": score,
                            }
                        )
        except (OSError, UnicodeDecodeError):
            pass

        return matches

    async def _read_context(
        self,
        path: str,
        center_line: int,
        context_lines: int = 20,
    ) -> str:
        """Read file content around a specific line."""
        start_line = max(1, center_line - context_lines)
        end_line = center_line + context_lines

        try:
            lines = []
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if start_line <= line_num <= end_line:
                        lines.append(line)
                    if line_num > end_line:
                        break

            return "".join(lines)
        except (OSError, UnicodeDecodeError):
            return ""
