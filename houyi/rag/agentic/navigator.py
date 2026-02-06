"""Directory navigator for Agentic mode."""

from __future__ import annotations

from pathlib import Path


class DirectoryNavigator:
    """Navigate knowledge base directory structure using index files.

    Uses data_structure.md files as hierarchical indexes to understand
    directory contents and select relevant paths for search.

    Reference: rag-skill project's directory navigation approach
    """

    def __init__(
        self,
        knowledge_dir: str,
        index_file: str = "data_structure.md",
    ) -> None:
        """Initialize navigator.

        Args:
            knowledge_dir: Knowledge base root directory
            index_file: Name of directory index file
        """
        self._knowledge_dir = Path(knowledge_dir)
        self._index_file = index_file
        self._index_cache: dict[str, str] = {}

    async def find_candidates(
        self,
        query: str,
        max_depth: int = 3,
    ) -> list[str]:
        """Find candidate file paths relevant to the query.

        Traverses directory structure using index files to identify
        the most relevant paths for searching.

        Args:
            query: User query string
            max_depth: Maximum directory depth to traverse

        Returns:
            List of candidate file paths to search
        """
        if not self._knowledge_dir.exists():
            return []

        candidates: list[str] = []

        # Start from root directory
        await self._traverse_directory(
            directory=self._knowledge_dir,
            query=query,
            depth=0,
            max_depth=max_depth,
            candidates=candidates,
        )

        return candidates

    async def _traverse_directory(
        self,
        directory: Path,
        query: str,
        depth: int,
        max_depth: int,
        candidates: list[str],
    ) -> None:
        """Recursively traverse directory using index files."""
        if depth > max_depth:
            return

        # Try to read index file
        index_path = directory / self._index_file
        index_content = None

        if index_path.exists():
            index_content = await self._read_index(index_path)

        # Get all items in directory
        try:
            items = list(directory.iterdir())
        except PermissionError:
            return

        # Separate directories and files
        subdirs = [d for d in items if d.is_dir() and not d.name.startswith(".")]
        files = [f for f in items if f.is_file() and not f.name.startswith(".")]

        # If we have an index, use it to prioritize
        if index_content:
            relevant_items = self._select_relevant_items(
                query=query,
                index_content=index_content,
                subdirs=subdirs,
                files=files,
            )
        else:
            # Without index, add all non-hidden files and explore subdirs
            relevant_items = {
                "files": files,
                "subdirs": subdirs,
            }

        # Add relevant files to candidates
        for f in relevant_items.get("files", []):
            if self._is_searchable_file(f):
                candidates.append(str(f))

        # Recurse into relevant subdirectories
        for subdir in relevant_items.get("subdirs", []):
            await self._traverse_directory(
                directory=subdir,
                query=query,
                depth=depth + 1,
                max_depth=max_depth,
                candidates=candidates,
            )

    async def _read_index(self, index_path: Path) -> str:
        """Read and cache index file content."""
        path_str = str(index_path)
        if path_str in self._index_cache:
            return self._index_cache[path_str]

        try:
            content = index_path.read_text(encoding="utf-8")
            self._index_cache[path_str] = content
            return content
        except (OSError, UnicodeDecodeError):
            return ""

    def _select_relevant_items(
        self,
        query: str,
        index_content: str,
        subdirs: list[Path],
        files: list[Path],
    ) -> dict[str, list[Path]]:
        """Select relevant items based on index content and query.

        In full implementation, this would use LLM to analyze index.
        For now, uses simple keyword matching.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        index_lower = index_content.lower()

        # Score each subdirectory
        scored_subdirs = []
        for subdir in subdirs:
            name = subdir.name.lower()
            # Check if directory name appears in index with positive context
            score = 0
            if name in index_lower:
                score += 2
            if any(word in name for word in query_words):
                score += 3
            scored_subdirs.append((subdir, score))

        # Score each file
        scored_files = []
        for f in files:
            name = f.name.lower()
            score = 0
            if name in index_lower:
                score += 2
            if any(word in name for word in query_words):
                score += 3
            scored_files.append((f, score))

        # Sort by score and filter
        scored_subdirs.sort(key=lambda x: x[1], reverse=True)
        scored_files.sort(key=lambda x: x[1], reverse=True)

        # Take top items (at least score > 0 or top 3)
        relevant_subdirs = [s[0] for s in scored_subdirs if s[1] > 0 or scored_subdirs.index(s) < 3]
        relevant_files = [f[0] for f in scored_files if f[1] > 0 or scored_files.index(f) < 5]

        return {
            "subdirs": relevant_subdirs,
            "files": relevant_files,
        }

    def _is_searchable_file(self, path: Path) -> bool:
        """Check if file is searchable (text-based)."""
        searchable_extensions = {
            ".md",
            ".txt",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".css",
            ".xml",
            ".csv",
            ".rst",
            ".log",
        }
        return path.suffix.lower() in searchable_extensions
