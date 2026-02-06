"""Document loaders for various file formats."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from houyi.rag.types import Document


async def load_documents(paths: list[str]) -> list[Document]:
    """Load documents from file paths.

    Supports: .txt, .md, .pdf (requires pypdf), .html

    Args:
        paths: List of file paths or directories

    Returns:
        List of Document objects
    """
    documents: list[Document] = []

    for path in paths:
        path_obj = Path(path)

        if path_obj.is_dir():
            # Recursively load all files in directory
            for root, _, files in os.walk(path):
                for f in files:
                    file_path = Path(root) / f
                    doc = await _load_single_file(file_path)
                    if doc:
                        documents.append(doc)
        elif path_obj.is_file():
            doc = await _load_single_file(path_obj)
            if doc:
                documents.append(doc)

    return documents


async def _load_single_file(path: Path) -> Document | None:
    """Load a single file."""
    if not path.exists():
        return None

    suffix = path.suffix.lower()

    try:
        if suffix in {".txt", ".md", ".rst"}:
            return _load_text_file(path)
        elif suffix == ".pdf":
            return _load_pdf_file(path)
        elif suffix in {".html", ".htm"}:
            return _load_html_file(path)
        elif suffix in {".json", ".yaml", ".yml"}:
            return _load_text_file(path)
    except Exception:
        return None

    return None


def _load_text_file(path: Path) -> Document:
    """Load plain text file."""
    content = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        doc_id=str(uuid.uuid4()),
        content=content,
        source=str(path),
        doc_type="text",
        metadata={"filename": path.name},
    )


def _load_pdf_file(path: Path) -> Document | None:
    """Load PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        # pypdf not installed
        return None

    reader = PdfReader(str(path))
    content = ""
    for page in reader.pages:
        content += page.extract_text() + "\n"

    return Document(
        doc_id=str(uuid.uuid4()),
        content=content,
        source=str(path),
        doc_type="pdf",
        metadata={
            "filename": path.name,
            "pages": len(reader.pages),
        },
    )


def _load_html_file(path: Path) -> Document:
    """Load HTML file, extracting text content."""
    content = path.read_text(encoding="utf-8", errors="ignore")

    # Simple HTML tag removal
    import re

    text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return Document(
        doc_id=str(uuid.uuid4()),
        content=text,
        source=str(path),
        doc_type="html",
        metadata={"filename": path.name},
    )
