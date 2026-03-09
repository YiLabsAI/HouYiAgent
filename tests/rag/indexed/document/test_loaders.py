from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from houyi.rag.indexed.document import loaders as loaders_module
from houyi.rag.indexed.document.loaders import (
    SUPPORTED_DOCUMENT_SUFFIXES,
    _load_html_file,
    _load_single_file,
    _load_text_file,
    load_documents,
)
from houyi.rag.types import Document


class TestDocumentLoaders:
    @staticmethod
    def _expected_loader_for_suffix(suffix: str) -> str:
        if suffix in {".txt", ".md", ".rst", ".json", ".yaml", ".yml"}:
            return "_load_text_file"
        if suffix == ".pdf":
            return "_load_pdf_file"
        if suffix in {".html", ".htm"}:
            return "_load_html_file"
        if suffix == ".csv":
            return "_load_csv_file"
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            return "_load_excel_file"
        if suffix == ".doc":
            return "_load_doc_file"
        if suffix == ".docx":
            return "_load_docx_file"
        if suffix == ".pptx":
            return "_load_pptx_file"
        if suffix == ".epub":
            return "_load_epub_file"
        raise AssertionError(f"Unhandled supported suffix: {suffix}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("suffix", SUPPORTED_DOCUMENT_SUFFIXES)
    async def test_supported_suffix_dispatches_to_expected_loader(
        self,
        tmp_path: Path,
        suffix: str,
    ) -> None:
        loader_name = self._expected_loader_for_suffix(suffix)
        path = tmp_path / f"sample{suffix}"
        path.write_text("placeholder", encoding="utf-8")

        sentinel = Document(doc_id="d1", content="ok", source=str(path), doc_type="sentinel")
        with patch(
            f"houyi.rag.indexed.document.loaders.{loader_name}",
            return_value=sentinel,
        ) as mock_loader:
            doc = await _load_single_file(path)

        assert doc is sentinel
        mock_loader.assert_called_once_with(path)

    @pytest.mark.asyncio
    async def test_load_text_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            f.flush()
            path = Path(f.name)

        try:
            doc = _load_text_file(path)
            assert doc.content == "Hello, World!"
            assert doc.doc_type == "text"
            assert doc.source == str(path)
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_markdown_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nContent here")
            f.flush()
            path = Path(f.name)

        try:
            doc = await _load_single_file(path)
            assert doc is not None
            assert "# Title" in doc.content
            assert doc.doc_type == "text"
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_html_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><body><h1>Title</h1><p>Content</p></body></html>")
            f.flush()
            path = Path(f.name)

        try:
            doc = _load_html_file(path)
            assert "Title" in doc.content
            assert "Content" in doc.content
            assert "<h1>" not in doc.content
            assert doc.doc_type == "html"
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_html_with_script(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><script>alert('bad')</script><body>Safe content</body></html>")
            f.flush()
            path = Path(f.name)

        try:
            doc = _load_html_file(path)
            assert "alert" not in doc.content
            assert "Safe content" in doc.content
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self) -> None:
        doc = await _load_single_file(Path("/nonexistent/file.txt"))
        assert doc is None

    @pytest.mark.asyncio
    async def test_load_documents_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc1.txt").write_text("Document 1")
            (Path(tmpdir) / "doc2.md").write_text("# Document 2")

            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "doc3.txt").write_text("Document 3")

            docs = await load_documents([tmpdir])

            assert len(docs) == 3
            contents = [d.content for d in docs]
            assert any("Document 1" in c for c in contents)
            assert any("# Document 2" in c for c in contents)
            assert any("Document 3" in c for c in contents)

    @pytest.mark.asyncio
    async def test_load_documents_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file2 = Path(tmpdir) / "doc2.txt"
            file1.write_text("Content 1")
            file2.write_text("Content 2")

            docs = await load_documents([str(file1), str(file2)])

            assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_load_unsupported_file_type(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Unknown format")
            f.flush()
            path = Path(f.name)

        try:
            doc = await _load_single_file(path)
            assert doc is None
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_csv_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,role\nAlice,Engineer\nBob,Designer\n")
            f.flush()
            path = Path(f.name)

        try:
            doc = await _load_single_file(path)
            assert doc is not None
            assert doc.doc_type == "csv"
            assert "Columns: name, role" in doc.content
            assert "Row 1 - name: Alice" in doc.content
            assert doc.metadata["rows"] == 2
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_excel_file(self, tmp_path: Path) -> None:
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        path = tmp_path / "table.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(
                [
                    {"name": "Alice", "role": "Engineer"},
                    {"name": "Bob", "role": "Designer"},
                ]
            ).to_excel(writer, index=False, sheet_name="People")

        doc = await _load_single_file(path)
        assert doc is not None
        assert doc.doc_type == "excel"
        assert "Sheet: People" in doc.content
        assert "Row 1 - name: Alice" in doc.content
        assert doc.metadata["sheets"] == 1

    @pytest.mark.asyncio
    async def test_load_excel_without_pandas_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "table.xlsx"
        path.write_bytes(b"not-a-real-excel-file")

        with patch("builtins.__import__") as mock_import:
            original_import = __import__

            def _fake_import(name, *args, **kwargs):
                if name == "pandas":
                    raise ImportError("pandas missing")
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = _fake_import
            doc = await _load_single_file(path)

        assert doc is None

    @pytest.mark.asyncio
    async def test_load_doc_without_antiword_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.doc"
        path.write_bytes(b"dummy-doc")

        with patch.object(loaders_module.shutil, "which", return_value=None):
            doc = await _load_single_file(path)

        assert doc is None

    @pytest.mark.asyncio
    async def test_load_docx_file(self, tmp_path: Path) -> None:
        docx = pytest.importorskip("docx")

        path = tmp_path / "sample.docx"
        file_doc = docx.Document()
        file_doc.add_paragraph("Project Alpha status")
        table = file_doc.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Owner"
        table.rows[0].cells[1].text = "Von"
        file_doc.save(path)

        doc = await _load_single_file(path)
        assert doc is not None
        assert doc.doc_type == "docx"
        assert "Project Alpha status" in doc.content
        assert "Owner | Von" in doc.content

    @pytest.mark.asyncio
    async def test_load_pptx_file(self, tmp_path: Path) -> None:
        pptx = pytest.importorskip("pptx")

        path = tmp_path / "deck.pptx"
        prs = pptx.Presentation()
        slide_layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = "Quarterly Review"
        slide.placeholders[1].text = "Revenue up 20%"
        prs.save(path)

        doc = await _load_single_file(path)
        assert doc is not None
        assert doc.doc_type == "pptx"
        assert "Slide 1" in doc.content
        assert "Quarterly Review" in doc.content

    @pytest.mark.asyncio
    async def test_load_epub_file(self, tmp_path: Path) -> None:
        pytest.importorskip("bs4")
        epub = pytest.importorskip("ebooklib.epub")

        path = tmp_path / "book.epub"
        book = epub.EpubBook()
        book.set_identifier("book-001")
        book.set_title("Sample EPUB")
        book.set_language("en")

        chapter = epub.EpubHtml(
            title="Intro",
            file_name="intro.xhtml",
            content="<h1>Intro</h1><p>Hello EPUB world</p>",
        )
        book.add_item(chapter)
        book.toc = (epub.Link("intro.xhtml", "Intro", "intro"),)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", chapter]
        epub.write_epub(str(path), book)

        doc = await _load_single_file(path)
        assert doc is not None
        assert doc.doc_type == "epub"
        assert "Hello EPUB world" in doc.content
        assert doc.metadata["title"] == "Sample EPUB"
