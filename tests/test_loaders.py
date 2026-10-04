from __future__ import annotations

from pathlib import Path

import pytest

from src.loaders import DocumentLoadError, UnsupportedFormatError, load_document


def test_load_txt_and_markdown(tmp_path: Path) -> None:
    text_path = tmp_path / "note.txt"
    text_path.write_text("Первая строка.\r\n\r\nВторая строка.\n", encoding="utf-8")
    markdown_path = tmp_path / "manual.md"
    markdown_path.write_text("# Руководство\n\n```python\nprint('ok')\n```", encoding="utf-8")

    text_document = load_document(text_path)[0]
    markdown_document = load_document(markdown_path)[0]

    assert text_document.file_type == "txt"
    assert "\r" not in text_document.text
    assert markdown_document.title == "Руководство"
    assert "print('ok')" in markdown_document.text


def test_pdf_pages_are_extracted_separately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "pages.pdf"
    pdf_path.write_bytes(b"not-empty")

    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, path: Path) -> None:
            self.pages = [Page("Страница один"), Page(""), Page("Страница три")]

    monkeypatch.setattr("src.loaders.PdfReader", Reader)
    documents = load_document(pdf_path)

    assert [document.page_number for document in documents] == [1, 3]
    assert len({document.document_id for document in documents}) == 1


def test_empty_and_unsupported_files_report_clear_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="пуст"):
        load_document(empty)
    with pytest.raises(UnsupportedFormatError, match="не поддерживается"):
        load_document(unsupported)
