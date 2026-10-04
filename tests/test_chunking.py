from __future__ import annotations

from src.chunking import chunk_fixed, chunk_structural
from src.models import Document
from src.text_utils import RegexTokenizer


def document(text: str, file_type: str = "txt") -> Document:
    return Document("doc-1", f"sample.{file_type}", "Sample", file_type, text)


def tokens(text: str) -> list[str]:
    spans = RegexTokenizer().token_spans(text)
    return [text[span.start : span.end] for span in spans]


def test_fixed_chunking_respects_size_and_keeps_last_fragment() -> None:
    chunks = chunk_fixed(document(" ".join(f"word{i}" for i in range(23))), 10, 2)

    assert len(chunks) == 3
    assert all(0 < chunk.token_count <= 10 for chunk in chunks)
    assert "word22" in chunks[-1].content
    assert all(chunk.strategy == "fixed" for chunk in chunks)


def test_fixed_chunking_contains_real_overlap() -> None:
    chunks = chunk_fixed(document(" ".join(f"w{i}" for i in range(16))), 8, 3)

    assert tokens(chunks[0].content)[-3:] == tokens(chunks[1].content)[:3]
    assert chunks[0].start_position == 0
    assert chunks[1].start_position is not None


def test_markdown_is_split_by_heading_path() -> None:
    markdown = "# Установка\nТекст.\n\n## Docker\nКоманда.\n\n# Поиск\nЗапрос."
    chunks = chunk_structural(document(markdown, "md"), 100, 10)

    assert [chunk.section for chunk in chunks] == ["Установка", "Установка > Docker", "Поиск"]
    assert chunks[1].content.startswith("## Docker")


def test_python_is_split_by_functions_classes_and_methods() -> None:
    source = '''"""Модуль."""\nVALUE = 1\n\ndef first():\n    return VALUE\n\nasync def later():\n    return 2\n\nclass Worker:\n    kind = "demo"\n\n    def run(self):\n        return self.kind\n\n    async def stop(self):\n        return None\n'''
    chunks = chunk_structural(document(source, "py"), 100, 10)
    sections = {chunk.section for chunk in chunks}

    assert {"Модуль", "first", "later", "Worker", "Worker.run", "Worker.stop"} <= sections


def test_large_structural_section_is_split_and_preserves_section() -> None:
    markdown = "# Большой раздел\n" + " ".join(f"слово{i}" for i in range(31))
    chunks = chunk_structural(document(markdown, "md"), 10, 2)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert all(chunk.section == "Большой раздел" for chunk in chunks)
    assert all(chunk.strategy == "structural" for chunk in chunks)


def test_invalid_overlap_is_rejected() -> None:
    try:
        chunk_fixed(document("текст"), 10, 10)
    except ValueError as exc:
        assert "Overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
