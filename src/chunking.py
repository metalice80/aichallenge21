"""Фиксированное и структурное разбиение документов."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .models import Chunk, Document
from .text_utils import OffsetTokenizer, RegexTokenizer, stable_id


@dataclass(slots=True)
class _Section:
    name: str | None
    text: str
    start: int
    end: int


def _validate_sizes(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("Размер чанка должен быть больше нуля.")
    if overlap < 0:
        raise ValueError("Overlap не может быть отрицательным.")
    if overlap >= chunk_size:
        raise ValueError("Overlap должен быть меньше размера чанка.")


def _make_chunk(
    document: Document,
    *,
    content: str,
    token_count: int,
    strategy: str,
    section: str | None,
    start: int,
    end: int,
    ordinal: int,
) -> Chunk:
    return Chunk(
        chunk_id=stable_id(
            document.source,
            document.page_number,
            strategy,
            section,
            ordinal,
            start,
            end,
        ),
        document_id=document.document_id,
        source=document.source,
        title=document.title,
        section=section,
        strategy=strategy,
        content=content,
        token_count=token_count,
        page_number=document.page_number,
        start_position=start,
        end_position=end,
    )


def _window_chunks(
    document: Document,
    text: str,
    *,
    absolute_start: int,
    chunk_size: int,
    overlap: int,
    strategy: str,
    section: str | None,
    tokenizer: OffsetTokenizer,
    ordinal_start: int = 0,
) -> list[Chunk]:
    spans = tokenizer.token_spans(text)
    if not spans:
        return []
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    ordinal = ordinal_start
    for token_start in range(0, len(spans), step):
        token_end = min(token_start + chunk_size, len(spans))
        char_start = spans[token_start].start
        char_end = spans[token_end - 1].end
        content = text[char_start:char_end]
        if content.strip():
            chunks.append(
                _make_chunk(
                    document,
                    content=content,
                    token_count=token_end - token_start,
                    strategy=strategy,
                    section=section,
                    start=absolute_start + char_start,
                    end=absolute_start + char_end,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
        if token_end == len(spans):
            break
    return chunks


def chunk_fixed(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 80,
    *,
    tokenizer: OffsetTokenizer | None = None,
) -> list[Chunk]:
    """Разбить документ окнами токенов заданного размера."""
    _validate_sizes(chunk_size, overlap)
    return _window_chunks(
        document,
        document.text,
        absolute_start=0,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy="fixed",
        section=None,
        tokenizer=tokenizer or RegexTokenizer(),
    )


def _markdown_sections(text: str) -> list[_Section]:
    matches = list(re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$", text))
    if not matches:
        return [_Section(None, text, 0, len(text))]
    sections: list[_Section] = []
    if text[: matches[0].start()].strip():
        sections.append(_Section("Введение", text[: matches[0].start()], 0, matches[0].start()))
    hierarchy: list[str] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        hierarchy = hierarchy[: level - 1]
        hierarchy.append(heading)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(_Section(" > ".join(hierarchy), text[match.start():end], match.start(), end))
    return sections


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    offsets.append(len(text))
    return offsets


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    lines = [getattr(node, "lineno", 1)] + [getattr(item, "lineno", 1) for item in decorators]
    return min(lines) - 1


def _python_sections(text: str) -> list[_Section]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _generic_sections(text)
    offsets = _line_offsets(text)
    top_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not top_nodes:
        return [_Section("Модуль", text, 0, len(text))]

    sections: list[_Section] = []
    cursor_line = 0
    for node in top_nodes:
        start_line = _node_start(node)
        end_line = int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))
        if start_line > cursor_line:
            start, end = offsets[cursor_line], offsets[start_line]
            if text[start:end].strip():
                sections.append(_Section("Модуль", text[start:end], start, end))
        if isinstance(node, ast.ClassDef):
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not methods:
                start, end = offsets[start_line], offsets[end_line]
                sections.append(_Section(node.name, text[start:end], start, end))
            else:
                first_method_line = _node_start(methods[0])
                start, end = offsets[start_line], offsets[first_method_line]
                if text[start:end].strip():
                    sections.append(_Section(node.name, text[start:end], start, end))
                for method_index, method in enumerate(methods):
                    method_start_line = _node_start(method)
                    method_end_line = (
                        _node_start(methods[method_index + 1])
                        if method_index + 1 < len(methods)
                        else end_line
                    )
                    start, end = offsets[method_start_line], offsets[method_end_line]
                    sections.append(_Section(f"{node.name}.{method.name}", text[start:end], start, end))
        else:
            start, end = offsets[start_line], offsets[end_line]
            sections.append(_Section(node.name, text[start:end], start, end))
        cursor_line = max(cursor_line, end_line)
    if cursor_line < len(offsets) - 1:
        start, end = offsets[cursor_line], len(text)
        if text[start:end].strip():
            sections.append(_Section("Модуль", text[start:end], start, end))
    return sections


def _looks_like_heading(value: str) -> bool:
    stripped = value.strip()
    if not stripped or len(stripped) > 100 or len(stripped.split()) > 12:
        return False
    return stripped.isupper() or stripped.endswith(":") or stripped.istitle()


def _generic_sections(text: str) -> list[_Section]:
    blocks = list(re.finditer(r"(?s)(?:^|(?<=\n\n))\s*(.+?)(?=\n\n|$)", text))
    if not blocks:
        return [_Section(None, text, 0, len(text))]
    sections: list[_Section] = []
    current_name: str | None = None
    current_start: int | None = None
    current_end = 0
    for block in blocks:
        value = block.group(0)
        first_line = value.strip().splitlines()[0] if value.strip() else ""
        is_heading = len(value.strip().splitlines()) == 1 and _looks_like_heading(first_line)
        if is_heading:
            if current_start is not None:
                sections.append(_Section(current_name, text[current_start:current_end], current_start, current_end))
            current_name = first_line.rstrip(":")
            current_start = block.start()
        elif current_start is None:
            current_start = block.start()
        current_end = block.end()
    if current_start is not None:
        sections.append(_Section(current_name, text[current_start:current_end], current_start, current_end))
    return sections or [_Section(None, text, 0, len(text))]


def _document_sections(document: Document) -> list[_Section]:
    if document.file_type == "md":
        return _markdown_sections(document.text)
    if document.file_type == "py":
        return _python_sections(document.text)
    return _generic_sections(document.text)


def chunk_structural(
    document: Document,
    max_chunk_size: int = 700,
    overlap: int = 80,
    *,
    tokenizer: OffsetTokenizer | None = None,
) -> list[Chunk]:
    """Разбить документ по заголовкам, абзацам или AST-сущностям."""
    _validate_sizes(max_chunk_size, overlap)
    active_tokenizer = tokenizer or RegexTokenizer()
    chunks: list[Chunk] = []
    for section in _document_sections(document):
        section_chunks = _window_chunks(
            document,
            section.text,
            absolute_start=section.start,
            chunk_size=max_chunk_size,
            overlap=overlap,
            strategy="structural",
            section=section.name,
            tokenizer=active_tokenizer,
            ordinal_start=len(chunks),
        )
        chunks.extend(section_chunks)
    return chunks
