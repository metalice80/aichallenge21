"""Загрузка PDF, Markdown, TXT и Python-файлов."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import SUPPORTED_EXTENSIONS
from .models import Document
from .text_utils import content_hash, normalize_text, portable_source, stable_id

logger = logging.getLogger(__name__)


class DocumentLoadError(ValueError):
    """Понятная пользователю ошибка чтения документа."""


class UnsupportedFormatError(DocumentLoadError):
    """Формат файла не поддерживается."""


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Файл {path.name} не является корректным UTF-8: {exc}"
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(f"Не удалось прочитать {path.name}: {exc}") from exc


def _document_id(path: Path) -> str:
    try:
        digest = content_hash(path.read_bytes())
    except OSError as exc:
        raise DocumentLoadError(f"Не удалось прочитать {path.name}: {exc}") from exc
    return stable_id(portable_source(path), digest)


def _markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def load_document(path: Path) -> list[Document]:
    """Извлечь один или несколько объектов Document из файла."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Формат {suffix or 'без расширения'} не поддерживается. "
            "Допустимы PDF, Markdown, TXT и Python."
        )
    if not path.is_file():
        raise DocumentLoadError(f"Файл не найден: {path.name}")
    if path.stat().st_size == 0:
        raise DocumentLoadError(f"Файл {path.name} пуст.")

    source = portable_source(path)
    document_id = _document_id(path)
    title = path.stem

    if suffix == ".pdf":
        try:
            reader = PdfReader(path)
            documents: list[Document] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = normalize_text(page.extract_text() or "")
                if not text.strip():
                    logger.warning("Пропущена пустая страница %s файла %s", page_number, source)
                    continue
                documents.append(
                    Document(
                        document_id=document_id,
                        source=source,
                        title=title,
                        file_type="pdf",
                        text=text,
                        page_number=page_number,
                    )
                )
        except (PdfReadError, OSError, ValueError) as exc:
            logger.exception("Ошибка чтения PDF %s", source)
            raise DocumentLoadError(f"Не удалось извлечь текст из PDF {source}: {exc}") from exc
        if not documents:
            raise DocumentLoadError(
                f"PDF {source} не содержит извлекаемого текста. OCR не поддерживается."
            )
        return documents

    raw_text = _read_utf8(path)
    preserve_source = suffix == ".py"
    text = normalize_text(raw_text, preserve_source=preserve_source)
    if not text.strip():
        raise DocumentLoadError(f"Файл {source} не содержит текста.")
    if suffix == ".md":
        title = _markdown_title(text, title)
    return [
        Document(
            document_id=document_id,
            source=source,
            title=title,
            file_type=suffix.lstrip("."),
            text=text,
        )
    ]


def load_documents(paths: list[Path]) -> list[Document]:
    """Загрузить набор файлов, сохранив порядок входных путей."""
    documents: list[Document] = []
    for path in paths:
        documents.extend(load_document(path))
    return documents
