"""Типы данных документов, чанков и результатов индексации."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Document:
    """Извлечённый документ или отдельная страница PDF."""

    document_id: str
    source: str
    title: str
    file_type: str
    text: str
    page_number: int | None = None


@dataclass(slots=True)
class Chunk:
    """Текстовый фрагмент вместе с индексируемыми метаданными."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    section: str | None
    strategy: str
    content: str
    token_count: int
    page_number: int | None = None
    start_position: int | None = None
    end_position: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Вернуть сериализуемое представление чанка."""
        return asdict(self)


@dataclass(slots=True)
class IndexingResult:
    """Итог построения одного индекса."""

    strategy: str
    document_count: int
    chunk_count: int
    average_chunk_size: float
    elapsed_seconds: float
    index_path: str
    chunking_seconds: float
    embedding_seconds: float
