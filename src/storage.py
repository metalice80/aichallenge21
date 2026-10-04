"""Согласованное хранение FAISS-векторов и SQLite-метаданных."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .config import INDEX_ROOT, STRATEGIES
from .models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    vector_id INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    section TEXT,
    strategy TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    page_number INTEGER,
    start_position INTEGER,
    end_position INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);
"""


class IndexNotFoundError(FileNotFoundError):
    """Индекс выбранной стратегии ещё не создан."""


class IndexConsistencyError(RuntimeError):
    """FAISS и SQLite содержат несовместимые записи."""


class IndexStorage:
    """Дисковое хранилище одного стратегического индекса."""

    def __init__(self, strategy: str, index_root: Path = INDEX_ROOT) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Неизвестная стратегия: {strategy}")
        self.strategy = strategy
        self.directory = Path(index_root) / strategy
        self.faiss_path = self.directory / "vectors.faiss"
        self.database_path = self.directory / "metadata.sqlite"
        self.manifest_path = self.directory / "manifest.json"

    @property
    def exists(self) -> bool:
        """Проверить наличие всех трёх файлов индекса."""
        return all(path.is_file() for path in (self.faiss_path, self.database_path, self.manifest_path))

    def save(self, vectors: np.ndarray, chunks: list[Chunk], manifest: dict[str, Any]) -> None:
        """Атомарно записать индекс, метаданные и манифест."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Матрица эмбеддингов должна быть двумерной.")
        if len(chunks) != vectors.shape[0]:
            raise IndexConsistencyError("Количество чанков не совпадает с количеством векторов.")
        if not chunks:
            raise ValueError("Нельзя сохранить пустой индекс.")
        if any(chunk.strategy != self.strategy for chunk in chunks):
            raise IndexConsistencyError("Стратегия чанка не совпадает с каталогом индекса.")

        self.directory.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        temp_faiss = self.directory / f"vectors.{suffix}.tmp"
        temp_database = self.directory / f"metadata.{suffix}.tmp"
        temp_manifest = self.directory / f"manifest.{suffix}.tmp"
        try:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(temp_faiss))

            with sqlite3.connect(temp_database) as connection:
                connection.executescript(_SCHEMA)
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        vector_id, chunk_id, document_id, source, title, section,
                        strategy, content, token_count, page_number,
                        start_position, end_position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            vector_id,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.title,
                            chunk.section,
                            chunk.strategy,
                            chunk.content,
                            chunk.token_count,
                            chunk.page_number,
                            chunk.start_position,
                            chunk.end_position,
                        )
                        for vector_id, chunk in enumerate(chunks)
                    ],
                )

            temp_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temp_faiss, self.faiss_path)
            os.replace(temp_database, self.database_path)
            os.replace(temp_manifest, self.manifest_path)
        finally:
            for temporary in (temp_faiss, temp_database, temp_manifest):
                temporary.unlink(missing_ok=True)
        self.validate()

    def load_index(self) -> faiss.Index:
        """Загрузить FAISS и проверить соответствие SQLite."""
        if not self.exists:
            raise IndexNotFoundError(f"Индекс {self.strategy} ещё не создан.")
        index = faiss.read_index(str(self.faiss_path))
        self.validate(index)
        return index

    def load_manifest(self) -> dict[str, Any]:
        """Прочитать JSON-манифест индекса."""
        if not self.manifest_path.is_file():
            raise IndexNotFoundError(f"Манифест {self.strategy} не найден.")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def validate(self, index: faiss.Index | None = None) -> None:
        """Проверить количество и непрерывность vector_id."""
        if not self.database_path.is_file() or not self.faiss_path.is_file():
            raise IndexNotFoundError(f"Индекс {self.strategy} неполон.")
        active_index = index or faiss.read_index(str(self.faiss_path))
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute("SELECT vector_id FROM chunks ORDER BY vector_id").fetchall()
        ids = [int(row[0]) for row in rows]
        expected = list(range(active_index.ntotal))
        if ids != expected:
            raise IndexConsistencyError(
                f"FAISS содержит {active_index.ntotal} векторов, SQLite — {len(ids)} записей."
            )

    def fetch_by_ids(self, vector_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Получить метаданные по идентификаторам FAISS."""
        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE vector_id IN ({placeholders})", vector_ids
            ).fetchall()
        return {int(row["vector_id"]): dict(row) for row in rows}

    def fetch_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Получить актуальный текст и метаданные чанка по стабильному ID."""
        if not self.database_path.is_file():
            return None
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_chunks(
        self,
        *,
        source: str | None = None,
        section: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Вывести чанки с безопасными фильтрами."""
        if not self.database_path.is_file():
            return []
        clauses: list[str] = []
        parameters: list[Any] = []
        for expression, value in (
            ("source = ?", source),
            ("section = ?", section),
            ("token_count >= ?", min_tokens),
            ("token_count <= ?", max_tokens),
        ):
            if value is not None:
                clauses.append(expression)
                parameters.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM chunks" + where + " ORDER BY vector_id", parameters
            ).fetchall()
        return [dict(row) for row in rows]

    def sources(self) -> list[str]:
        """Вернуть источники индекса в алфавитном порядке."""
        if not self.database_path.is_file():
            return []
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute("SELECT DISTINCT source FROM chunks ORDER BY source").fetchall()
        return [str(row[0]) for row in rows]
