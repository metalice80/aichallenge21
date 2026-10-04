"""Семантический поиск по локальным FAISS-индексам."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import INDEX_ROOT
from .embeddings import EmbeddingService
from .storage import IndexConsistencyError, IndexStorage


def semantic_search(
    query: str,
    strategy: str,
    top_k: int = 5,
    source_filter: str | None = None,
    *,
    embedding_service: EmbeddingService | None = None,
    index_root: Path = INDEX_ROOT,
) -> list[dict[str, Any]]:
    """Найти наиболее близкие чанки выбранной стратегии."""
    if not query.strip():
        raise ValueError("Введите непустой поисковый запрос.")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k должен находиться в диапазоне от 1 до 20.")
    storage = IndexStorage(strategy, index_root)
    index = storage.load_index()
    service = embedding_service or EmbeddingService()
    vector = np.ascontiguousarray(service.encode_query(query).reshape(1, -1), dtype=np.float32)
    if vector.shape[1] != index.d:
        raise IndexConsistencyError(
            f"Размерность запроса {vector.shape[1]} не совпадает с индексом {index.d}."
        )
    search_size = index.ntotal if source_filter else min(top_k, index.ntotal)
    if search_size == 0:
        return []
    scores, ids = index.search(vector, search_size)
    valid_pairs = [
        (int(vector_id), float(score))
        for vector_id, score in zip(ids[0], scores[0], strict=True)
        if vector_id >= 0
    ]
    metadata = storage.fetch_by_ids([vector_id for vector_id, _ in valid_pairs])
    if len(metadata) != len(valid_pairs):
        raise IndexConsistencyError("Для части результатов FAISS отсутствуют метаданные SQLite.")

    results: list[dict[str, Any]] = []
    for vector_id, score in valid_pairs:
        row = metadata[vector_id]
        if source_filter and row["source"] != source_filter:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "vector_id": vector_id,
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "title": row["title"],
                "section": row["section"],
                "page_number": row["page_number"],
                "content": row["content"],
                "strategy": row["strategy"],
                "token_count": row["token_count"],
                "start_position": row["start_position"],
                "end_position": row["end_position"],
            }
        )
        if len(results) == top_k:
            break
    return results
