"""Оркестрация загрузки, chunking, embeddings и сохранения."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_fixed, chunk_structural
from .config import INDEX_ROOT, MODEL_NAME, STRATEGIES
from .embeddings import EmbeddingService
from .loaders import load_documents
from .models import Chunk, Document, IndexingResult
from .storage import IndexStorage

ProgressCallback = Callable[[str, float, str], None]


def _notify(callback: ProgressCallback | None, stage: str, fraction: float, message: str) -> None:
    if callback:
        callback(stage, fraction, message)


def _chunks_for_strategy(
    documents: list[Document],
    strategy: str,
    *,
    fixed_size: int,
    overlap: int,
    structural_size: int,
    tokenizer: object,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        if strategy == "fixed":
            chunks.extend(
                chunk_fixed(document, fixed_size, overlap, tokenizer=tokenizer)  # type: ignore[arg-type]
            )
        else:
            chunks.extend(
                chunk_structural(document, structural_size, overlap, tokenizer=tokenizer)  # type: ignore[arg-type]
            )
    return chunks


def build_indexes(
    paths: Iterable[Path],
    strategies: Iterable[str] = STRATEGIES,
    *,
    fixed_size: int = 500,
    overlap: int = 80,
    structural_size: int = 700,
    batch_size: int = 32,
    embedding_service: EmbeddingService | None = None,
    index_root: Path = INDEX_ROOT,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, IndexingResult]:
    """Построить один или два полностью автономных индекса."""
    selected = list(dict.fromkeys(strategies))
    if not selected or any(strategy not in STRATEGIES for strategy in selected):
        raise ValueError("Нужно выбрать fixed, structural или обе стратегии.")
    source_paths = [Path(path) for path in paths]
    if not source_paths:
        raise ValueError("Нет документов для индексации.")

    started = time.perf_counter()
    _notify(progress_callback, "Загрузка", 0.03, "Чтение файлов")
    documents = load_documents(source_paths)
    if not documents:
        raise ValueError("Документы не содержат извлекаемого текста.")
    _notify(progress_callback, "Извлечение текста", 0.12, f"Извлечено документов: {len(documents)}")

    service = embedding_service or EmbeddingService(batch_size=batch_size)
    tokenizer = service.tokenizer
    results: dict[str, IndexingResult] = {}
    for strategy_index, strategy in enumerate(selected):
        base = 0.15 + strategy_index * (0.8 / len(selected))
        span = 0.8 / len(selected)
        chunk_started = time.perf_counter()
        _notify(progress_callback, "Chunking", base, f"Стратегия {strategy}")
        chunks = _chunks_for_strategy(
            documents,
            strategy,
            fixed_size=fixed_size,
            overlap=overlap,
            structural_size=structural_size,
            tokenizer=tokenizer,
        )
        chunking_seconds = time.perf_counter() - chunk_started
        if not chunks:
            raise ValueError(f"Стратегия {strategy} не создала ни одного чанка.")

        embedding_started = time.perf_counter()
        _notify(progress_callback, "Embeddings", base + span * 0.2, f"Эмбеддинги: {len(chunks)}")
        vectors = service.encode_documents(
            [chunk.content for chunk in chunks],
            progress_callback=lambda value, b=base, s=span: _notify(
                progress_callback,
                "Embeddings",
                b + s * (0.2 + value * 0.55),
                f"Готово {value:.0%}",
            ),
        )
        embedding_seconds = time.perf_counter() - embedding_started
        strategy_elapsed = chunking_seconds + embedding_seconds
        manifest = {
            "strategy": strategy,
            "model": service.model_name,
            "dimension": int(vectors.shape[1]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "document_count": len({document.document_id for document in documents}),
            "document_parts": len(documents),
            "chunk_count": len(chunks),
            "parameters": {
                "fixed_chunk_size": fixed_size,
                "structural_max_chunk_size": structural_size,
                "overlap": overlap,
                "batch_size": batch_size,
            },
            "chunking_seconds": chunking_seconds,
            "embedding_seconds": embedding_seconds,
            "indexing_seconds": strategy_elapsed,
        }
        _notify(progress_callback, "Сохранение", base + span * 0.85, f"Сохранение {strategy}")
        storage = IndexStorage(strategy, index_root)
        storage.save(vectors, chunks, manifest)
        results[strategy] = IndexingResult(
            strategy=strategy,
            document_count=manifest["document_count"],
            chunk_count=len(chunks),
            average_chunk_size=sum(chunk.token_count for chunk in chunks) / len(chunks),
            elapsed_seconds=strategy_elapsed,
            index_path=str(storage.directory),
            chunking_seconds=chunking_seconds,
            embedding_seconds=embedding_seconds,
        )
    _notify(progress_callback, "Готово", 1.0, f"Общее время: {time.perf_counter() - started:.2f} с")
    return results
