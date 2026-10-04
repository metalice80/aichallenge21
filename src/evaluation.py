"""Аналитика индексов и контрольная оценка стратегий."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .config import INDEX_ROOT, REPORT_DIR, STRATEGIES
from .embeddings import EmbeddingService
from .search import semantic_search
from .storage import IndexStorage


def index_statistics(strategy: str, index_root: Path = INDEX_ROOT) -> dict[str, Any]:
    """Рассчитать описательные метрики сохранённого индекса."""
    storage = IndexStorage(strategy, index_root)
    chunks = storage.list_chunks()
    if not chunks:
        return {"strategy": strategy, "chunk_count": 0}
    sizes = [int(chunk["token_count"]) for chunk in chunks]
    manifest = storage.load_manifest()
    return {
        "strategy": strategy,
        "document_count": manifest.get("document_count", 0),
        "chunk_count": len(chunks),
        "min_tokens": min(sizes),
        "max_tokens": max(sizes),
        "mean_tokens": statistics.fmean(sizes),
        "median_tokens": statistics.median(sizes),
        "chunking_seconds": float(manifest.get("chunking_seconds", 0.0)),
        "embedding_seconds": float(manifest.get("embedding_seconds", 0.0)),
        "indexing_seconds": float(manifest.get("indexing_seconds", 0.0)),
        "section_share": sum(bool(chunk["section"]) for chunk in chunks) / len(chunks),
    }


def load_evaluation_queries(path: Path) -> list[dict[str, str]]:
    """Загрузить и проверить контрольные запросы JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 10:
        raise ValueError("Нужно минимум 10 контрольных запросов.")
    for item in data:
        if not isinstance(item, dict) or not item.get("query") or not item.get("expected_source"):
            raise ValueError("Каждый запрос должен содержать query и expected_source.")
    return data


def evaluate_strategies(
    queries: list[dict[str, str]],
    *,
    embedding_service: EmbeddingService | None = None,
    index_root: Path = INDEX_ROOT,
) -> dict[str, dict[str, Any]]:
    """Вычислить Hit@1, Hit@3 и время поиска для обоих индексов."""
    if not queries:
        raise ValueError("Список контрольных запросов пуст.")
    service = embedding_service or EmbeddingService()
    service.encode_query(queries[0]["query"])
    output: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        hit_1 = 0
        hit_3 = 0
        started = time.perf_counter()
        for item in queries:
            results = semantic_search(
                item["query"],
                strategy,
                top_k=3,
                embedding_service=service,
                index_root=index_root,
            )
            expected_source = item["expected_source"]
            expected_section = item.get("expected_section")

            def matches(result: dict[str, Any]) -> bool:
                if result["source"] != expected_source:
                    return False
                result_section = result.get("section")
                if expected_section and result_section:
                    return expected_section.lower() in result_section.lower()
                return True

            hit_1 += int(bool(results and matches(results[0])))
            hit_3 += int(any(matches(result) for result in results[:3]))
        elapsed = time.perf_counter() - started
        metrics = index_statistics(strategy, index_root)
        metrics.update(
            {
                "search_seconds": elapsed,
                "mean_search_seconds": elapsed / len(queries),
                "hit_at_1": hit_1 / len(queries),
                "hit_at_3": hit_3 / len(queries),
            }
        )
        output[strategy] = metrics
    return output


def write_comparison_report(
    metrics: dict[str, dict[str, Any]],
    path: Path = REPORT_DIR / "comparison.md",
) -> Path:
    """Сформировать Markdown-отчёт с таблицей и автоматическим выводом."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Сравнение стратегий chunking",
        "",
        "| Метрика | fixed | structural |",
        "|---|---:|---:|",
    ]
    labels = [
        ("Чанки", "chunk_count", ".0f"),
        ("Минимум токенов", "min_tokens", ".0f"),
        ("Максимум токенов", "max_tokens", ".0f"),
        ("Среднее токенов", "mean_tokens", ".2f"),
        ("Медиана токенов", "median_tokens", ".2f"),
        ("Chunking, с", "chunking_seconds", ".3f"),
        ("Embeddings, с", "embedding_seconds", ".3f"),
        ("Поиск, с", "search_seconds", ".3f"),
        ("Hit@1", "hit_at_1", ".2%"),
        ("Hit@3", "hit_at_3", ".2%"),
        ("Чанки с секцией", "section_share", ".2%"),
    ]
    for label, key, spec in labels:
        fixed = metrics.get("fixed", {}).get(key, 0)
        structural = metrics.get("structural", {}).get(key, 0)
        rows.append(f"| {label} | {fixed:{spec}} | {structural:{spec}} |")
    fixed_hit = metrics.get("fixed", {}).get("hit_at_3", 0)
    structural_hit = metrics.get("structural", {}).get("hit_at_3", 0)
    better = "structural" if structural_hit > fixed_hit else "fixed" if fixed_hit > structural_hit else "обе"
    rows.extend(
        [
            "",
            "## Вывод",
            "",
            f"По Hit@3 лучший результат показывает стратегия **{better}**. "
            "Качество зависит от состава документов и контрольных запросов.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
