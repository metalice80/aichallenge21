from __future__ import annotations

from pathlib import Path

from src.pipeline import build_indexes
from src.search import semantic_search
from src.storage import IndexStorage


def test_pipeline_builds_comparable_persistent_indexes(tmp_path: Path, fake_embeddings) -> None:
    documents = tmp_path / "documents"
    indexes = tmp_path / "indexes"
    documents.mkdir()
    markdown = documents / "manual.md"
    markdown.write_text(
        "# Установка\nИспользуйте виртуальное окружение.\n\n"
        "# Поиск\nСемантический поиск использует векторы.",
        encoding="utf-8",
    )
    text = documents / "notes.txt"
    text.write_text("ХРАНЕНИЕ\n\nFAISS и SQLite работают совместно.", encoding="utf-8")
    stages: list[str] = []

    results = build_indexes(
        [markdown, text],
        embedding_service=fake_embeddings,
        index_root=indexes,
        fixed_size=12,
        structural_size=12,
        overlap=2,
        progress_callback=lambda stage, fraction, message: stages.append(stage),
    )

    assert set(results) == {"fixed", "structural"}
    assert "Готово" in stages
    for strategy in ("fixed", "structural"):
        storage = IndexStorage(strategy, indexes)
        assert storage.exists
        storage.validate()
        assert storage.load_index().ntotal == len(storage.list_chunks())
        found = semantic_search(
            "виртуальное окружение",
            strategy,
            top_k=2,
            embedding_service=fake_embeddings,
            index_root=indexes,
        )
        assert found
        assert all(row["strategy"] == strategy for row in found)
