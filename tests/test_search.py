from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import Chunk
from src.search import semantic_search
from src.storage import IndexNotFoundError, IndexStorage


def build_index(tmp_path: Path, fake_embeddings) -> None:
    contents = ["установка docker compose", "семантический поиск faiss", "обработка pdf страниц"]
    chunks = [
        Chunk(
            chunk_id=f"id-{index}",
            document_id=f"doc-{index}",
            source=f"source-{index}.txt",
            title=f"Source {index}",
            section="Раздел",
            strategy="fixed",
            content=content,
            token_count=3,
        )
        for index, content in enumerate(contents)
    ]
    vectors = fake_embeddings.encode_documents(contents)
    IndexStorage("fixed", tmp_path).save(
        np.asarray(vectors, dtype=np.float32), chunks, {"strategy": "fixed", "chunk_count": 3}
    )


def test_search_returns_no_more_than_top_k(tmp_path: Path, fake_embeddings) -> None:
    build_index(tmp_path, fake_embeddings)
    results = semantic_search(
        "установка docker", "fixed", 2, embedding_service=fake_embeddings, index_root=tmp_path
    )

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["source"] == "source-0.txt"
    assert results[0]["strategy"] == "fixed"


def test_search_source_filter_is_applied_before_limit(tmp_path: Path, fake_embeddings) -> None:
    build_index(tmp_path, fake_embeddings)
    results = semantic_search(
        "установка",
        "fixed",
        1,
        source_filter="source-2.txt",
        embedding_service=fake_embeddings,
        index_root=tmp_path,
    )

    assert len(results) == 1
    assert results[0]["source"] == "source-2.txt"


def test_empty_query_and_missing_index_are_reported(tmp_path: Path, fake_embeddings) -> None:
    with pytest.raises(ValueError, match="непустой"):
        semantic_search(" ", "fixed", embedding_service=fake_embeddings, index_root=tmp_path)
    with pytest.raises(IndexNotFoundError, match="не создан"):
        semantic_search("запрос", "fixed", embedding_service=fake_embeddings, index_root=tmp_path)
