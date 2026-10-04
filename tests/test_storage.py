from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import Chunk
from src.storage import IndexConsistencyError, IndexStorage


def make_chunks(count: int, strategy: str = "fixed") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"chunk-{index}",
            document_id="doc-1",
            source="sample.txt",
            title="Sample",
            section=None,
            strategy=strategy,
            content=f"content {index}",
            token_count=2,
            start_position=index * 10,
            end_position=index * 10 + 9,
        )
        for index in range(count)
    ]


def test_metadata_round_trip_and_vector_ids_match(tmp_path: Path) -> None:
    storage = IndexStorage("fixed", tmp_path)
    vectors = np.eye(3, dtype=np.float32)
    storage.save(vectors, make_chunks(3), {"strategy": "fixed", "chunk_count": 3})

    index = storage.load_index()
    rows = storage.list_chunks()

    assert index.ntotal == len(rows) == 3
    assert [row["vector_id"] for row in rows] == [0, 1, 2]
    assert rows[1]["chunk_id"] == "chunk-1"
    assert storage.load_manifest()["strategy"] == "fixed"


def test_storage_rejects_mismatched_vector_count(tmp_path: Path) -> None:
    storage = IndexStorage("fixed", tmp_path)
    with pytest.raises(IndexConsistencyError, match="не совпадает"):
        storage.save(np.eye(2, dtype=np.float32), make_chunks(1), {})
