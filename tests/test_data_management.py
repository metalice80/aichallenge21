from __future__ import annotations

from pathlib import Path

import pytest

from src.data_management import delete_indexes


def create_index_files(root: Path, strategy: str) -> None:
    directory = root / strategy
    directory.mkdir(parents=True)
    for name in ("vectors.faiss", "metadata.sqlite", "manifest.json"):
        (directory / name).write_text("data", encoding="utf-8")
    (directory / ".gitkeep").write_text("", encoding="utf-8")


def test_delete_indexes_only_removes_allowlisted_files(tmp_path: Path) -> None:
    create_index_files(tmp_path, "fixed")
    create_index_files(tmp_path, "structural")
    document = tmp_path / "important.txt"
    document.write_text("keep", encoding="utf-8")

    deleted = delete_indexes("fixed", tmp_path)

    assert len(deleted) == 3
    assert not (tmp_path / "fixed" / "vectors.faiss").exists()
    assert (tmp_path / "fixed" / ".gitkeep").exists()
    assert (tmp_path / "structural" / "vectors.faiss").exists()
    assert document.read_text(encoding="utf-8") == "keep"


def test_delete_indexes_rejects_arbitrary_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed"):
        delete_indexes("../../documents", tmp_path)
