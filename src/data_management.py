"""Безопасные операции управления локальными индексами."""

from __future__ import annotations

from pathlib import Path

from .config import INDEX_ROOT

_INDEX_FILES = ("vectors.faiss", "metadata.sqlite", "manifest.json")
_SELECTIONS: dict[str, tuple[str, ...]] = {
    "fixed": ("fixed",),
    "structural": ("structural",),
    "both": ("fixed", "structural"),
}


def delete_indexes(selection: str, index_root: Path = INDEX_ROOT) -> list[Path]:
    """Удалить только заранее определённые файлы выбранных индексов."""
    strategies = _SELECTIONS.get(selection)
    if strategies is None:
        raise ValueError("Можно удалить fixed, structural или оба индекса.")
    root = Path(index_root)
    deleted: list[Path] = []
    for strategy in strategies:
        for filename in _INDEX_FILES:
            target = root / strategy / filename
            if target.is_file():
                target.unlink()
                deleted.append(target)
    return deleted
