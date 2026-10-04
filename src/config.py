"""Конфигурация локального индексатора документов."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
INDEX_ROOT = PROJECT_ROOT / "indexes"
REPORT_DIR = PROJECT_ROOT / "reports"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
LOG_FILE = PROJECT_ROOT / "data" / "document_indexer.log"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".py"}
STRATEGIES = ("fixed", "structural")


def ensure_directories() -> None:
    """Создать все каталоги, используемые приложением."""
    for path in (UPLOAD_DIR, REPORT_DIR, DOCUMENTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        (INDEX_ROOT / strategy).mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    """Настроить файловое и консольное логирование один раз."""
    ensure_directories()
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
    )
