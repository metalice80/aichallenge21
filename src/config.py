"""Конфигурация локального индексатора документов."""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
INDEX_ROOT = PROJECT_ROOT / "indexes"
REPORT_DIR = PROJECT_ROOT / "reports"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
LOG_FILE = PROJECT_ROOT / "data" / "document_indexer.log"
RESULTS_DATABASE = PROJECT_ROOT / "data" / "results.sqlite"
CHAT_DATABASE = PROJECT_ROOT / "data" / "chat.sqlite"
CHAT_EXPORT_DIR = PROJECT_ROOT / "data" / "chat_exports"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RAG_QUESTIONS_PATH = EVALUATION_DIR / "questions.json"
RAG_RESULTS_PATH = EVALUATION_DIR / "results.json"
RAG_REPORT_PATH = REPORT_DIR / "rag_comparison.md"
CHAT_SCENARIOS_PATH = EVALUATION_DIR / "chat_scenarios.json"
CHAT_REPORT_PATH = REPORT_DIR / "chat_memory_evaluation.md"
RERANKING_REPORT_PATH = REPORT_DIR / "reranking_comparison.md"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".py"}
STRATEGIES = ("fixed", "structural")

CHAT_RECENT_MESSAGE_COUNT = 8
CHAT_SUMMARY_TRIGGER_MESSAGES = 12

def ensure_directories() -> None:
    """Создать все каталоги, используемые приложением."""
    for path in (UPLOAD_DIR, REPORT_DIR, DOCUMENTS_DIR, EVALUATION_DIR, CHAT_EXPORT_DIR):
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
