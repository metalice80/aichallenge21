"""Локальная генерация нормализованных Sentence Transformer эмбеддингов."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import MODEL_NAME
from .text_utils import TokenSpan

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Ошибка загрузки модели или генерации эмбеддингов."""


class HuggingFaceOffsetTokenizer:
    """Адаптер токенизатора Sentence Transformers для chunking."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def token_spans(self, text: str) -> list[TokenSpan]:
        """Вернуть смещения без специальных токенов."""
        if not text:
            return []
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded.get("offset_mapping", [])
        return [TokenSpan(int(start), int(end)) for start, end in offsets if int(end) > int(start)]


class EmbeddingService:
    """Кешируемый сервис локальной модели эмбеддингов."""

    _models: dict[str, Any] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("Размер пакета должен быть больше нуля.")
        self.model_name = model_name
        self.batch_size = batch_size

    @property
    def model(self) -> Any:
        """Лениво загрузить и кешировать модель процесса."""
        if self.model_name not in self._models:
            with self._lock:
                if self.model_name not in self._models:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Загрузка локальной модели %s", self.model_name)
                        self._models[self.model_name] = SentenceTransformer(self.model_name)
                    except Exception as exc:
                        logger.exception("Не удалось загрузить модель эмбеддингов")
                        raise EmbeddingError(
                            f"Не удалось загрузить модель {self.model_name}: {exc}"
                        ) from exc
        return self._models[self.model_name]

    @property
    def tokenizer(self) -> HuggingFaceOffsetTokenizer:
        """Вернуть токенизатор загруженной модели."""
        return HuggingFaceOffsetTokenizer(self.model.tokenizer)

    @property
    def dimension(self) -> int:
        """Вернуть размерность эмбеддингов модели."""
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32, copy=False)

    def encode_documents(
        self,
        texts: list[str],
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        """Пакетно кодировать тексты, не передавая модели пустые строки."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        nonempty = [(index, text) for index, text in enumerate(texts) if text.strip()]
        if not nonempty:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        indexes: list[int] = []
        total = len(nonempty)
        try:
            for start in range(0, total, self.batch_size):
                batch_items = nonempty[start : start + self.batch_size]
                batch = [text for _, text in batch_items]
                encoded = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                batches.append(np.asarray(encoded, dtype=np.float32))
                indexes.extend(index for index, _ in batch_items)
                if progress_callback:
                    progress_callback(min(1.0, (start + len(batch_items)) / total))
        except Exception as exc:
            logger.exception("Ошибка генерации эмбеддингов")
            raise EmbeddingError(f"Не удалось создать эмбеддинги: {exc}") from exc
        combined = self._normalize(np.vstack(batches))
        if len(nonempty) == len(texts):
            return combined
        result = np.zeros((len(texts), combined.shape[1]), dtype=np.float32)
        result[indexes] = combined
        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Кодировать один непустой поисковый запрос."""
        if not query.strip():
            raise ValueError("Поисковый запрос не может быть пустым.")
        return self.encode_documents([query])[0]
