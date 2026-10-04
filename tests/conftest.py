from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.text_utils import RegexTokenizer


class FakeEmbeddingService:
    model_name = "fake-local-model"

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension
        self.tokenizer = RegexTokenizer()

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "little") % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_documents(self, texts: list[str], progress_callback=None) -> np.ndarray:
        vectors = np.vstack([self._encode(text) for text in texts]).astype(np.float32)
        if progress_callback:
            progress_callback(1.0)
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("empty query")
        return self._encode(query)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingService:
    return FakeEmbeddingService()
