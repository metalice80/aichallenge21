"""Типы данных документов, чанков и результатов индексации."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Извлечённый документ или отдельная страница PDF."""

    document_id: str
    source: str
    title: str
    file_type: str
    text: str
    page_number: int | None = None


@dataclass(slots=True)
class Chunk:
    """Текстовый фрагмент вместе с индексируемыми метаданными."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    section: str | None
    strategy: str
    content: str
    token_count: int
    page_number: int | None = None
    start_position: int | None = None
    end_position: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Вернуть сериализуемое представление чанка."""
        return asdict(self)


@dataclass(slots=True)
class IndexingResult:
    """Итог построения одного индекса."""

    strategy: str
    document_count: int
    chunk_count: int
    average_chunk_size: float
    elapsed_seconds: float
    index_path: str
    chunking_seconds: float
    embedding_seconds: float


@dataclass(slots=True)
class RetrievedChunk:
    """Чанк, найденный для формирования RAG-контекста."""

    chunk_id: str
    source: str
    title: str | None
    section: str | None
    page_number: int | None
    content: str
    score: float
    token_count: int = 0
    source_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Вернуть сериализуемое представление найденного чанка."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Проверенная конфигурация расширенного retrieval."""

    candidate_top_k: int = 20
    final_top_k: int = 5
    similarity_threshold: float = 0.35
    rerank_threshold: float | None = None
    use_query_rewrite: bool = True
    use_similarity_filter: bool = True
    reranker_type: str = "heuristic"
    max_alternative_queries: int = 2
    max_chunks_per_source: int | None = 3

    def __post_init__(self) -> None:
        if not 1 <= self.candidate_top_k <= 20:
            raise ValueError("candidate_top_k должен находиться в диапазоне от 1 до 20.")
        if not 1 <= self.final_top_k <= self.candidate_top_k:
            raise ValueError("final_top_k должен быть от 1 до candidate_top_k.")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold должен находиться между 0 и 1.")
        if self.rerank_threshold is not None and not 0.0 <= self.rerank_threshold <= 1.0:
            raise ValueError("Rerank threshold должен находиться между 0 и 1.")
        if not 0 <= self.max_alternative_queries <= 3:
            raise ValueError("Число альтернативных запросов должно быть от 0 до 3.")
        if self.reranker_type not in {"none", "heuristic"}:
            raise ValueError("Поддерживаются reranker_type: none и heuristic.")
        if self.max_chunks_per_source is not None and self.max_chunks_per_source < 1:
            raise ValueError("Лимит чанков одного источника должен быть больше нуля.")


@dataclass(slots=True)
class RewrittenQuery:
    """Проверенный результат Query Rewrite или безопасный fallback."""

    original_query: str
    search_query: str
    alternative_queries: list[str] = field(default_factory=list)
    rewrite_used: bool = False
    fallback_reason: str | None = None
    rewrite_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Вернуть сериализуемое представление переписанного запроса."""
        return asdict(self)


@dataclass(slots=True)
class RetrievalCandidate:
    """Кандидат retrieval со всеми компонентами ранжирования."""

    chunk_id: str
    source: str
    title: str | None
    section: str | None
    page_number: int | None
    content: str
    similarity: float
    original_rank: int
    matched_queries: list[str] = field(default_factory=list)
    fusion_score: float | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None
    metadata_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    final_rank: int | None = None
    status: str = "candidate"
    rejection_reason: str | None = None
    token_count: int = 0

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        """Вернуть сериализуемое представление кандидата."""
        payload = asdict(self)
        if not include_content:
            payload.pop("content", None)
        return payload


@dataclass(slots=True)
class EnhancedRetrievalResult:
    """Результат полного retrieval с диагностикой каждого этапа."""

    original_query: str
    rewritten_query: str | None
    alternative_queries: list[str]
    initial_candidates: list[RetrievalCandidate]
    accepted_candidates: list[RetrievalCandidate]
    rejected_candidates: list[RetrievalCandidate]
    final_chunks: list[RetrievalCandidate]
    candidate_count: int
    accepted_count: int
    reranked_count: int
    final_count: int
    rewrite_used: bool
    fallback_reason: str | None
    rewrite_time: float
    search_time: float
    filter_time: float
    rerank_time: float
    total_retrieval_time: float

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        """Вернуть сериализуемую retrieval-диагностику."""
        payload = asdict(self)
        if not include_content:
            for name in (
                "initial_candidates",
                "accepted_candidates",
                "rejected_candidates",
                "final_chunks",
            ):
                for candidate in payload[name]:
                    candidate.pop("content", None)
        return payload


@dataclass(slots=True)
class AnswerResult:
    """Структурированный результат генерации ответа."""

    question: str
    answer: str
    mode: str
    model_name: str
    strategy: str | None
    top_k: int | None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    context_token_count: int = 0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    retrieval_mode: str = "no_rag"
    original_query: str = ""
    rewritten_query: str | None = None
    alternative_queries: list[str] = field(default_factory=list)
    candidate_count: int = 0
    accepted_count: int = 0
    final_count: int = 0
    rewrite_time: float = 0.0
    filter_time: float = 0.0
    rerank_time: float = 0.0
    retrieval_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Вернуть сериализуемое представление ответа."""
        return asdict(self)
