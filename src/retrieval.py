"""Query Rewrite, fusion, фильтрация и локальный reranking без зависимости от UI."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .config import INDEX_ROOT
from .embeddings import EmbeddingService
from .llm import LLMProvider
from .models import (
    EnhancedRetrievalResult,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievedChunk,
    RewrittenQuery,
)
from .prompts import build_query_rewrite_messages
from .search import semantic_search


class RetrievalError(RuntimeError):
    """Контролируемая ошибка расширенного retrieval."""


class CandidateSearcher(Protocol):
    """Интерфейс широкого поиска кандидатов для одного запроса."""

    def search(self, query: str, strategy: str, top_k: int) -> list[dict[str, Any]]:
        """Вернуть ранжированные результаты поиска."""


class Reranker(Protocol):
    """Общий интерфейс локального reranker."""

    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """Вернуть новые экземпляры кандидатов в итоговом порядке."""


class SemanticCandidateSearcher:
    """Адаптер существующего FAISS-поиска без скрытой фильтрации."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        index_root: Path = INDEX_ROOT,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.index_root = Path(index_root)

    @property
    def tokenizer(self):
        """Вернуть токенизатор embedding-модели."""
        return self.embedding_service.tokenizer

    def search(self, query: str, strategy: str, top_k: int) -> list[dict[str, Any]]:
        """Выполнить исходный semantic search с указанным Top-K."""
        return semantic_search(
            query,
            strategy,
            top_k,
            embedding_service=self.embedding_service,
            index_root=self.index_root,
        )


def _normalize_variant(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Поисковый запрос в JSON должен быть строкой.")
    normalized = " ".join(value.split())
    if not normalized:
        return ""
    if len(normalized) > 500 or len(normalized.split()) > 60:
        raise ValueError("Query Rewrite вернул ответ вместо краткого поискового запроса.")
    return normalized


def rewrite_query(
    question: str,
    llm: LLMProvider,
    max_alternatives: int = 2,
) -> RewrittenQuery:
    """Безопасно переписать вопрос; любая ошибка даёт fallback на исходный вопрос."""
    original = question.strip()
    if not original:
        raise ValueError("Вопрос не может быть пустым.")
    if not 0 <= max_alternatives <= 3:
        raise ValueError("Число альтернативных запросов должно быть от 0 до 3.")
    started = time.perf_counter()
    try:
        raw = llm.generate(build_query_rewrite_messages(original, max_alternatives), 0.0)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Query Rewrite должен вернуть JSON-объект.")
        if set(payload) - {"search_query", "alternative_queries"}:
            raise ValueError("Query Rewrite вернул неизвестные поля JSON.")
        search_query = _normalize_variant(payload.get("search_query"))
        alternatives_raw = payload.get("alternative_queries", [])
        if not search_query:
            raise ValueError("Query Rewrite вернул пустой search_query.")
        if not isinstance(alternatives_raw, list):
            raise ValueError("alternative_queries должен быть JSON-массивом.")

        seen = {original.casefold(), search_query.casefold()}
        alternatives: list[str] = []
        for value in alternatives_raw:
            alternative = _normalize_variant(value)
            key = alternative.casefold()
            if not alternative or key in seen:
                continue
            seen.add(key)
            alternatives.append(alternative)
            if len(alternatives) == max_alternatives:
                break
        rewrite_used = search_query.casefold() != original.casefold() or bool(alternatives)
        return RewrittenQuery(
            original_query=original,
            search_query=search_query,
            alternative_queries=alternatives,
            rewrite_used=rewrite_used,
            rewrite_time=time.perf_counter() - started,
        )
    except Exception as exc:
        return RewrittenQuery(
            original_query=original,
            search_query=original,
            fallback_reason=f"Query Rewrite не применён: {exc}",
            rewrite_time=time.perf_counter() - started,
        )


def _candidate_from_row(row: dict[str, Any], query: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=str(row["chunk_id"]),
        source=str(row["source"]),
        title=row.get("title"),
        section=row.get("section"),
        page_number=row.get("page_number"),
        content=str(row["content"]),
        similarity=float(row.get("score", row.get("similarity", 0.0))),
        original_rank=int(row.get("rank") or rank),
        matched_queries=[query],
        token_count=int(row.get("token_count") or 0),
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalCandidate]],
    k: int = 60,
) -> list[RetrievalCandidate]:
    """Объединить списки по RRF с детерминированным разрешением равенств."""
    if k < 1:
        raise ValueError("Параметр RRF k должен быть больше нуля.")
    fused: dict[str, RetrievalCandidate] = {}
    scores: Counter[str] = Counter()
    for ranked in ranked_lists:
        for position, candidate in enumerate(ranked, start=1):
            scores[candidate.chunk_id] += 1.0 / (k + position)
            current = fused.get(candidate.chunk_id)
            if current is None:
                fused[candidate.chunk_id] = replace(
                    candidate,
                    matched_queries=list(candidate.matched_queries),
                )
                continue
            matched = list(current.matched_queries)
            for query in candidate.matched_queries:
                if query not in matched:
                    matched.append(query)
            best = current
            if (-candidate.similarity, candidate.original_rank, candidate.chunk_id) < (
                -current.similarity,
                current.original_rank,
                current.chunk_id,
            ):
                best = candidate
            fused[candidate.chunk_id] = replace(
                best,
                matched_queries=matched,
                original_rank=min(current.original_rank, candidate.original_rank),
                similarity=max(current.similarity, candidate.similarity),
            )
    output = [replace(candidate, fusion_score=float(scores[chunk_id])) for chunk_id, candidate in fused.items()]
    return sorted(
        output,
        key=lambda item: (
            -(item.fusion_score or 0.0),
            -item.similarity,
            item.original_rank,
            item.chunk_id,
        ),
    )


def retrieve_candidates(
    queries: list[str],
    strategy: str,
    candidate_top_k: int,
    *,
    searcher: CandidateSearcher | None = None,
) -> list[RetrievalCandidate]:
    """Искать по каждому запросу, затем дедуплицировать результаты по chunk_id."""
    normalized_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        normalized_queries.append(normalized)
    if not normalized_queries:
        raise ValueError("Нужен хотя бы один непустой поисковый запрос.")
    if not 1 <= candidate_top_k <= 20:
        raise ValueError("candidate_top_k должен находиться в диапазоне от 1 до 20.")
    active_searcher = searcher or SemanticCandidateSearcher()
    ranked_lists: list[list[RetrievalCandidate]] = []
    for query in normalized_queries:
        rows = active_searcher.search(query, strategy, candidate_top_k)
        ranked_lists.append(
            [_candidate_from_row(row, query, rank) for rank, row in enumerate(rows, start=1)]
        )
    if len(ranked_lists) == 1:
        return [replace(item, matched_queries=list(item.matched_queries)) for item in ranked_lists[0]]
    return reciprocal_rank_fusion(ranked_lists)


def filter_by_similarity(
    candidates: list[RetrievalCandidate],
    threshold: float,
) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
    """Разделить кандидатов по исходной similarity без изменения score."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold должен находиться между 0 и 1.")
    accepted: list[RetrievalCandidate] = []
    rejected: list[RetrievalCandidate] = []
    for candidate in candidates:
        if candidate.similarity >= threshold:
            accepted.append(replace(candidate, status="accepted", rejection_reason=None))
        else:
            rejected.append(
                replace(
                    candidate,
                    status="rejected_similarity",
                    rejection_reason=(
                        f"Similarity {candidate.similarity:.3f} ниже установленного "
                        f"порога {threshold:.3f}."
                    ),
                )
            )
    return accepted, rejected


def normalize_semantic_score(similarity: float) -> float:
    """Преобразовать cosine similarity из [-1, 1] в [0, 1]."""
    return (max(-1.0, min(1.0, float(similarity))) + 1.0) / 2.0


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+", re.UNICODE)
_STOP_WORDS = {
    "а", "без", "в", "где", "для", "и", "из", "как", "какая", "какие", "какой",
    "на", "не", "он", "она", "оно", "по", "при", "с", "что", "это", "the", "a",
    "an", "and", "for", "how", "in", "is", "of", "on", "to", "what", "where",
}


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    }


def lexical_score(query: str, content: str) -> float:
    """Оценить долю значимых терминов вопроса, найденных в тексте чанка."""
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    return len(query_terms & _terms(content)) / len(query_terms)


def _is_technical(term: str) -> bool:
    return "_" in term or any(character.isdigit() for character in term) or (
        any(character.isupper() for character in term[1:])
    )


def metadata_score(query: str, candidate: RetrievalCandidate) -> float:
    """Оценить совпадение запроса с source/title/section, усиливая идентификаторы."""
    raw_terms = [token for token in _TOKEN_RE.findall(query) if token.casefold() not in _STOP_WORDS]
    if not raw_terms:
        return 0.0
    metadata = " ".join(
        value for value in (candidate.source, candidate.title, candidate.section) if value
    ).casefold()
    numerator = 0.0
    denominator = 0.0
    for term in raw_terms:
        weight = 2.0 if _is_technical(term) else 1.0
        denominator += weight
        if term.casefold() in metadata:
            numerator += weight
    return numerator / denominator if denominator else 0.0


class NoOpReranker:
    """Reranker, сохраняющий входной порядок."""

    name = "none"

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        del query
        return [
            replace(
                candidate,
                semantic_score=normalize_semantic_score(candidate.similarity),
                final_score=normalize_semantic_score(candidate.similarity),
                final_rank=rank,
            )
            for rank, candidate in enumerate(candidates, start=1)
        ]


class HeuristicReranker:
    """Детерминированный локальный reranker без дополнительной модели."""

    name = "heuristic"

    def __init__(
        self,
        semantic_weight: float = 0.70,
        lexical_weight: float = 0.20,
        metadata_weight: float = 0.10,
    ) -> None:
        weights = (semantic_weight, lexical_weight, metadata_weight)
        if any(weight < 0.0 for weight in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Веса heuristic reranker должны быть неотрицательными и давать сумму 1.")
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.metadata_weight = metadata_weight

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        scored: list[RetrievalCandidate] = []
        for candidate in candidates:
            semantic = normalize_semantic_score(candidate.similarity)
            lexical = lexical_score(query, candidate.content)
            metadata = metadata_score(query, candidate)
            final = (
                self.semantic_weight * semantic
                + self.lexical_weight * lexical
                + self.metadata_weight * metadata
            )
            scored.append(
                replace(
                    candidate,
                    semantic_score=semantic,
                    lexical_score=lexical,
                    metadata_score=metadata,
                    rerank_score=final,
                    final_score=final,
                )
            )
        scored.sort(
            key=lambda item: (
                -(item.final_score or 0.0),
                -(item.fusion_score or 0.0),
                -item.similarity,
                item.original_rank,
                item.chunk_id,
            )
        )
        return [replace(candidate, final_rank=rank) for rank, candidate in enumerate(scored, start=1)]


def limit_chunks_per_source(
    candidates: list[RetrievalCandidate],
    max_chunks_per_source: int | None,
) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
    """Сохранить лучшие чанки каждого источника после ранжирования."""
    if max_chunks_per_source is None:
        return [replace(item) for item in candidates], []
    if max_chunks_per_source < 1:
        raise ValueError("Лимит чанков одного источника должен быть больше нуля.")
    counts: Counter[str] = Counter()
    accepted: list[RetrievalCandidate] = []
    rejected: list[RetrievalCandidate] = []
    for candidate in candidates:
        if counts[candidate.source] >= max_chunks_per_source:
            rejected.append(
                replace(
                    candidate,
                    status="rejected_source_limit",
                    rejection_reason=(
                        f"Источник {candidate.source} уже дал максимум "
                        f"{max_chunks_per_source} чанков."
                    ),
                )
            )
            continue
        counts[candidate.source] += 1
        accepted.append(replace(candidate))
    return accepted, rejected


def enhanced_retrieve(
    question: str,
    strategy: str,
    config: RetrievalConfig,
    rewrite_llm: LLMProvider | None,
    reranker: Reranker | None = None,
    *,
    searcher: CandidateSearcher | None = None,
) -> EnhancedRetrievalResult:
    """Выполнить rewrite → multi-search → RRF → filter → rerank → Final Top-K."""
    original = question.strip()
    if not original:
        raise ValueError("Вопрос не может быть пустым.")
    started = time.perf_counter()
    rewritten = RewrittenQuery(original, original)
    if config.use_query_rewrite:
        if rewrite_llm is None:
            rewritten = RewrittenQuery(
                original,
                original,
                fallback_reason="Query Rewrite не применён: LLM-провайдер не задан.",
            )
        else:
            rewritten = rewrite_query(original, rewrite_llm, config.max_alternative_queries)

    queries = [original]
    if rewritten.search_query.casefold() != original.casefold():
        queries.append(rewritten.search_query)
    queries.extend(rewritten.alternative_queries)

    search_started = time.perf_counter()
    candidates = retrieve_candidates(
        queries,
        strategy,
        config.candidate_top_k,
        searcher=searcher,
    )
    candidates = candidates[: config.candidate_top_k]
    search_time = time.perf_counter() - search_started

    filter_started = time.perf_counter()
    if config.use_similarity_filter:
        accepted, rejected_similarity = filter_by_similarity(
            candidates, config.similarity_threshold
        )
    else:
        accepted = [replace(candidate, status="accepted") for candidate in candidates]
        rejected_similarity = []
    filter_time = time.perf_counter() - filter_started

    rerank_started = time.perf_counter()
    active_reranker = reranker
    if active_reranker is None:
        active_reranker = (
            HeuristicReranker() if config.reranker_type == "heuristic" else NoOpReranker()
        )
    try:
        reranked = active_reranker.rerank(original, accepted)
    except Exception as exc:
        raise RetrievalError(f"Reranking завершился ошибкой: {exc}") from exc

    passed_rerank: list[RetrievalCandidate] = []
    rejected_rerank: list[RetrievalCandidate] = []
    for candidate in reranked:
        score = candidate.final_score
        if config.rerank_threshold is not None and (
            score is None or score < config.rerank_threshold
        ):
            rejected_rerank.append(
                replace(
                    candidate,
                    status="rejected_rerank",
                    rejection_reason=(
                        f"Rerank score {score or 0.0:.3f} ниже установленного "
                        f"порога {config.rerank_threshold:.3f}."
                    ),
                )
            )
        else:
            passed_rerank.append(replace(candidate, status="accepted"))

    limited, rejected_source = limit_chunks_per_source(
        passed_rerank, config.max_chunks_per_source
    )
    selected = [replace(item, status="selected", final_rank=rank) for rank, item in enumerate(limited[: config.final_top_k], start=1)]
    not_selected = [
        replace(item, status="accepted_not_selected")
        for item in limited[config.final_top_k :]
    ]
    rerank_time = time.perf_counter() - rerank_started

    status_by_id = {
        item.chunk_id: item
        for item in (
            rejected_similarity
            + rejected_rerank
            + rejected_source
            + selected
            + not_selected
        )
    }
    diagnostic_candidates = [status_by_id.get(item.chunk_id, item) for item in candidates]
    accepted_diagnostics = [
        status_by_id.get(item.chunk_id, item) for item in accepted
    ]
    rejected = [
        item
        for item in diagnostic_candidates
        if item.status.startswith("rejected_")
    ]
    return EnhancedRetrievalResult(
        original_query=original,
        rewritten_query=(rewritten.search_query if rewritten.rewrite_used else None),
        alternative_queries=list(rewritten.alternative_queries),
        initial_candidates=diagnostic_candidates,
        accepted_candidates=accepted_diagnostics,
        rejected_candidates=rejected,
        final_chunks=selected,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        reranked_count=len(passed_rerank),
        final_count=len(selected),
        rewrite_used=rewritten.rewrite_used,
        fallback_reason=rewritten.fallback_reason,
        rewrite_time=rewritten.rewrite_time,
        search_time=search_time,
        filter_time=filter_time,
        rerank_time=rerank_time,
        total_retrieval_time=time.perf_counter() - started,
    )


def candidate_to_retrieved_chunk(candidate: RetrievalCandidate) -> RetrievedChunk:
    """Преобразовать выбранного кандидата в существующий формат RAG-контекста."""
    return RetrievedChunk(
        chunk_id=candidate.chunk_id,
        source=candidate.source,
        title=candidate.title,
        section=candidate.section,
        page_number=candidate.page_number,
        content=candidate.content,
        score=candidate.similarity,
        token_count=candidate.token_count,
    )
