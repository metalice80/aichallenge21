"""Независимый от UI пайплайн ответов no-RAG, Baseline, Filtered и Enhanced RAG."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from .config import INDEX_ROOT
from .embeddings import EmbeddingService
from .llm import LLMError, LLMProvider
from .models import AnswerResult, RetrievalConfig, RetrievedChunk
from .prompts import build_no_rag_messages, build_rag_messages
from .retrieval import (
    CandidateSearcher,
    NoOpReranker,
    Reranker,
    SemanticCandidateSearcher,
    candidate_to_retrieved_chunk,
    enhanced_retrieve,
)
from .search import semantic_search
from .text_utils import OffsetTokenizer, RegexTokenizer


ProgressCallback = Callable[[str], None]
RAGMode = Literal["no_rag", "rag", "baseline_rag", "filtered_rag", "enhanced_rag"]


class Retriever(Protocol):
    """Интерфейс получения релевантных чанков для Baseline RAG."""

    def retrieve(self, query: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        """Найти релевантные чанки в порядке FAISS."""


class RAGError(RuntimeError):
    """Контролируемая ошибка RAG-пайплайна."""


class SemanticRetriever:
    """Адаптер существующего семантического поиска без скрытого reranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        min_score: float = -1.0,
        index_root: Path = INDEX_ROOT,
    ) -> None:
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("Порог similarity должен находиться между -1 и 1.")
        self.embedding_service = embedding_service or EmbeddingService()
        self.min_score = min_score
        self.index_root = Path(index_root)

    @property
    def tokenizer(self) -> OffsetTokenizer:
        """Вернуть токенизатор модели индекса."""
        return self.embedding_service.tokenizer

    def retrieve(self, query: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        """Получить обычный Top-K FAISS; фильтр применяется только при явном min_score."""
        rows = semantic_search(
            query,
            strategy,
            top_k,
            embedding_service=self.embedding_service,
            index_root=self.index_root,
        )
        return [
            RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                source=str(row["source"]),
                title=row.get("title"),
                section=row.get("section"),
                page_number=row.get("page_number"),
                content=str(row["content"]),
                score=float(row["score"]),
                token_count=int(row.get("token_count") or 0),
            )
            for row in rows
            if float(row["score"]) >= self.min_score
        ]


class _RetrieverCandidateAdapter:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def search(self, query: str, strategy: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "rank": rank,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "section": chunk.section,
                "page_number": chunk.page_number,
                "content": chunk.content,
                "token_count": chunk.token_count,
            }
            for rank, chunk in enumerate(
                self.retriever.retrieve(query, strategy, top_k), start=1
            )
        ]


def _count_tokens(text: str, tokenizer: OffsetTokenizer) -> int:
    return len(tokenizer.token_spans(text))


def _source_key(chunk: RetrievedChunk) -> tuple[str, str | None, int | None]:
    return chunk.source, chunk.section, chunk.page_number


def build_context(
    chunks: list[RetrievedChunk],
    max_tokens: int,
    *,
    tokenizer: OffsetTokenizer | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Собрать нумерованный контекст целыми чанками в пределах бюджета."""
    if max_tokens <= 0:
        raise ValueError("Бюджет контекста должен быть больше нуля.")
    active_tokenizer = tokenizer or RegexTokenizer()
    prefix = "--- НАЧАЛО НЕДОВЕРЕННОГО КОНТЕКСТА ---\n\n"
    suffix = "\n\n--- КОНЕЦ НЕДОВЕРЕННОГО КОНТЕКСТА ---"
    seen_chunk_ids: set[str] = set()
    source_numbers: dict[tuple[str, str | None, int | None], int] = {}
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    token_total = _count_tokens(prefix + suffix, active_tokenizer)
    if token_total > max_tokens:
        return "", []
    for chunk in chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        key = _source_key(chunk)
        source_number = source_numbers.get(key, len(source_numbers) + 1)
        block = (
            f"[Источник {source_number}]\n"
            f"Файл: {chunk.source}\n"
            f"Заголовок: {chunk.title or '—'}\n"
            f"Раздел: {chunk.section or '—'}\n"
            f"Страница: {chunk.page_number if chunk.page_number is not None else '—'}\n"
            f"Chunk ID: {chunk.chunk_id}\n\n"
            f"{chunk.content.strip()}"
        )
        block_tokens = _count_tokens(block, active_tokenizer)
        if token_total + block_tokens > max_tokens:
            continue
        source_numbers.setdefault(key, source_number)
        blocks.append(block)
        used.append(replace(chunk, source_number=source_number))
        token_total += block_tokens
    if not blocks:
        return "", []
    return prefix + "\n\n".join(blocks) + suffix, used


def _sources_from_chunks(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    seen: set[tuple[str, str | None, int | None]] = set()
    for chunk in chunks:
        key = _source_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "number": chunk.source_number or len(sources) + 1,
                "source": chunk.source,
                "title": chunk.title,
                "section": chunk.section,
                "page_number": chunk.page_number,
            }
        )
    return sources


def _ensure_source_references(answer: str, sources: list[dict[str, object]]) -> str:
    valid_numbers = {int(source["number"]) for source in sources}
    cited = {int(value) for value in re.findall(r"\[(\d+)]", answer)}
    if cited & valid_numbers:
        return answer
    references = ", ".join(
        f"[{source['number']}] {source['source']}"
        + (f" — {source['section']}" if source.get("section") else "")
        for source in sources
    )
    return f"{answer.rstrip()}\n\nИсточники: {references}"


def _empty_context_result(
    *,
    question: str,
    mode: str,
    model_name: str,
    strategy: str,
    top_k: int,
    started: float,
    retrieval_time: float,
    diagnostics: dict[str, object],
) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=(
            "Релевантный контекст в локальных документах не найден; "
            "ответ по базе невозможен."
        ),
        mode=mode,
        model_name=model_name,
        strategy=strategy,
        top_k=top_k,
        retrieval_time=retrieval_time,
        total_time=time.perf_counter() - started,
        retrieval_mode=("baseline_rag" if mode == "rag" else mode),
        original_query=question,
        rewritten_query=diagnostics.get("rewritten_query"),
        alternative_queries=list(diagnostics.get("alternative_queries", [])),
        candidate_count=int(diagnostics.get("candidate_count", 0)),
        accepted_count=int(diagnostics.get("accepted_count", 0)),
        final_count=0,
        rewrite_time=float(diagnostics.get("rewrite_time", 0.0)),
        filter_time=float(diagnostics.get("filter_time", 0.0)),
        rerank_time=float(diagnostics.get("rerank_time", 0.0)),
        retrieval_diagnostics=diagnostics,
    )


def answer_question(
    question: str,
    mode: RAGMode,
    llm: LLMProvider,
    strategy: str = "structural",
    top_k: int = 5,
    max_context_tokens: int = 4000,
    temperature: float = 0.1,
    retrieval_config: RetrievalConfig | None = None,
    *,
    retriever: Retriever | None = None,
    candidate_searcher: CandidateSearcher | None = None,
    rewrite_llm: LLMProvider | None = None,
    reranker: Reranker | None = None,
    tokenizer: OffsetTokenizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnswerResult:
    """Ответить в одном из режимов, не передавая отклонённые чанки в LLM."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Вопрос не может быть пустым.")
    allowed_modes = {"no_rag", "rag", "baseline_rag", "filtered_rag", "enhanced_rag"}
    if mode not in allowed_modes:
        raise ValueError("Неизвестный режим ответа.")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k должен находиться в диапазоне от 1 до 20.")
    if max_context_tokens <= 0:
        raise ValueError("Бюджет контекста должен быть больше нуля.")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("Temperature должна находиться между 0 и 2.")

    started = time.perf_counter()
    model_name = str(getattr(llm, "model_name", llm.__class__.__name__))
    requested_top_k = top_k
    if mode == "no_rag":
        if progress_callback:
            progress_callback("no_rag_generation")
        generation_started = time.perf_counter()
        try:
            answer = llm.generate(build_no_rag_messages(normalized_question), temperature)
        except LLMError:
            raise
        except Exception as exc:
            raise RAGError(f"Не удалось получить ответ без RAG: {exc}") from exc
        generation_time = time.perf_counter() - generation_started
        if not answer.strip():
            raise RAGError("LLM вернула пустой ответ без RAG.")
        return AnswerResult(
            question=normalized_question,
            answer=answer.strip(),
            mode="no_rag",
            model_name=model_name,
            strategy=None,
            top_k=None,
            generation_time=generation_time,
            total_time=time.perf_counter() - started,
            retrieval_mode="no_rag",
            original_query=normalized_question,
        )

    retrieval_started = time.perf_counter()
    diagnostics: dict[str, object]
    active_tokenizer: OffsetTokenizer
    if mode in {"rag", "baseline_rag"}:
        if progress_callback:
            progress_callback("query_embedding")
        active_retriever = retriever or SemanticRetriever()
        retrieved = active_retriever.retrieve(normalized_question, strategy, top_k)
        retrieval_time = time.perf_counter() - retrieval_started
        active_tokenizer = tokenizer or getattr(active_retriever, "tokenizer", RegexTokenizer())
        diagnostics = {
            "original_query": normalized_question,
            "rewritten_query": None,
            "alternative_queries": [],
            "rewrite_used": False,
            "fallback_reason": None,
            "candidate_count": len(retrieved),
            "accepted_count": len(retrieved),
            "reranked_count": len(retrieved),
            "final_count": len(retrieved),
            "rewrite_time": 0.0,
            "search_time": retrieval_time,
            "filter_time": 0.0,
            "rerank_time": 0.0,
            "initial_candidates": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "similarity": chunk.score,
                    "original_rank": rank,
                    "final_rank": rank,
                    "status": "selected",
                    "rejection_reason": None,
                    "matched_queries": [normalized_question],
                }
                for rank, chunk in enumerate(retrieved, start=1)
            ],
        }
    else:
        config = retrieval_config or RetrievalConfig()
        if mode == "filtered_rag":
            config = replace(
                config,
                use_query_rewrite=False,
                use_similarity_filter=True,
                reranker_type="none",
                rerank_threshold=None,
                max_chunks_per_source=config.candidate_top_k,
            )
            reranker = NoOpReranker()
        active_searcher = candidate_searcher
        if active_searcher is None:
            if retriever is not None:
                active_searcher = _RetrieverCandidateAdapter(retriever)
            else:
                active_searcher = SemanticCandidateSearcher()
        active_tokenizer = tokenizer or getattr(active_searcher, "tokenizer", RegexTokenizer())
        if progress_callback and config.use_query_rewrite and mode == "enhanced_rag":
            progress_callback("query_rewrite")
        retrieval_result = enhanced_retrieve(
            normalized_question,
            strategy,
            config,
            (rewrite_llm or llm) if mode == "enhanced_rag" else None,
            reranker,
            searcher=active_searcher,
        )
        diagnostics = retrieval_result.to_dict(include_content=True)
        retrieved = [
            candidate_to_retrieved_chunk(candidate)
            for candidate in retrieval_result.final_chunks
        ]
        retrieval_time = retrieval_result.total_retrieval_time

    if progress_callback:
        progress_callback("retrieval")
    context, used_chunks = build_context(
        retrieved,
        max_context_tokens,
        tokenizer=active_tokenizer,
    )
    if progress_callback:
        progress_callback("context")
    if not used_chunks:
        return _empty_context_result(
            question=normalized_question,
            mode=mode,
            model_name=model_name,
            strategy=strategy,
            top_k=(retrieval_config.final_top_k if retrieval_config and mode != "baseline_rag" else top_k),
            started=started,
            retrieval_time=retrieval_time,
            diagnostics=diagnostics,
        )

    generation_started = time.perf_counter()
    if progress_callback:
        progress_callback("rag_generation")
    try:
        answer = llm.generate(build_rag_messages(normalized_question, context), temperature)
    except LLMError:
        raise
    except Exception as exc:
        raise RAGError(f"Не удалось получить ответ с RAG: {exc}") from exc
    generation_time = time.perf_counter() - generation_started
    if not answer.strip():
        raise RAGError("LLM вернула пустой ответ с RAG.")
    sources = _sources_from_chunks(used_chunks)
    answer = _ensure_source_references(answer.strip(), sources)
    return AnswerResult(
        question=normalized_question,
        answer=answer,
        mode=mode,
        model_name=model_name,
        strategy=strategy,
        top_k=requested_top_k,
        retrieved_chunks=used_chunks,
        sources=sources,
        context_token_count=_count_tokens(context, active_tokenizer),
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=time.perf_counter() - started,
        retrieval_mode=("baseline_rag" if mode == "rag" else mode),
        original_query=normalized_question,
        rewritten_query=diagnostics.get("rewritten_query"),
        alternative_queries=list(diagnostics.get("alternative_queries", [])),
        candidate_count=int(diagnostics.get("candidate_count", len(retrieved))),
        accepted_count=int(diagnostics.get("accepted_count", len(retrieved))),
        final_count=len(used_chunks),
        rewrite_time=float(diagnostics.get("rewrite_time", 0.0)),
        filter_time=float(diagnostics.get("filter_time", 0.0)),
        rerank_time=float(diagnostics.get("rerank_time", 0.0)),
        retrieval_diagnostics=diagnostics,
    )


def compare_answers(
    question: str,
    llm: LLMProvider,
    *,
    strategy: str = "structural",
    top_k: int = 5,
    max_context_tokens: int = 4000,
    temperature: float = 0.1,
    retriever: Retriever | None = None,
    tokenizer: OffsetTokenizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[AnswerResult, AnswerResult]:
    """Сохранить совместимое сравнение no-RAG и прежнего Baseline RAG."""
    without_rag = answer_question(
        question,
        "no_rag",
        llm,
        strategy=strategy,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        progress_callback=progress_callback,
    )
    with_rag = answer_question(
        question,
        "rag",
        llm,
        strategy=strategy,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        retriever=retriever,
        tokenizer=tokenizer,
        progress_callback=progress_callback,
    )
    return without_rag, with_rag


def compare_retrieval_modes(
    question: str,
    llm: LLMProvider,
    *,
    strategy: str = "structural",
    baseline_top_k: int = 5,
    retrieval_config: RetrievalConfig | None = None,
    max_context_tokens: int = 4000,
    temperature: float = 0.1,
    retriever: Retriever | None = None,
    candidate_searcher: CandidateSearcher | None = None,
    tokenizer: OffsetTokenizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[AnswerResult, AnswerResult]:
    """Сравнить Baseline и Enhanced при одинаковой LLM и генерационных настройках."""
    config = retrieval_config or RetrievalConfig()
    baseline = answer_question(
        question,
        "baseline_rag",
        llm,
        strategy=strategy,
        top_k=baseline_top_k,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        retriever=retriever,
        tokenizer=tokenizer,
        progress_callback=progress_callback,
    )
    enhanced = answer_question(
        question,
        "enhanced_rag",
        llm,
        strategy=strategy,
        top_k=config.final_top_k,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        retrieval_config=config,
        retriever=retriever,
        candidate_searcher=candidate_searcher,
        rewrite_llm=llm,
        tokenizer=tokenizer,
        progress_callback=progress_callback,
    )
    return baseline, enhanced
