"""Независимый от UI пайплайн ответов с RAG и без RAG."""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from .config import INDEX_ROOT
from .embeddings import EmbeddingService
from .llm import LLMError, LLMProvider
from .models import AnswerResult, RetrievedChunk
from .prompts import build_no_rag_messages, build_rag_messages
from .search import semantic_search
from .text_utils import OffsetTokenizer, RegexTokenizer


ProgressCallback = Callable[[str], None]

class Retriever(Protocol):
    """Интерфейс получения релевантных чанков."""

    def retrieve(self, query: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        """Найти релевантные чанки в порядке убывания сходства."""


class RAGError(RuntimeError):
    """Контролируемая ошибка RAG-пайплайна."""


class SemanticRetriever:
    """Адаптер существующего семантического поиска для RAG."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        min_score: float = 0.25,
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
        """Найти, дедуплицировать и по возможности разнообразить чанки."""
        candidate_count = min(20, max(top_k * 3, top_k))
        rows = semantic_search(
            query,
            strategy,
            candidate_count,
            embedding_service=self.embedding_service,
            index_root=self.index_root,
        )
        candidates: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        for row in rows:
            if float(row["score"]) < self.min_score:
                continue
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=str(row["source"]),
                    title=row.get("title"),
                    section=row.get("section"),
                    page_number=row.get("page_number"),
                    content=str(row["content"]),
                    score=float(row["score"]),
                    token_count=int(row.get("token_count") or 0),
                )
            )
        selected: list[RetrievedChunk] = []
        deferred: list[RetrievedChunk] = []
        per_source: Counter[str] = Counter()
        for chunk in candidates:
            if per_source[chunk.source] >= 3:
                deferred.append(chunk)
                continue
            selected.append(chunk)
            per_source[chunk.source] += 1
            if len(selected) == top_k:
                return selected
        for chunk in deferred:
            selected.append(chunk)
            if len(selected) == top_k:
                break
        return selected


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
    context = prefix + "\n\n".join(blocks) + suffix
    return context, used


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


def answer_question(
    question: str,
    mode: Literal["rag", "no_rag"],
    llm: LLMProvider,
    strategy: str = "structural",
    top_k: int = 5,
    max_context_tokens: int = 4000,
    temperature: float = 0.1,
    *,
    retriever: Retriever | None = None,
    tokenizer: OffsetTokenizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnswerResult:
    """Ответить на вопрос в одном из двух сравниваемых режимов."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Вопрос не может быть пустым.")
    if mode not in ("rag", "no_rag"):
        raise ValueError("Режим должен быть rag или no_rag.")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k должен находиться в диапазоне от 1 до 20.")
    if max_context_tokens <= 0:
        raise ValueError("Бюджет контекста должен быть больше нуля.")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("Temperature должна находиться между 0 и 2.")

    started = time.perf_counter()
    model_name = getattr(llm, "model_name", llm.__class__.__name__)
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
            model_name=str(model_name),
            strategy=None,
            top_k=None,
            generation_time=generation_time,
            total_time=time.perf_counter() - started,
        )

    retrieval_started = time.perf_counter()
    if progress_callback:
        progress_callback("query_embedding")
    active_retriever = retriever or SemanticRetriever()
    retrieved = active_retriever.retrieve(normalized_question, strategy, top_k)
    if progress_callback:
        progress_callback("retrieval")
    retrieval_time = time.perf_counter() - retrieval_started
    active_tokenizer = tokenizer or getattr(active_retriever, "tokenizer", RegexTokenizer())
    context, used_chunks = build_context(
        retrieved,
        max_context_tokens,
        tokenizer=active_tokenizer,
    )
    if progress_callback:
        progress_callback("context")
    if not used_chunks:
        return AnswerResult(
            question=normalized_question,
            answer="Релевантный контекст в локальных документах не найден; ответ по базе невозможен.",
            mode="rag",
            model_name=str(model_name),
            strategy=strategy,
            top_k=top_k,
            retrieved_chunks=[],
            sources=[],
            retrieval_time=retrieval_time,
            total_time=time.perf_counter() - started,
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
        mode="rag",
        model_name=str(model_name),
        strategy=strategy,
        top_k=top_k,
        retrieved_chunks=used_chunks,
        sources=sources,
        context_token_count=_count_tokens(context, active_tokenizer),
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=time.perf_counter() - started,
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
    """Последовательно получить честную пару ответов одной LLM."""
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
