"""Сквозная оркестрация постоянного чата поверх существующего Enhanced Retrieval."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chat_context import PreparedChatContext, prepare_chat_context
from .chat_models import (
    ChatContextBudget,
    ChatMessage,
    ChatTurnResult,
    TaskState,
    TaskStateChange,
)
from .chat_prompts import build_chat_messages
from .chat_storage import ChatStorage, ChatStorageError
from .config import (
    CHAT_DATABASE,
    CHAT_RECENT_MESSAGE_COUNT,
    CHAT_SUMMARY_TRIGGER_MESSAGES,
)
from .contextualization import contextualize_question_result
from .conversation_summary import messages_to_summarize, update_conversation_summary
from .llm import LLMProvider
from .models import RetrievalCandidate, RetrievalConfig, RetrievedChunk
from .retrieval import (
    CandidateSearcher,
    Reranker,
    candidate_to_retrieved_chunk,
    enhanced_retrieve,
)
from .task_memory import update_task_state
from .text_utils import OffsetTokenizer, RegexTokenizer

ProgressCallback = Callable[[str], None]
_NO_SOURCES_ANSWER = (
    "Релевантные источники в локальной базе не найдены. "
    "Я не могу подтвердить ответ документами проекта."
)


def _sanitize_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    """Удалить ссылки на несуществующие номера и добавить валидные при необходимости."""
    valid = {int(source["number"]) for source in sources}

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid else ""

    cleaned = re.sub(r"\[(\d+)]", replace, answer).strip()
    cited = {int(value) for value in re.findall(r"\[(\d+)]", cleaned)}
    if valid and not cited.intersection(valid):
        references = ", ".join(f"[{number}]" for number in sorted(valid))
        cleaned = f"{cleaned}\n\nИсточники: {references}"
    return cleaned


def _settings_int(settings: dict[str, Any], key: str, default: int, minimum: int) -> int:
    value = settings.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _budget_from_settings(settings: dict[str, Any]) -> ChatContextBudget:
    raw = settings.get("context_budget", {})
    if not isinstance(raw, dict):
        raw = {}
    defaults = ChatContextBudget()
    return ChatContextBudget(
        task_state_tokens=int(raw.get("task_state_tokens", defaults.task_state_tokens)),
        summary_tokens=int(raw.get("summary_tokens", defaults.summary_tokens)),
        recent_messages_tokens=int(
            raw.get("recent_messages_tokens", defaults.recent_messages_tokens)
        ),
        rag_context_tokens=int(raw.get("rag_context_tokens", defaults.rag_context_tokens)),
        response_reserve_tokens=int(
            raw.get("response_reserve_tokens", defaults.response_reserve_tokens)
        ),
    )


def _safe_error(exc: Exception, stage: str) -> str:
    message = " ".join(str(exc).split())[:500]
    if not message:
        message = "неизвестная ошибка"
    return f"Не удалось завершить этап «{stage}»: {message}"


def _source_rows(
    used_chunks: list[RetrievedChunk],
    selected_candidates: list[RetrievalCandidate],
) -> list[dict[str, Any]]:
    candidates = {candidate.chunk_id: candidate for candidate in selected_candidates}
    sources: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    for chunk in used_chunks:
        number = chunk.source_number or len(sources) + 1
        if number in seen_numbers:
            continue
        seen_numbers.add(number)
        candidate = candidates.get(chunk.chunk_id)
        sources.append(
            {
                "number": number,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "section": chunk.section,
                "page_number": chunk.page_number,
                "similarity": chunk.score,
                "rerank_score": candidate.rerank_score if candidate else None,
                "final_rank": candidate.final_rank if candidate else None,
                "content": chunk.content,
            }
        )
    return sources


def _change_for_message(storage: ChatStorage, conversation_id: str, message_id: str) -> TaskStateChange:
    for item in storage.list_task_state_versions(conversation_id):
        if item["related_message_id"] == message_id:
            payload = item["change"]
            return TaskStateChange(
                added=dict(payload.get("added", {})),
                removed=dict(payload.get("removed", {})),
                changed=dict(payload.get("changed", {})),
                reason=payload.get("reason"),
            )
    return TaskStateChange(reason="Содержательных изменений нет.")


class ChatService:
    """Идемпотентно обрабатывает ходы и сохраняет все независимые состояния."""

    def __init__(
        self,
        storage: ChatStorage,
        *,
        candidate_searcher: CandidateSearcher | None = None,
        tokenizer: OffsetTokenizer | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.storage = storage
        self.candidate_searcher = candidate_searcher
        self.tokenizer = tokenizer or getattr(candidate_searcher, "tokenizer", RegexTokenizer())
        self.reranker = reranker

    def process_chat_message(
        self,
        conversation_id: str,
        user_text: str,
        llm: LLMProvider,
        retrieval_config: RetrievalConfig,
        *,
        idempotency_key: str | None = None,
        context_budget: ChatContextBudget | None = None,
        temperature: float = 0.1,
        progress_callback: ProgressCallback | None = None,
        force_regenerate: bool = False,
    ) -> ChatTurnResult:
        """Выполнить pending → memory → contextualization → retrieval → generation."""
        normalized = user_text.strip()
        if not normalized:
            raise ValueError("Сообщение не может быть пустым.")
        conversation = self.storage.require_conversation(conversation_id)
        turn = self.storage.create_or_get_turn(
            conversation_id,
            idempotency_key=idempotency_key,
        )
        user_message, existing_assistant = self.storage.get_turn_messages(turn["id"])
        if user_message is not None and user_message.content != normalized:
            raise ChatStorageError(
                "Idempotency key уже связан с другим пользовательским сообщением."
            )
        if (
            turn["status"] == "completed"
            and existing_assistant is not None
            and not force_regenerate
        ):
            return self._restore_result(
                turn["id"], user_message, existing_assistant, conversation_id
            )
        user_message = user_message or self.storage.save_user_message(turn["id"], normalized)
        if conversation.title == "Новый диалог":
            self.storage.update_conversation(
                conversation_id,
                title=normalized[:60],
            )
        self.storage.update_message_status(user_message.id, "pending")

        timings: dict[str, float] = {}
        standalone_query = normalized
        rewritten_query: str | None = None
        retrieval_diagnostics: dict[str, Any] = {}
        task_state = self.storage.latest_task_state(conversation_id)
        task_change = TaskStateChange(reason="Память ещё не обновлялась.")
        sources: list[dict[str, Any]] = []
        prepared: PreparedChatContext | None = None
        stage = "pending"
        started = time.perf_counter()

        def enter(next_stage: str) -> None:
            nonlocal stage
            stage = next_stage
            self.storage.update_turn(turn["id"], next_stage, stage=next_stage)
            if progress_callback:
                progress_callback(next_stage)

        try:
            enter("updating_memory")
            memory_started = time.perf_counter()
            messages = self.storage.list_messages(conversation_id)
            prior_messages = [message for message in messages if message.id != user_message.id]
            turn_memory_version = turn.get("memory_version")
            if turn_memory_version is None:
                if conversation.settings.get("automatic_memory", True):
                    candidate_state, task_change = update_task_state(
                        task_state,
                        normalized,
                        prior_messages[-6:],
                        llm,
                    )
                    if task_change.has_changes:
                        task_state = self.storage.save_task_state(
                            conversation_id,
                            candidate_state,
                            task_change,
                            source="automatic",
                            related_message_id=user_message.id,
                            expected_version=task_state.version,
                        )
                else:
                    task_change = TaskStateChange(
                        reason="Автоматическое обновление памяти отключено."
                    )
                self.storage.update_turn(
                    turn["id"],
                    "updating_memory",
                    memory_version=task_state.version,
                )
            else:
                stored_state = self.storage.get_task_state_version(
                    conversation_id, int(turn_memory_version)
                )
                if stored_state is not None:
                    task_state = stored_state
                task_change = _change_for_message(
                    self.storage, conversation_id, user_message.id
                )
            timings["memory_update"] = time.perf_counter() - memory_started

            recent_count = _settings_int(
                conversation.settings,
                "recent_message_count",
                CHAT_RECENT_MESSAGE_COUNT,
                1,
            )
            summary = self.storage.get_summary(conversation_id)
            recent_messages = prior_messages[-recent_count:]

            enter("contextualizing")
            contextual = contextualize_question_result(
                normalized,
                task_state,
                recent_messages,
                summary.summary if summary else "",
                llm,
            )
            standalone_query = contextual.standalone_query
            timings["contextualization"] = contextual.elapsed_seconds

            enter("retrieving")
            retrieval = enhanced_retrieve(
                standalone_query,
                conversation.strategy,
                retrieval_config,
                llm if retrieval_config.use_query_rewrite else None,
                self.reranker,
                searcher=self.candidate_searcher,
            )
            rewritten_query = retrieval.rewritten_query
            retrieval_diagnostics = retrieval.to_dict(include_content=False)
            retrieval_diagnostics["contextualization"] = {
                "fallback_used": contextual.fallback_used,
                "fallback_reason": contextual.fallback_reason,
                "elapsed_seconds": contextual.elapsed_seconds,
            }
            timings["retrieval"] = retrieval.total_retrieval_time
            selected_chunks = [
                candidate_to_retrieved_chunk(candidate)
                for candidate in retrieval.final_chunks
                if candidate.status == "selected"
            ]
            prepared = prepare_chat_context(
                task_state=task_state,
                conversation_summary=summary.summary if summary else "",
                recent_messages=recent_messages,
                retrieved_chunks=selected_chunks,
                current_question=normalized,
                budget=context_budget or _budget_from_settings(conversation.settings),
                tokenizer=self.tokenizer,
            )
            sources = _source_rows(prepared.used_chunks, retrieval.final_chunks)

            enter("generating")
            generation_started = time.perf_counter()
            if not prepared.used_chunks:
                answer = _NO_SOURCES_ANSWER
            else:
                answer = llm.generate(
                    build_chat_messages(
                        task_state_block=prepared.task_state_block,
                        summary_block=prepared.summary_block,
                        recent_messages_block=prepared.recent_messages_block,
                        retrieved_context=prepared.rag_context_block,
                        current_question=normalized,
                    ),
                    temperature,
                ).strip()
                if not answer:
                    raise ValueError("LLM вернула пустой ответ.")
                answer = _sanitize_citations(answer, sources)
            timings["generation"] = time.perf_counter() - generation_started
            timings["total"] = time.perf_counter() - started

            assistant = self.storage.save_assistant_message(
                turn_id=turn["id"],
                content=answer,
                status="completed",
                standalone_query=standalone_query,
                rewritten_query=rewritten_query,
                retrieval_diagnostics=retrieval_diagnostics,
                token_usage=prepared.token_usage,
                sources=sources,
            )
            self.storage.update_message_status(user_message.id, "completed")
            self.storage.update_turn(turn["id"], "completed", stage="completed")
            self._update_summary_if_needed(conversation_id, task_state, llm)
            return ChatTurnResult(
                turn_id=turn["id"],
                user_message=self.storage.get_message(user_message.id) or user_message,
                assistant_message=assistant,
                task_state=task_state,
                task_state_change=task_change,
                sources=sources,
                retrieved_chunks=[asdict(chunk) for chunk in prepared.used_chunks],
                standalone_query=standalone_query,
                rewritten_query=rewritten_query,
                retrieval_diagnostics=retrieval_diagnostics,
                timings=timings,
            )
        except Exception as exc:
            safe_message = _safe_error(exc, stage)
            timings["total"] = time.perf_counter() - started
            self.storage.update_message_status(
                user_message.id, "failed", error_message=safe_message
            )
            self.storage.update_turn(
                turn["id"],
                "failed",
                stage=stage,
                error_message=safe_message,
            )
            assistant = self.storage.save_assistant_message(
                turn_id=turn["id"],
                content=safe_message,
                status="failed",
                standalone_query=standalone_query,
                rewritten_query=rewritten_query,
                retrieval_diagnostics=retrieval_diagnostics,
                token_usage=prepared.token_usage if prepared else {},
                sources=[],
                error_message=safe_message,
            )
            return ChatTurnResult(
                turn_id=turn["id"],
                user_message=self.storage.get_message(user_message.id) or user_message,
                assistant_message=assistant,
                task_state=task_state,
                task_state_change=task_change,
                sources=[],
                retrieved_chunks=[],
                standalone_query=standalone_query,
                rewritten_query=rewritten_query,
                retrieval_diagnostics=retrieval_diagnostics,
                timings=timings,
            )

    def retry_turn(
        self,
        turn_id: str,
        llm: LLMProvider,
        retrieval_config: RetrievalConfig,
        *,
        context_budget: ChatContextBudget | None = None,
        temperature: float = 0.1,
        progress_callback: ProgressCallback | None = None,
    ) -> ChatTurnResult:
        """Повторить ход с тем же вопросом, памятью и idempotency key."""
        turn = self.storage.get_turn(turn_id)
        if turn is None:
            raise ChatStorageError("Ход чата не найден.")
        user_message, _ = self.storage.get_turn_messages(turn_id)
        if user_message is None:
            raise ChatStorageError("Пользовательское сообщение хода не найдено.")
        return self.process_chat_message(
            turn["conversation_id"],
            user_message.content,
            llm,
            retrieval_config,
            idempotency_key=turn["idempotency_key"],
            context_budget=context_budget,
            temperature=temperature,
            progress_callback=progress_callback,
            force_regenerate=True,
        )

    def _restore_result(
        self,
        turn_id: str,
        user_message: ChatMessage | None,
        assistant_message: ChatMessage,
        conversation_id: str,
    ) -> ChatTurnResult:
        if user_message is None:
            raise ChatStorageError("Завершённый ход не содержит вопрос пользователя.")
        task_state = self.storage.latest_task_state(conversation_id)
        sources = self.storage.list_sources(assistant_message.id)
        return ChatTurnResult(
            turn_id=turn_id,
            user_message=user_message,
            assistant_message=assistant_message,
            task_state=task_state,
            task_state_change=_change_for_message(
                self.storage, conversation_id, user_message.id
            ),
            sources=sources,
            retrieved_chunks=[],
            standalone_query=assistant_message.standalone_query or user_message.content,
            rewritten_query=assistant_message.rewritten_query,
            retrieval_diagnostics=assistant_message.retrieval_diagnostics,
            timings={},
        )

    def _update_summary_if_needed(
        self,
        conversation_id: str,
        task_state: TaskState,
        llm: LLMProvider,
    ) -> None:
        conversation = self.storage.require_conversation(conversation_id)
        if not conversation.settings.get("automatic_summary", True):
            return
        messages = self.storage.list_messages(conversation_id)
        trigger = _settings_int(
            conversation.settings,
            "summary_trigger_messages",
            CHAT_SUMMARY_TRIGGER_MESSAGES,
            2,
        )
        if len(messages) <= trigger:
            return
        recent_count = _settings_int(
            conversation.settings,
            "recent_message_count",
            CHAT_RECENT_MESSAGE_COUNT,
            1,
        )
        current = self.storage.get_summary(conversation_id)
        pending = messages_to_summarize(
            messages,
            summarized_until_message_id=(
                current.summarized_until_message_id if current else None
            ),
            recent_message_count=recent_count,
        )
        if not pending:
            return
        previous = current.summary if current else ""
        updated = update_conversation_summary(previous, pending, task_state, llm)
        if updated != previous:
            self.storage.save_summary(conversation_id, updated, pending[-1].id)


def process_chat_message(
    conversation_id: str,
    user_text: str,
    llm: LLMProvider,
    retrieval_config: RetrievalConfig,
    *,
    database_path: Path = CHAT_DATABASE,
    candidate_searcher: CandidateSearcher | None = None,
    tokenizer: OffsetTokenizer | None = None,
    reranker: Reranker | None = None,
    idempotency_key: str | None = None,
) -> ChatTurnResult:
    """Совместимый функциональный интерфейс полного хода чата."""
    service = ChatService(
        ChatStorage(database_path),
        candidate_searcher=candidate_searcher,
        tokenizer=tokenizer,
        reranker=reranker,
    )
    return service.process_chat_message(
        conversation_id,
        user_text,
        llm,
        retrieval_config,
        idempotency_key=idempotency_key,
    )
