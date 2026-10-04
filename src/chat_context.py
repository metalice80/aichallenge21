"""Сборка независимых блоков чата с измеряемым токенным бюджетом."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .chat_models import ChatContextBudget, ChatMessage, TaskState
from .models import RetrievedChunk
from .rag import build_context
from .text_utils import OffsetTokenizer, RegexTokenizer


@dataclass(slots=True)
class PreparedChatContext:
    """Готовые блоки промпта и фактически использованные RAG-чанки."""

    task_state_block: str
    summary_block: str
    recent_messages_block: str
    rag_context_block: str
    used_chunks: list[RetrievedChunk]
    token_usage: dict[str, int]


def count_tokens(text: str, tokenizer: OffsetTokenizer) -> int:
    """Посчитать токены тем же интерфейсом, который применяет индексатор."""
    return len(tokenizer.token_spans(text))


def _truncate(text: str, limit: int, tokenizer: OffsetTokenizer) -> str:
    if limit <= 0 or not text:
        return ""
    spans = tokenizer.token_spans(text)
    if len(spans) <= limit:
        return text
    return text[: spans[limit - 1].end].rstrip()


def _task_state_block(
    state: TaskState,
    limit: int,
    tokenizer: OffsetTokenizer,
) -> str:
    core: dict[str, Any] = {
        "goal": state.goal,
        "constraints": list(state.constraints),
    }
    rendered = json.dumps(core, ensure_ascii=False, indent=2)
    if count_tokens(rendered, tokenizer) > limit:
        raise ValueError(
            "Цель и активные ограничения не помещаются в бюджет памяти; "
            "сократите память вручную."
        )
    optional = (
        ("terms", dict(state.terms)),
        ("decisions", list(state.decisions)),
        ("confirmed_facts", list(state.confirmed_facts)),
        ("open_questions", list(state.open_questions)),
        ("user_preferences", list(state.user_preferences)),
    )
    for key, value in optional:
        candidate = dict(core)
        candidate[key] = value
        candidate_rendered = json.dumps(candidate, ensure_ascii=False, indent=2)
        if count_tokens(candidate_rendered, tokenizer) <= limit:
            core = candidate
            rendered = candidate_rendered
    return rendered


def _recent_messages_block(
    messages: list[ChatMessage],
    limit: int,
    tokenizer: OffsetTokenizer,
) -> str:
    selected: list[str] = []
    used = 0
    for message in reversed(messages):
        block = f"[{message.role}]\n{message.content}"
        block_tokens = count_tokens(block, tokenizer)
        if block_tokens > limit:
            continue
        separator_tokens = 1 if selected else 0
        if used + separator_tokens + block_tokens > limit:
            continue
        selected.append(block)
        used += separator_tokens + block_tokens
    selected.reverse()
    return "\n\n".join(selected)


def prepare_chat_context(
    *,
    task_state: TaskState,
    conversation_summary: str,
    recent_messages: list[ChatMessage],
    retrieved_chunks: list[RetrievedChunk],
    current_question: str,
    budget: ChatContextBudget,
    tokenizer: OffsetTokenizer | None = None,
) -> PreparedChatContext:
    """Подготовить блоки, сохраняя вопрос и исключая целые нерелевантные чанки."""
    active_tokenizer = tokenizer or RegexTokenizer()
    task_block = _task_state_block(
        task_state, budget.task_state_tokens, active_tokenizer
    )
    summary_block = _truncate(
        conversation_summary, budget.summary_tokens, active_tokenizer
    )
    recent_block = _recent_messages_block(
        recent_messages, budget.recent_messages_tokens, active_tokenizer
    )
    rag_block, used_chunks = build_context(
        retrieved_chunks,
        budget.rag_context_tokens,
        tokenizer=active_tokenizer,
    )
    token_usage = {
        "task_state": count_tokens(task_block, active_tokenizer),
        "conversation_summary": count_tokens(summary_block, active_tokenizer),
        "recent_messages": count_tokens(recent_block, active_tokenizer),
        "rag_context": count_tokens(rag_block, active_tokenizer),
        "current_question": count_tokens(current_question, active_tokenizer),
        "response_reserve": budget.response_reserve_tokens,
    }
    token_usage["input_total"] = sum(
        token_usage[key]
        for key in (
            "task_state",
            "conversation_summary",
            "recent_messages",
            "rag_context",
            "current_question",
        )
    )
    token_usage["budget_total"] = (
        budget.task_state_tokens
        + budget.summary_tokens
        + budget.recent_messages_tokens
        + budget.rag_context_tokens
        + token_usage["current_question"]
    )
    return PreparedChatContext(
        task_state_block=task_block,
        summary_block=summary_block,
        recent_messages_block=recent_block,
        rag_context_block=rag_block,
        used_chunks=used_chunks,
        token_usage=token_usage,
    )
