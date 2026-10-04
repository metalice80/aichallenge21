"""Преобразование зависимого follow-up в самостоятельный поисковый запрос."""

from __future__ import annotations

import re
import time

from .chat_models import ChatMessage, ContextualizationResult, TaskState
from .chat_prompts import build_contextualization_messages
from .llm import LLMProvider

_TECHNICAL_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+|"
    r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|--?[A-Za-z][A-Za-z0-9_-]*|"
    r"[A-Za-z_][A-Za-z0-9_]{2,}|\b[A-Z][A-Za-z0-9]+[A-Z][A-Za-z0-9]+\b)"
)


def _validate_query(query: str, current_message: str) -> str:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("LLM вернула пустой standalone query.")
    if len(normalized) > 700 or len(normalized.split()) > 100:
        raise ValueError("LLM вернула ответ вместо краткого standalone query.")
    if normalized.startswith(("{", "[", "```")):
        raise ValueError("Standalone query содержит служебную разметку.")
    missing = [
        value
        for value in _TECHNICAL_RE.findall(current_message)
        if value not in normalized
    ]
    if missing:
        raise ValueError(
            "Standalone query изменил технические идентификаторы: " + ", ".join(missing)
        )
    return normalized


def contextualize_question_result(
    current_message: str,
    task_state: TaskState,
    recent_messages: list[ChatMessage],
    conversation_summary: str,
    llm: LLMProvider,
) -> ContextualizationResult:
    """Вернуть standalone query и безопасную диагностику fallback."""
    original = current_message.strip()
    if not original:
        raise ValueError("Сообщение не может быть пустым.")
    started = time.perf_counter()
    try:
        raw = llm.generate(
            build_contextualization_messages(
                original,
                task_state,
                recent_messages,
                conversation_summary,
            ),
            0.0,
        )
        query = _validate_query(raw, original)
        return ContextualizationResult(
            standalone_query=query,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return ContextualizationResult(
            standalone_query=original,
            fallback_used=True,
            fallback_reason=f"Contextualization не применена: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )


def contextualize_question(
    current_message: str,
    task_state: TaskState,
    recent_messages: list[ChatMessage],
    conversation_summary: str,
    llm: LLMProvider,
) -> str:
    """Вернуть standalone query; при любой ошибке сохранить исходное сообщение."""
    return contextualize_question_result(
        current_message,
        task_state,
        recent_messages,
        conversation_summary,
        llm,
    ).standalone_query
