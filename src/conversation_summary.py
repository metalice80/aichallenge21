"""Инкрементальное summary старой части диалога без повторной обработки сообщений."""

from __future__ import annotations

from .chat_models import ChatMessage, TaskState
from .chat_prompts import build_summary_messages
from .llm import LLMProvider


def messages_to_summarize(
    messages: list[ChatMessage],
    *,
    summarized_until_message_id: str | None,
    recent_message_count: int,
) -> list[ChatMessage]:
    """Выбрать ещё не суммированные сообщения до окна recent messages."""
    if recent_message_count < 1:
        raise ValueError("Число recent messages должно быть больше нуля.")
    eligible = messages[:-recent_message_count] if len(messages) > recent_message_count else []
    if not eligible:
        return []
    if summarized_until_message_id is None:
        return eligible
    for index, message in enumerate(eligible):
        if message.id == summarized_until_message_id:
            return eligible[index + 1 :]
    return eligible


def update_conversation_summary(
    previous_summary: str,
    messages_to_summarize: list[ChatMessage],
    task_state: TaskState,
    llm: LLMProvider,
) -> str:
    """Обновить summary только новыми сообщениями; при ошибке вернуть прежнее."""
    if not messages_to_summarize:
        return previous_summary
    try:
        summary = llm.generate(
            build_summary_messages(previous_summary, messages_to_summarize, task_state),
            0.0,
        ).strip()
        if not summary:
            raise ValueError("LLM вернула пустое summary.")
        if len(summary) > 8000:
            raise ValueError("Summary превышает безопасный размер.")
        return summary
    except Exception:
        return previous_summary
