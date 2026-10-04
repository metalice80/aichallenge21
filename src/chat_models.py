"""Типы данных постоянного RAG-чата и структурированной памяти задачи."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Conversation:
    """Постоянный диалог с собственными retrieval-настройками."""

    id: str
    title: str
    created_at: str
    updated_at: str
    model_name: str
    strategy: str
    status: str = "active"
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatMessage:
    """Сообщение чата и сохранённая диагностика его обработки."""

    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str
    status: str = "completed"
    turn_id: str | None = None
    standalone_query: str | None = None
    rewritten_query: str | None = None
    retrieval_diagnostics: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(slots=True)
class TaskState:
    """Подтверждённое пользователем состояние текущей задачи."""

    goal: str | None = None
    constraints: list[str] = field(default_factory=list)
    terms: dict[str, str] = field(default_factory=dict)
    confirmed_facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    user_preferences: list[str] = field(default_factory=list)
    version: int = 0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Вернуть JSON-совместимое представление состояния."""
        return asdict(self)


@dataclass(slots=True)
class TaskStateChange:
    """Проверенная разница между двумя версиями памяти."""

    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)
    changed: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    @property
    def has_changes(self) -> bool:
        """Проверить наличие содержательных изменений."""
        return bool(self.added or self.removed or self.changed)

    def to_dict(self) -> dict[str, Any]:
        """Вернуть JSON-совместимое представление изменений."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChatContextBudget:
    """Независимые токенные бюджеты блоков промпта чата."""

    task_state_tokens: int = 800
    summary_tokens: int = 1000
    recent_messages_tokens: int = 1500
    rag_context_tokens: int = 4000
    response_reserve_tokens: int = 1500

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"Бюджет {name} не может быть отрицательным.")
        if self.rag_context_tokens == 0:
            raise ValueError("Бюджет RAG-контекста должен быть больше нуля.")


@dataclass(slots=True)
class ConversationSummary:
    """Сводка старой части диалога и граница уже обработанных сообщений."""

    conversation_id: str
    summary: str
    summarized_until_message_id: str | None
    updated_at: str


@dataclass(slots=True)
class ContextualizationResult:
    """Standalone query вместе с диагностикой безопасного fallback."""

    standalone_query: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class ChatTurnResult:
    """Полный результат одного идемпотентного хода чата."""

    turn_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage
    task_state: TaskState
    task_state_change: TaskStateChange
    sources: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    standalone_query: str
    rewritten_query: str | None
    retrieval_diagnostics: dict[str, Any]
    timings: dict[str, float]
