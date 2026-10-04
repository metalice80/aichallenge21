"""Транзакционное SQLite-хранилище диалогов, памяти, summary и источников."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .chat_models import (
    ChatMessage,
    Conversation,
    ConversationSummary,
    TaskState,
    TaskStateChange,
)
from .config import CHAT_DATABASE

CHAT_SCHEMA_VERSION = 1
_ALLOWED_TURN_STATUSES = {
    "pending",
    "updating_memory",
    "contextualizing",
    "retrieving",
    "generating",
    "completed",
    "failed",
    "cancelled",
}

_MIGRATION_1 = (
    """CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        strategy TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        settings_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS chat_turns (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        current_stage TEXT,
        error_message TEXT,
        memory_version INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        turn_id TEXT,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'completed',
        standalone_query TEXT,
        rewritten_query TEXT,
        retrieval_diagnostics_json TEXT NOT NULL DEFAULT '{}',
        token_usage_json TEXT NOT NULL DEFAULT '{}',
        error_message TEXT,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
        FOREIGN KEY (turn_id) REFERENCES chat_turns(id) ON DELETE SET NULL,
        UNIQUE(turn_id, role)
    )""",
    """CREATE TABLE IF NOT EXISTS message_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT NOT NULL,
        source_number INTEGER NOT NULL,
        chunk_id TEXT NOT NULL,
        source TEXT NOT NULL,
        title TEXT,
        section TEXT,
        page_number INTEGER,
        similarity REAL,
        rerank_score REAL,
        final_rank INTEGER,
        FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
        UNIQUE(message_id, source_number)
    )""",
    """CREATE TABLE IF NOT EXISTS task_state_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        state_json TEXT NOT NULL,
        change_summary_json TEXT NOT NULL DEFAULT '{}',
        change_source TEXT NOT NULL DEFAULT 'automatic',
        related_message_id TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
        FOREIGN KEY (related_message_id) REFERENCES chat_messages(id) ON DELETE SET NULL,
        UNIQUE(conversation_id, version)
    )""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_task_state_related_message
        ON task_state_versions(related_message_id) WHERE related_message_id IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS conversation_summaries (
        conversation_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        summarized_until_message_id TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
        FOREIGN KEY (summarized_until_message_id) REFERENCES chat_messages(id) ON DELETE SET NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_conversations_updated
        ON conversations(updated_at DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation
        ON chat_turns(conversation_id, created_at)""",
    """CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
        ON chat_messages(conversation_id, created_at)""",
    """CREATE INDEX IF NOT EXISTS idx_message_sources_message
        ON message_sources(message_id, source_number)""",
    """CREATE INDEX IF NOT EXISTS idx_task_state_conversation
        ON task_state_versions(conversation_id, version)""",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _atomic_write(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _state_from_payload(payload: dict[str, Any], *, version: int, updated_at: str) -> TaskState:
    return TaskState(
        goal=payload.get("goal"),
        constraints=list(payload.get("constraints", [])),
        terms=dict(payload.get("terms", {})),
        confirmed_facts=list(payload.get("confirmed_facts", [])),
        decisions=list(payload.get("decisions", [])),
        open_questions=list(payload.get("open_questions", [])),
        user_preferences=list(payload.get("user_preferences", [])),
        version=version,
        updated_at=updated_at,
    )


def _state_payload(state: TaskState) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "constraints": list(state.constraints),
        "terms": dict(state.terms),
        "confirmed_facts": list(state.confirmed_facts),
        "decisions": list(state.decisions),
        "open_questions": list(state.open_questions),
        "user_preferences": list(state.user_preferences),
    }


class ChatStorageError(RuntimeError):
    """Контролируемая ошибка постоянного хранилища чата."""


class ConversationNotFoundError(ChatStorageError):
    """Запрошенный диалог отсутствует."""


class TaskStateConflictError(ChatStorageError):
    """Память была изменена параллельно с текущей операцией."""


class ChatStorage:
    """Изолированное от FAISS-метаданных хранилище RAG-чата."""

    def __init__(self, database_path: Path = CHAT_DATABASE) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )"""
            )
            applied = {
                int(row[0])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in applied:
                for statement in _MIGRATION_1:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, _now()),
                )
            connection.execute(f"PRAGMA user_version = {CHAT_SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def schema_versions(self) -> list[int]:
        """Вернуть применённые версии миграций."""
        with self._connect() as connection:
            return [
                int(row[0])
                for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ]

    def create_conversation(
        self,
        *,
        title: str = "Новый диалог",
        model_name: str,
        strategy: str = "structural",
        settings: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Создать независимый диалог и вернуть сохранённую запись."""
        normalized_title = " ".join(title.split())[:80] or "Новый диалог"
        identifier = conversation_id or uuid.uuid4().hex
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO conversations (
                    id, title, created_at, updated_at, model_name, strategy, status,
                    settings_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?)""",
                (
                    identifier,
                    normalized_title,
                    timestamp,
                    timestamp,
                    model_name,
                    strategy,
                    _json(settings or {}),
                ),
            )
        conversation = self.get_conversation(identifier)
        assert conversation is not None
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Получить диалог по идентификатору."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return self._conversation_from_row(row) if row else None

    def require_conversation(self, conversation_id: str) -> Conversation:
        """Получить диалог либо сообщить о его отсутствии."""
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("Диалог не найден или уже удалён.")
        return conversation

    def list_conversations(self, *, include_archived: bool = False) -> list[Conversation]:
        """Вернуть диалоги в порядке последней активности."""
        query = "SELECT * FROM conversations"
        parameters: tuple[Any, ...] = ()
        if not include_archived:
            query += " WHERE status = ?"
            parameters = ("active",)
        query += " ORDER BY updated_at DESC, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._conversation_from_row(row) for row in rows]

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        model_name: str | None = None,
        strategy: str | None = None,
        status: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Conversation:
        """Обновить выбранные настройки диалога."""
        current = self.require_conversation(conversation_id)
        normalized_title = (
            (" ".join(title.split())[:80] or current.title) if title is not None else current.title
        )
        with self._connect() as connection:
            connection.execute(
                """UPDATE conversations SET title = ?, model_name = ?, strategy = ?,
                    status = ?, settings_json = ?, updated_at = ? WHERE id = ?""",
                (
                    normalized_title,
                    model_name or current.model_name,
                    strategy or current.strategy,
                    status or current.status,
                    _json(current.settings if settings is None else settings),
                    _now(),
                    conversation_id,
                ),
            )
        return self.require_conversation(conversation_id)

    def create_or_get_turn(
        self,
        conversation_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Создать ход один раз для заданного idempotency key."""
        self.require_conversation(conversation_id)
        key = idempotency_key or uuid.uuid4().hex
        timestamp = _now()
        turn_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO chat_turns (
                    id, conversation_id, idempotency_key, status, current_stage,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', 'pending', ?, ?)""",
                (turn_id, conversation_id, key, timestamp, timestamp),
            )
            row = connection.execute(
                "SELECT * FROM chat_turns WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return dict(row)

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        """Получить состояние хода."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_turns WHERE id = ?", (turn_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_turn(
        self,
        turn_id: str,
        status: str,
        *,
        stage: str | None = None,
        error_message: str | None = None,
        memory_version: int | None = None,
    ) -> None:
        """Сохранить этап обработки и безопасную ошибку хода."""
        if status not in _ALLOWED_TURN_STATUSES:
            raise ValueError(f"Неизвестный статус хода: {status}")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE chat_turns SET status = ?, current_stage = ?, error_message = ?,
                    memory_version = COALESCE(?, memory_version), updated_at = ? WHERE id = ?""",
                (status, stage or status, error_message, memory_version, _now(), turn_id),
            )
            if cursor.rowcount != 1:
                raise ChatStorageError("Ход чата не найден.")

    def save_user_message(self, turn_id: str, content: str) -> ChatMessage:
        """Идемпотентно сохранить единственное пользовательское сообщение хода."""
        normalized = content.strip()
        if not normalized:
            raise ValueError("Сообщение не может быть пустым.")
        turn = self.get_turn(turn_id)
        if turn is None:
            raise ChatStorageError("Ход чата не найден.")
        message_id = f"{turn_id}:user"
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO chat_messages (
                    id, conversation_id, turn_id, role, content, created_at, status
                ) VALUES (?, ?, ?, 'user', ?, ?, 'pending')""",
                (message_id, turn["conversation_id"], turn_id, normalized, _now()),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), turn["conversation_id"]),
            )
        message = self.get_message(message_id)
        assert message is not None
        return message

    def update_message_status(
        self,
        message_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        """Обновить статус существующего сообщения."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_messages SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, message_id),
            )

    def save_assistant_message(
        self,
        *,
        turn_id: str,
        content: str,
        status: str,
        standalone_query: str,
        rewritten_query: str | None,
        retrieval_diagnostics: dict[str, Any],
        token_usage: dict[str, int],
        sources: Iterable[dict[str, Any]],
        error_message: str | None = None,
    ) -> ChatMessage:
        """Атомарно заменить ответ хода и его источники без дублей."""
        turn = self.get_turn(turn_id)
        if turn is None:
            raise ChatStorageError("Ход чата не найден.")
        message_id = f"{turn_id}:assistant"
        timestamp = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO chat_messages (
                    id, conversation_id, turn_id, role, content, created_at, status,
                    standalone_query, rewritten_query, retrieval_diagnostics_json,
                    token_usage_json, error_message
                ) VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    status = excluded.status,
                    standalone_query = excluded.standalone_query,
                    rewritten_query = excluded.rewritten_query,
                    retrieval_diagnostics_json = excluded.retrieval_diagnostics_json,
                    token_usage_json = excluded.token_usage_json,
                    error_message = excluded.error_message""",
                (
                    message_id,
                    turn["conversation_id"],
                    turn_id,
                    content.strip(),
                    timestamp,
                    status,
                    standalone_query,
                    rewritten_query,
                    _json(retrieval_diagnostics),
                    _json(token_usage),
                    error_message,
                ),
            )
            connection.execute("DELETE FROM message_sources WHERE message_id = ?", (message_id,))
            for source in sources:
                connection.execute(
                    """INSERT INTO message_sources (
                        message_id, source_number, chunk_id, source, title, section,
                        page_number, similarity, rerank_score, final_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        message_id,
                        int(source["number"]),
                        str(source["chunk_id"]),
                        str(source["source"]),
                        source.get("title"),
                        source.get("section"),
                        source.get("page_number"),
                        source.get("similarity"),
                        source.get("rerank_score"),
                        source.get("final_rank"),
                    ),
                )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, turn["conversation_id"]),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        message = self.get_message(message_id)
        assert message is not None
        return message

    def get_message(self, message_id: str) -> ChatMessage | None:
        """Получить одно сообщение."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_messages WHERE id = ?", (message_id,)
            ).fetchone()
        return self._message_from_row(row) if row else None

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        """Восстановить сообщения в детерминированном порядке."""
        self.require_conversation(conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM chat_messages WHERE conversation_id = ?
                    ORDER BY created_at, rowid""",
                (conversation_id,),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def get_turn_messages(self, turn_id: str) -> tuple[ChatMessage | None, ChatMessage | None]:
        """Вернуть пользовательское и ассистентское сообщение хода."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE turn_id = ? ORDER BY rowid", (turn_id,)
            ).fetchall()
        messages = {row["role"]: self._message_from_row(row) for row in rows}
        return messages.get("user"), messages.get("assistant")

    def list_sources(self, message_id: str) -> list[dict[str, Any]]:
        """Вернуть snapshot источников конкретного ответа."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT source_number AS number, chunk_id, source, title, section,
                    page_number, similarity, rerank_score, final_rank
                    FROM message_sources WHERE message_id = ? ORDER BY source_number""",
                (message_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_task_state(self, conversation_id: str) -> TaskState:
        """Вернуть последнюю корректную память или пустую версию 0."""
        self.require_conversation(conversation_id)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM task_state_versions WHERE conversation_id = ?
                    ORDER BY version DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
        return self._state_from_row(row) if row else TaskState()

    def get_task_state_version(self, conversation_id: str, version: int) -> TaskState | None:
        """Получить конкретную версию памяти."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM task_state_versions
                    WHERE conversation_id = ? AND version = ?""",
                (conversation_id, version),
            ).fetchone()
        return self._state_from_row(row) if row else None

    def save_task_state(
        self,
        conversation_id: str,
        state: TaskState,
        change: TaskStateChange,
        *,
        source: str,
        related_message_id: str | None = None,
        expected_version: int | None = None,
    ) -> TaskState:
        """Создать следующую версию памяти с optimistic-lock проверкой."""
        self.require_conversation(conversation_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT version FROM task_state_versions WHERE conversation_id = ?
                    ORDER BY version DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            current_version = int(row[0]) if row else 0
            if expected_version is not None and current_version != expected_version:
                raise TaskStateConflictError(
                    f"Ожидалась версия памяти {expected_version}, актуальна {current_version}."
                )
            if related_message_id:
                existing = connection.execute(
                    "SELECT * FROM task_state_versions WHERE related_message_id = ?",
                    (related_message_id,),
                ).fetchone()
                if existing:
                    connection.rollback()
                    return self._state_from_row(existing)
            version = current_version + 1
            timestamp = _now()
            connection.execute(
                """INSERT INTO task_state_versions (
                    conversation_id, version, state_json, change_summary_json,
                    change_source, related_message_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    version,
                    _json(_state_payload(state)),
                    _json(change.to_dict()),
                    source,
                    related_message_id,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
            connection.commit()
            return _state_from_payload(_state_payload(state), version=version, updated_at=timestamp)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def list_task_state_versions(self, conversation_id: str) -> list[dict[str, Any]]:
        """Вернуть версии памяти и сохранённые изменения."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT version, state_json, change_summary_json, change_source,
                    related_message_id, created_at FROM task_state_versions
                    WHERE conversation_id = ? ORDER BY version DESC""",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "version": int(row["version"]),
                "state": _decode(row["state_json"], {}),
                "change": _decode(row["change_summary_json"], {}),
                "source": row["change_source"],
                "related_message_id": row["related_message_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_manual_task_state(
        self,
        conversation_id: str,
        state: TaskState,
        change: TaskStateChange,
    ) -> TaskState:
        """Сохранить проверенное ручное исправление новой версией."""
        current = self.latest_task_state(conversation_id)
        return self.save_task_state(
            conversation_id,
            state,
            change,
            source="manual",
            expected_version=current.version,
        )

    def restore_task_state(self, conversation_id: str, version: int) -> TaskState:
        """Восстановить старое содержимое как новую, аудируемую версию."""
        target = self.get_task_state_version(conversation_id, version)
        if target is None:
            raise ChatStorageError("Выбранная версия памяти не найдена.")
        current = self.latest_task_state(conversation_id)
        change = TaskStateChange(
            changed={"restored_version": {"from": current.version, "to": version}},
            reason=f"Восстановлена версия {version}.",
        )
        return self.save_task_state(
            conversation_id,
            target,
            change,
            source="restore",
            expected_version=current.version,
        )

    def clear_task_state(self, conversation_id: str) -> TaskState:
        """Очистить память новой версией, не удаляя сообщения."""
        current = self.latest_task_state(conversation_id)
        empty = TaskState()
        removed = {
            key: value
            for key, value in _state_payload(current).items()
            if value not in (None, [], {})
        }
        return self.save_task_state(
            conversation_id,
            empty,
            TaskStateChange(removed=removed, reason="Память очищена пользователем."),
            source="clear",
            expected_version=current.version,
        )

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        """Получить conversation summary и его границу."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return ConversationSummary(
            conversation_id=row["conversation_id"],
            summary=row["summary"],
            summarized_until_message_id=row["summarized_until_message_id"],
            updated_at=row["updated_at"],
        )

    def save_summary(
        self,
        conversation_id: str,
        summary: str,
        summarized_until_message_id: str,
    ) -> ConversationSummary:
        """Атомарно заменить summary и последнюю обработанную границу."""
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO conversation_summaries (
                    conversation_id, summary, summarized_until_message_id, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary = excluded.summary,
                    summarized_until_message_id = excluded.summarized_until_message_id,
                    updated_at = excluded.updated_at""",
                (conversation_id, summary.strip(), summarized_until_message_id, timestamp),
            )
        result = self.get_summary(conversation_id)
        assert result is not None
        return result

    def clear_messages(self, conversation_id: str) -> int:
        """Удалить ходы, сообщения, источники и summary, сохранив память."""
        self.require_conversation(conversation_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
            connection.execute(
                "DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,)
            )
            connection.execute(
                "DELETE FROM chat_turns WHERE conversation_id = ?", (conversation_id,)
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), conversation_id),
            )
            connection.commit()
            return count
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def delete_conversation(self, conversation_id: str) -> bool:
        """Каскадно удалить только данные выбранного диалога."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount == 1

    def export_payload(
        self,
        conversation_id: str,
        *,
        include_memory_history: bool = True,
    ) -> dict[str, Any]:
        """Собрать полный JSON-совместимый экспорт диалога."""
        conversation = self.require_conversation(conversation_id)
        messages = self.list_messages(conversation_id)
        summary = self.get_summary(conversation_id)
        return {
            "version": 1,
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "model_name": conversation.model_name,
                "strategy": conversation.strategy,
                "status": conversation.status,
                "settings": conversation.settings,
            },
            "messages": [
                {
                    "id": message.id,
                    "turn_id": message.turn_id,
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                    "status": message.status,
                    "standalone_query": message.standalone_query,
                    "rewritten_query": message.rewritten_query,
                    "retrieval_diagnostics": message.retrieval_diagnostics,
                    "token_usage": message.token_usage,
                    "error_message": message.error_message,
                    "sources": self.list_sources(message.id),
                }
                for message in messages
            ],
            "task_state": self.latest_task_state(conversation_id).to_dict(),
            "task_state_versions": (
                self.list_task_state_versions(conversation_id)
                if include_memory_history
                else []
            ),
            "conversation_summary": (
                {
                    "summary": summary.summary,
                    "summarized_until_message_id": summary.summarized_until_message_id,
                    "updated_at": summary.updated_at,
                }
                if summary
                else None
            ),
        }

    def render_markdown(self, conversation_id: str) -> str:
        """Сформировать читаемый Markdown-экспорт с источниками ответов."""
        payload = self.export_payload(conversation_id, include_memory_history=False)
        state = payload["task_state"]
        lines = [f"# {payload['conversation']['title']}", "", "## Память задачи", ""]
        lines.extend(
            [
                f"- **Цель:** {state.get('goal') or '—'}",
                "- **Ограничения:** " + ("; ".join(state.get("constraints", [])) or "—"),
                "- **Термины:** "
                + (
                    "; ".join(f"{key} = {value}" for key, value in state.get("terms", {}).items())
                    or "—"
                ),
                "",
                "## Диалог",
                "",
            ]
        )
        for message in payload["messages"]:
            heading = "Пользователь" if message["role"] == "user" else "Ассистент"
            lines.extend([f"### {heading}", "", message["content"], ""])
            if message["role"] == "assistant":
                lines.append("**Источники:**")
                sources = message["sources"]
                if sources:
                    for source in sources:
                        location = source.get("section") or source.get("title") or "без раздела"
                        lines.append(f"{source['number']}. {source['source']} — {location}")
                else:
                    lines.append("Релевантные источники в локальной базе не найдены.")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write_exports(
        self,
        conversation_id: str,
        directory: Path,
        *,
        include_memory_history: bool = True,
    ) -> tuple[Path, Path]:
        """Атомарно записать JSON- и Markdown-экспорт в заданный каталог."""
        directory = Path(directory)
        payload = self.export_payload(
            conversation_id, include_memory_history=include_memory_history
        )
        json_path = directory / f"conversation-{conversation_id}.json"
        markdown_path = directory / f"conversation-{conversation_id}.md"
        _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
        _atomic_write(markdown_path, self.render_markdown(conversation_id))
        return json_path, markdown_path

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model_name=row["model_name"],
            strategy=row["strategy"],
            status=row["status"],
            settings=dict(_decode(row["settings_json"], {})),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> ChatMessage:
        return ChatMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            turn_id=row["turn_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            status=row["status"],
            standalone_query=row["standalone_query"],
            rewritten_query=row["rewritten_query"],
            retrieval_diagnostics=dict(_decode(row["retrieval_diagnostics_json"], {})),
            token_usage={
                str(key): int(value)
                for key, value in dict(_decode(row["token_usage_json"], {})).items()
            },
            error_message=row["error_message"],
        )

    @staticmethod
    def _state_from_row(row: sqlite3.Row) -> TaskState:
        return _state_from_payload(
            dict(_decode(row["state_json"], {})),
            version=int(row["version"]),
            updated_at=row["created_at"],
        )
