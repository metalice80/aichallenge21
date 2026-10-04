from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.chat_models import TaskState, TaskStateChange
from src.chat_storage import ChatStorage


def test_migration_is_idempotent_and_preserves_existing_data(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy_data(value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_data VALUES ('keep-me')")

    first = ChatStorage(database)
    second = ChatStorage(database)

    assert first.schema_versions() == [1]
    assert second.schema_versions() == [1]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM legacy_data").fetchone()[0] == "keep-me"


def test_conversation_messages_memory_and_summary_survive_reconnect(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite"
    storage = ChatStorage(database)
    conversation = storage.create_conversation(
        title="Проверка",
        model_name="fake",
        strategy="structural",
    )
    turn = storage.create_or_get_turn(conversation.id, idempotency_key="turn-1")
    user = storage.save_user_message(turn["id"], "Моя цель — настроить RAG")
    saved_state = storage.save_task_state(
        conversation.id,
        TaskState(goal="Настроить RAG"),
        TaskStateChange(added={"goal": "Настроить RAG"}),
        source="automatic",
        related_message_id=user.id,
        expected_version=0,
    )
    storage.update_turn(turn["id"], "generating", memory_version=saved_state.version)
    assistant = storage.save_assistant_message(
        turn_id=turn["id"],
        content="Используйте structural index [1]",
        status="completed",
        standalone_query="Как настроить RAG?",
        rewritten_query="настройка RAG structural index",
        retrieval_diagnostics={"final_count": 1},
        token_usage={"rag_context": 12},
        sources=[
            {
                "number": 1,
                "chunk_id": "chunk-1",
                "source": "guide.md",
                "title": "Guide",
                "section": "RAG",
                "page_number": None,
                "similarity": 0.9,
                "rerank_score": 0.8,
                "final_rank": 1,
            }
        ],
    )
    storage.update_message_status(user.id, "completed")
    storage.update_turn(turn["id"], "completed")
    storage.save_summary(conversation.id, "Пользователь настраивает RAG.", assistant.id)

    reopened = ChatStorage(database)
    messages = reopened.list_messages(conversation.id)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == [
        "Моя цель — настроить RAG",
        "Используйте structural index [1]",
    ]
    assert reopened.latest_task_state(conversation.id).goal == "Настроить RAG"
    assert reopened.latest_task_state(conversation.id).version == 1
    assert reopened.get_summary(conversation.id).summarized_until_message_id == assistant.id
    assert reopened.list_sources(assistant.id)[0]["chunk_id"] == "chunk-1"


def test_idempotent_turn_and_assistant_replace_do_not_duplicate(tmp_path: Path) -> None:
    storage = ChatStorage(tmp_path / "chat.sqlite")
    conversation = storage.create_conversation(model_name="fake")
    first = storage.create_or_get_turn(conversation.id, idempotency_key="same-key")
    second = storage.create_or_get_turn(conversation.id, idempotency_key="same-key")
    user = storage.save_user_message(first["id"], "Вопрос")
    storage.save_user_message(second["id"], "Вопрос")

    for answer in ("Первый [1]", "Повторный [1]"):
        storage.save_assistant_message(
            turn_id=first["id"],
            content=answer,
            status="completed",
            standalone_query="Вопрос",
            rewritten_query=None,
            retrieval_diagnostics={},
            token_usage={},
            sources=[
                {
                    "number": 1,
                    "chunk_id": "one",
                    "source": "guide.md",
                    "title": None,
                    "section": None,
                    "page_number": None,
                    "similarity": 0.8,
                    "rerank_score": 0.7,
                    "final_rank": 1,
                }
            ],
        )

    assert first["id"] == second["id"]
    assert len(storage.list_messages(conversation.id)) == 2
    assert storage.list_messages(conversation.id)[0].id == user.id
    assistant = storage.list_messages(conversation.id)[1]
    assert assistant.content == "Повторный [1]"
    assert len(storage.list_sources(assistant.id)) == 1


def test_memory_restore_manual_edit_and_independent_clears(tmp_path: Path) -> None:
    storage = ChatStorage(tmp_path / "chat.sqlite")
    conversation = storage.create_conversation(model_name="fake")
    version_one = storage.save_task_state(
        conversation.id,
        TaskState(goal="Цель", constraints=["CPU"]),
        TaskStateChange(added={"goal": "Цель", "constraints": ["CPU"]}),
        source="manual",
        expected_version=0,
    )
    version_two = storage.save_manual_task_state(
        conversation.id,
        TaskState(goal="Новая цель", constraints=["CPU"]),
        TaskStateChange(changed={"goal": {"from": "Цель", "to": "Новая цель"}}),
    )
    restored = storage.restore_task_state(conversation.id, version_one.version)
    turn = storage.create_or_get_turn(conversation.id)
    storage.save_user_message(turn["id"], "Сообщение")

    assert version_two.version == 2
    assert restored.version == 3
    assert restored.goal == "Цель"
    assert storage.clear_messages(conversation.id) == 1
    assert storage.list_messages(conversation.id) == []
    assert storage.latest_task_state(conversation.id).version == 3
    cleared = storage.clear_task_state(conversation.id)
    assert cleared.version == 4
    assert cleared.goal is None


def test_exports_and_delete_are_scoped_to_selected_conversation(tmp_path: Path) -> None:
    storage = ChatStorage(tmp_path / "chat.sqlite")
    first = storage.create_conversation(title="Первый", model_name="fake")
    second = storage.create_conversation(title="Второй", model_name="fake")
    turn = storage.create_or_get_turn(first.id)
    storage.save_user_message(turn["id"], "Вопрос")
    assistant = storage.save_assistant_message(
        turn_id=turn["id"],
        content="Источников нет.",
        status="completed",
        standalone_query="Вопрос",
        rewritten_query=None,
        retrieval_diagnostics={},
        token_usage={},
        sources=[],
    )
    json_path, markdown_path = storage.write_exports(first.id, tmp_path / "exports")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["conversation"]["id"] == first.id
    assert payload["messages"][1]["id"] == assistant.id
    assert "Релевантные источники" in markdown_path.read_text(encoding="utf-8")
    assert storage.delete_conversation(first.id)
    assert storage.get_conversation(first.id) is None
    assert storage.get_conversation(second.id) is not None
