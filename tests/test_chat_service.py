from __future__ import annotations

import json
from pathlib import Path

from src.chat_service import ChatService
from src.chat_storage import ChatStorage
from src.models import RetrievalConfig
from src.text_utils import RegexTokenizer


class ChatLLM:
    model_name = "chat-fake"

    def __init__(self, answer_text=None) -> None:
        self.calls = []
        self.answer_number = 0
        self.answer_text = answer_text

    def generate(self, messages, temperature=0.1):
        self.calls.append(messages)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Кратко обнови резюме" in system:
            return "Цель — настроить локальный RAG; используется CPU."
        if "структурированную память" in system:
            goal = "Настроить локальный RAG" if "настроить" in user.casefold() else None
            constraints = ["Только CPU"] if "cpu" in user.casefold() else []
            if "current_state" in user and goal is None:
                current = json.loads(user.split("<current_state>\n", 1)[1].split("\n</current_state>", 1)[0])
                goal = current["goal"]
                constraints = current["constraints"]
            return json.dumps(
                {
                    "goal": goal,
                    "constraints": constraints,
                    "terms": {},
                    "confirmed_facts": [],
                    "decisions": [],
                    "open_questions": [],
                    "user_preferences": [],
                    "changes": {"added": {}, "removed": {}, "changed": {}},
                },
                ensure_ascii=False,
            )
        if "самостоятельный поисковый запрос" in system:
            if "его до трёх" in user:
                return "Можно ли уменьшить Final Top-K локального RAG до трёх?"
            return user.split("<current_message>\n", 1)[1].split("\n</current_message>", 1)[0]
        if "Преобразуй вопрос" in system:
            return json.dumps(
                {
                    "search_query": "настройка локального RAG CPU",
                    "alternative_queries": [],
                },
                ensure_ascii=False,
            )
        if "ассистент по локальной базе" in system:
            self.answer_number += 1
            return self.answer_text or f"Ответ версии {self.answer_number} подтверждён [1]"
        raise AssertionError(f"Неизвестный prompt: {system}")


class Searcher:
    tokenizer = RegexTokenizer()

    def __init__(self, fail=False, only_low=False) -> None:
        self.calls = []
        self.fail = fail
        self.only_low = only_low

    def search(self, query: str, strategy: str, top_k: int):
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("индекс временно недоступен")
        rows = [
            {
                "chunk_id": "rejected",
                "source": "wrong.md",
                "title": "Wrong",
                "section": None,
                "page_number": None,
                "content": "REJECTED SECRET",
                "score": 0.1,
                "token_count": 2,
            }
        ]
        if not self.only_low:
            rows.insert(
                0,
                {
                    "chunk_id": "selected",
                    "source": "guide.md",
                    "title": "Guide",
                    "section": "RAG",
                    "page_number": 2,
                    "content": "Используйте structural index и локальную модель на CPU.",
                    "score": 0.9,
                    "token_count": 9,
                },
            )
        return rows[:top_k]


def config() -> RetrievalConfig:
    return RetrievalConfig(
        candidate_top_k=5,
        final_top_k=3,
        similarity_threshold=0.35,
        max_chunks_per_source=3,
    )


def make_service(tmp_path: Path, searcher: Searcher | None = None):
    storage = ChatStorage(tmp_path / "chat.sqlite")
    active_searcher = searcher or Searcher()
    service = ChatService(
        storage,
        candidate_searcher=active_searcher,
        tokenizer=active_searcher.tokenizer,
    )
    conversation = storage.create_conversation(
        model_name="chat-fake",
        strategy="structural",
        settings={"recent_message_count": 2, "summary_trigger_messages": 2},
    )
    return storage, service, conversation, active_searcher


def test_full_turn_uses_status_chain_selected_chunks_and_sources(tmp_path: Path) -> None:
    storage, service, conversation, searcher = make_service(tmp_path)
    llm = ChatLLM()
    statuses = []

    result = service.process_chat_message(
        conversation.id,
        "Цель — настроить локальный RAG только на CPU.",
        llm,
        config(),
        idempotency_key="first-turn",
        progress_callback=statuses.append,
    )

    assert statuses == [
        "updating_memory",
        "contextualizing",
        "retrieving",
        "generating",
    ]
    assert result.assistant_message.status == "completed"
    assert result.task_state.goal == "Настроить локальный RAG"
    assert result.task_state.constraints == ["Только CPU"]
    assert result.sources[0]["chunk_id"] == "selected"
    assert storage.list_sources(result.assistant_message.id)[0]["source"] == "guide.md"
    generation = next(
        call for call in llm.calls if "ассистент по локальной базе" in call[0]["content"]
    )
    prompt = generation[-1]["content"]
    assert "Используйте structural index" in prompt
    assert "REJECTED SECRET" not in prompt
    assert searcher.calls[0] == result.standalone_query
    assert result.retrieval_diagnostics["final_count"] == 1


def test_follow_up_contextualization_and_summary_survive_reopen(tmp_path: Path) -> None:
    storage, service, conversation, _ = make_service(tmp_path)
    llm = ChatLLM()
    service.process_chat_message(
        conversation.id,
        "Цель — настроить локальный RAG только на CPU.",
        llm,
        config(),
    )
    result = service.process_chat_message(
        conversation.id,
        "А можно уменьшить его до трёх?",
        llm,
        config(),
    )

    assert result.standalone_query == "Можно ли уменьшить Final Top-K локального RAG до трёх?"
    reopened = ChatStorage(storage.database_path)
    assert len(reopened.list_messages(conversation.id)) == 4
    assert reopened.latest_task_state(conversation.id).goal == "Настроить локальный RAG"
    summary = reopened.get_summary(conversation.id)
    assert summary is not None
    assert summary.summarized_until_message_id == result.user_message.id or summary.summary
    assert len(storage.list_conversations()) == 1
    assert {
        message.conversation_id
        for message in reopened.list_messages(conversation.id)
    } == {conversation.id}


def test_retry_reuses_user_message_memory_version_and_sources(tmp_path: Path) -> None:
    storage, service, conversation, _ = make_service(tmp_path)
    llm = ChatLLM()
    first = service.process_chat_message(
        conversation.id,
        "Цель — настроить локальный RAG только на CPU.",
        llm,
        config(),
        idempotency_key="retry-me",
    )
    version = first.task_state.version

    retried = service.retry_turn(first.turn_id, llm, config())

    assert retried.assistant_message.content == "Ответ версии 2 подтверждён [1]"
    assert len(storage.list_messages(conversation.id)) == 2
    assert storage.latest_task_state(conversation.id).version == version
    assert len(storage.list_sources(retried.assistant_message.id)) == 1


def test_invalid_source_citation_is_removed_from_answer(tmp_path: Path) -> None:
    storage, service, conversation, _ = make_service(tmp_path)

    result = service.process_chat_message(
        conversation.id,
        "Фактический вопрос",
        ChatLLM(answer_text="Ответ с выдуманной ссылкой [99]"),
        config(),
    )

    assert "[99]" not in result.assistant_message.content
    assert "Источники: [1]" in result.assistant_message.content
    assert storage.list_sources(result.assistant_message.id)[0]["number"] == 1


def test_empty_retrieval_is_honest_and_skips_answer_generation(tmp_path: Path) -> None:
    storage, service, conversation, _ = make_service(tmp_path, Searcher(only_low=True))
    llm = ChatLLM()

    result = service.process_chat_message(conversation.id, "Есть ли ответ?", llm, config())

    assert "источники в локальной базе не найдены" in result.assistant_message.content
    assert result.sources == []
    assert not any("ассистент по локальной базе" in call[0]["content"] for call in llm.calls)
    assert storage.list_sources(result.assistant_message.id) == []


def test_failure_persists_stage_and_safe_message(tmp_path: Path) -> None:
    storage, service, conversation, _ = make_service(tmp_path, Searcher(fail=True))

    result = service.process_chat_message(
        conversation.id,
        "Фактический вопрос",
        ChatLLM(),
        config(),
    )
    turn = storage.get_turn(result.turn_id)

    assert turn["status"] == "failed"
    assert turn["current_stage"] == "retrieving"
    assert "индекс временно недоступен" in result.assistant_message.content
    assert result.assistant_message.status == "failed"
