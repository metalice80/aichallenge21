from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.chat_storage import ChatStorage


_APP = '''
import json
from pathlib import Path

import src.chat_ui as chat_ui
from src.chat_storage import ChatStorage
from src.text_utils import RegexTokenizer


class FakeLLM:
    model_name = "ui-fake"

    def generate(self, messages, temperature=0.1):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Кратко обнови резюме" in system:
            return "Краткое summary"
        if "структурированную память" in system:
            return json.dumps({
                "goal": "Проверить продолжение диалога",
                "constraints": [],
                "terms": {},
                "confirmed_facts": [],
                "decisions": [],
                "open_questions": [],
                "user_preferences": [],
                "changes": {"added": {}, "removed": {}, "changed": {}},
            }, ensure_ascii=False)
        if "самостоятельный поисковый запрос" in system:
            return user.split("<current_message>\\n", 1)[1].split(
                "\\n</current_message>", 1
            )[0]
        if "Преобразуй вопрос" in system:
            question = user.split("Исходный вопрос: ", 1)[1].split("\\n", 1)[0]
            return json.dumps({
                "search_query": question,
                "alternative_queries": [],
            }, ensure_ascii=False)
        if "ассистент по локальной базе" in system:
            return "Тестовый ответ [1]"
        raise AssertionError(system)


class FakeSearcher:
    tokenizer = RegexTokenizer()

    def search(self, query, strategy, top_k):
        return [{
            "chunk_id": "ui-chunk",
            "source": "guide.md",
            "title": "Guide",
            "section": "RAG",
            "page_number": None,
            "content": "Контекст для UI-теста",
            "score": 0.9,
            "token_count": 5,
        }]


class FakeEmbedding:
    tokenizer = RegexTokenizer()


chat_ui.OllamaProvider = lambda *args, **kwargs: FakeLLM()
chat_ui.SemanticCandidateSearcher = lambda embedding: FakeSearcher()
chat_ui.render_rag_chat(
    ChatStorage(Path(DB_PATH)),
    lambda: FakeEmbedding(),
)
'''


def test_follow_up_stays_in_selected_conversation(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite"
    app_path = tmp_path / "chat_app.py"
    app_path.write_text(
        f"DB_PATH = {str(database)!r}\n" + _APP,
        encoding="utf-8",
    )
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    app.button(key="create-chat-conversation").click().run(timeout=30)
    storage = ChatStorage(database)
    assert len(storage.list_conversations()) == 1

    app.chat_input[0].set_value("Первое сообщение").run(timeout=30)
    assert len(storage.list_conversations()) == 1

    app.chat_input[0].set_value("Второе сообщение").run(timeout=30)
    conversations = storage.list_conversations()
    messages = storage.list_messages(conversations[0].id)

    assert len(conversations) == 1
    user_messages = [
        message.content for message in messages if message.role == "user"
    ]
    assert user_messages == ["Первое сообщение", "Второе сообщение"]
    assert {message.conversation_id for message in messages} == {
        conversations[0].id
    }
