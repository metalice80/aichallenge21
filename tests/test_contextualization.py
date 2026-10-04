from __future__ import annotations

from src.chat_models import ChatMessage, TaskState
from src.contextualization import contextualize_question_result


class TextLLM:
    model_name = "text-fake"

    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, messages, temperature=0.1):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.response


def message(identifier: str, role: str, content: str) -> ChatMessage:
    return ChatMessage(identifier, "conversation", role, content, "2026-01-01")


def test_follow_up_uses_history_goal_and_terms_in_prompt() -> None:
    llm = TextLLM("Можно ли уменьшить Final Top-K Enhanced RAG с пяти до трёх?")
    result = contextualize_question_result(
        "А можно уменьшить его до трёх?",
        TaskState(
            goal="Повысить Source Hit@3",
            terms={"основной поиск": "Enhanced RAG"},
        ),
        [message("1", "assistant", "Сейчас Final Top-K равен пяти.")],
        "Обсуждается качество retrieval.",
        llm,
    )

    assert result.standalone_query == "Можно ли уменьшить Final Top-K Enhanced RAG с пяти до трёх?"
    prompt = llm.calls[0][-1]["content"]
    assert "Повысить Source Hit@3" in prompt
    assert "Сейчас Final Top-K равен пяти" in prompt


def test_error_falls_back_to_original_message() -> None:
    result = contextualize_question_result(
        "А что с ним?",
        TaskState(),
        [],
        "",
        TextLLM(error=RuntimeError("offline")),
    )

    assert result.standalone_query == "А что с ним?"
    assert result.fallback_used
    assert "offline" in result.fallback_reason


def test_missing_technical_identifier_causes_safe_fallback() -> None:
    result = contextualize_question_result(
        "Как изменить load_pdf в src/loaders.py?",
        TaskState(),
        [],
        "",
        TextLLM("Как изменить загрузчик PDF?"),
    )

    assert result.standalone_query == "Как изменить load_pdf в src/loaders.py?"
    assert result.fallback_used
