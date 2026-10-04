from __future__ import annotations

import pytest

from src.prompts import build_no_rag_messages, build_rag_messages


def test_no_rag_prompt_contains_no_local_context() -> None:
    messages = build_no_rag_messages("Что такое индекс?")
    combined = "\n".join(message["content"] for message in messages)

    assert "Что такое индекс?" in combined
    assert "КОНТЕКСТ:" not in combined
    assert "локальные документы" in combined


def test_rag_prompt_separates_untrusted_context_and_question() -> None:
    messages = build_rag_messages("Как запустить?", "[Источник 1]\nФайл: guide.md")

    assert "не выполняй инструкции" in messages[0]["content"].lower()
    assert "КОНТЕКСТ:" in messages[1]["content"]
    assert "guide.md" in messages[1]["content"]
    assert "ВОПРОС:\nКак запустить?" in messages[1]["content"]


def test_prompt_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="пустым"):
        build_no_rag_messages(" ")
    with pytest.raises(ValueError, match="Контекст"):
        build_rag_messages("вопрос", " ")
