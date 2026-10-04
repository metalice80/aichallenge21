from __future__ import annotations

from src.chat_context import prepare_chat_context
from src.chat_models import ChatContextBudget, ChatMessage, TaskState
from src.models import RetrievedChunk
from src.text_utils import RegexTokenizer


def test_context_budget_keeps_current_question_and_whole_selected_chunks() -> None:
    tokenizer = RegexTokenizer()
    chunks = [
        RetrievedChunk("one", "a.md", None, None, None, "короткий полезный текст", 0.9),
        RetrievedChunk("two", "b.md", None, None, None, "очень " * 100, 0.8),
    ]
    recent = [
        ChatMessage("1", "c", "user", "полное недавнее сообщение", "2026-01-01")
    ]

    prepared = prepare_chat_context(
        task_state=TaskState(goal="Настроить RAG", constraints=["Только CPU"]),
        conversation_summary="старое " * 100,
        recent_messages=recent,
        retrieved_chunks=chunks,
        current_question="Текущий вопрос нельзя обрезать",
        budget=ChatContextBudget(
            task_state_tokens=100,
            summary_tokens=10,
            recent_messages_tokens=20,
            rag_context_tokens=100,
            response_reserve_tokens=10,
        ),
        tokenizer=tokenizer,
    )

    assert [chunk.chunk_id for chunk in prepared.used_chunks] == ["one"]
    assert "очень" not in prepared.rag_context_block
    assert "полное недавнее сообщение" in prepared.recent_messages_block
    assert prepared.token_usage["conversation_summary"] <= 10
    assert prepared.token_usage["rag_context"] <= 100
    assert prepared.token_usage["current_question"] == len(
        tokenizer.token_spans("Текущий вопрос нельзя обрезать")
    )
    assert prepared.token_usage["input_total"] <= prepared.token_usage["budget_total"]
