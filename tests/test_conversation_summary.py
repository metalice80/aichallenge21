from __future__ import annotations

from src.chat_models import ChatMessage, TaskState
from src.conversation_summary import messages_to_summarize, update_conversation_summary


class SummaryLLM:
    model_name = "summary-fake"

    def __init__(self, response="Обновлённое summary", fail=False) -> None:
        self.response = response
        self.fail = fail
        self.calls = 0

    def generate(self, messages, temperature=0.1):
        self.calls += 1
        if self.fail:
            raise RuntimeError("offline")
        return self.response


def messages(count: int) -> list[ChatMessage]:
    return [
        ChatMessage(str(index), "conversation", "user", f"message {index}", f"2026-{index:02d}")
        for index in range(count)
    ]


def test_summary_selection_never_repeats_processed_messages() -> None:
    history = messages(14)

    first = messages_to_summarize(
        history,
        summarized_until_message_id=None,
        recent_message_count=8,
    )
    second = messages_to_summarize(
        history,
        summarized_until_message_id=first[-1].id,
        recent_message_count=8,
    )

    assert [item.id for item in first] == ["0", "1", "2", "3", "4", "5"]
    assert second == []


def test_summary_adds_only_newly_eligible_messages() -> None:
    history = messages(16)
    selected = messages_to_summarize(
        history,
        summarized_until_message_id="5",
        recent_message_count=8,
    )

    assert [item.id for item in selected] == ["6", "7"]


def test_summary_failure_preserves_previous_value() -> None:
    previous = "Стабильное summary"
    result = update_conversation_summary(
        previous,
        messages(2),
        TaskState(goal="Цель"),
        SummaryLLM(fail=True),
    )

    assert result == previous
