from __future__ import annotations

import json

from src.chat_models import TaskState
from src.task_memory import diff_task_states, update_task_state


class MemoryLLM:
    model_name = "memory-fake"

    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = []

    def generate(self, messages, temperature=0.1):
        self.calls.append(messages)
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)


def payload(**overrides):
    value = {
        "goal": None,
        "constraints": [],
        "terms": {},
        "confirmed_facts": [],
        "decisions": [],
        "open_questions": [],
        "user_preferences": [],
        "changes": {"added": {}, "removed": {}, "changed": {}},
    }
    value.update(overrides)
    return value


def test_memory_preserves_goal_and_multiple_constraints() -> None:
    llm = MemoryLLM(
        payload(
            goal="Настроить локальный RAG",
            constraints=["Только CPU", "Без внешних API"],
        )
    )

    state, change = update_task_state(
        TaskState(),
        "Цель — настроить локальный RAG. Только CPU и без внешних API.",
        [],
        llm,
    )

    assert state.goal == "Настроить локальный RAG"
    assert state.constraints == ["Только CPU", "Без внешних API"]
    assert change.added["goal"] == "Настроить локальный RAG"
    assert change.added["constraints"] == ["Только CPU", "Без внешних API"]


def test_constraint_is_removed_only_after_explicit_cancellation() -> None:
    current = TaskState(
        goal="Настроить локальный RAG",
        constraints=["Только CPU", "Без внешних API"],
    )
    llm = MemoryLLM(
        payload(goal=current.goal, constraints=["Только CPU"])
    )

    state, change = update_task_state(
        current,
        "Ограничение «без внешних API» отменяем.",
        [],
        llm,
    )

    assert state.constraints == ["Только CPU"]
    assert change.removed["constraints"] == ["Без внешних API"]


def test_missing_constraint_without_explicit_reason_is_restored() -> None:
    current = TaskState(goal="RAG", constraints=["Только CPU"])
    llm = MemoryLLM(payload(goal="RAG", constraints=[]))

    state, change = update_task_state(current, "Расскажи про PDF.", [], llm)

    assert state.constraints == ["Только CPU"]
    assert not change.has_changes


def test_cancelled_constraint_is_not_resurrected_by_old_context() -> None:
    current = TaskState(goal="Настроить RAG", constraints=["Только CPU"])
    llm = MemoryLLM(
        payload(
            goal="Настроить RAG",
            constraints=["Только CPU", "Без внешних API"],
        )
    )

    state, _ = update_task_state(current, "Вернёмся к основной цели.", [], llm)

    assert state.constraints == ["Только CPU"]


def test_invalid_json_keeps_last_valid_state_without_version_change() -> None:
    current = TaskState(goal="Не потерять", constraints=["CPU"], version=4)

    state, change = update_task_state(current, "Продолжай.", [], MemoryLLM("not-json"))

    assert state == current
    assert state.version == 4
    assert "последняя корректная версия" in change.reason


def test_diff_detects_term_change_and_no_change_is_empty() -> None:
    before = TaskState(terms={"основной индекс": "fixed"})
    after = TaskState(terms={"основной индекс": "structural"})

    change = diff_task_states(before, after)

    assert change.changed["terms"]["основной индекс"] == {
        "from": "fixed",
        "to": "structural",
    }
    assert not diff_task_states(after, after).has_changes
