"""Валидация, безопасное обновление и diff структурированной памяти задачи."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from .chat_models import ChatMessage, TaskState, TaskStateChange
from .chat_prompts import build_memory_update_messages
from .llm import LLMProvider
from .text_utils import RegexTokenizer

_LIST_LIMITS = {
    "constraints": 20,
    "confirmed_facts": 30,
    "decisions": 20,
    "open_questions": 20,
    "user_preferences": 15,
}
_STATE_FIELDS = {
    "goal",
    "constraints",
    "terms",
    "confirmed_facts",
    "decisions",
    "open_questions",
    "user_preferences",
}
_CANCEL_MARKERS = (
    "отмен",
    "убер",
    "снять огранич",
    "больше не",
    "не нужно",
    "можно ",
    "разреш",
    "исключи",
    "remove",
    "cancel",
    "no longer",
)
_REPLACE_MARKERS = (
    "вместо",
    "теперь",
    "измени",
    "замени",
    "исправ",
    "увелич",
    "уменьш",
    "rather than",
    "change",
    "replace",
)
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_@./:-]+", re.UNICODE)


def _normalize_text(value: Any, *, field: str, limit: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Поле {field} должно содержать строку.")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"Поле {field} не должно содержать пустую строку.")
    if len(normalized) > limit:
        raise ValueError(f"Поле {field} превышает лимит {limit} символов.")
    return normalized


def _normalize_list(value: Any, *, field: str, limit: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Поле {field} должно быть массивом строк.")
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _normalize_text(item, field=field)
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    if len(output) > limit:
        raise ValueError(f"Поле {field} превышает лимит {limit} элементов.")
    return output


def _normalize_terms(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Поле terms должно быть JSON-объектом строк.")
    output: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_text(raw_key, field="terms.key", limit=120)
        definition = _normalize_text(raw_value, field=f"terms.{key}", limit=500)
        output[key] = definition
    if len(output) > 30:
        raise ValueError("Поле terms превышает лимит 30 пар.")
    return output


def validate_task_state_payload(payload: Any) -> TaskState:
    """Строго проверить структурированный JSON LLM или ручного редактора."""
    if not isinstance(payload, dict):
        raise ValueError("Память должна быть JSON-объектом.")
    unknown = set(payload) - (_STATE_FIELDS | {"changes"})
    missing = _STATE_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"Неизвестные поля памяти: {', '.join(sorted(unknown))}.")
    if missing:
        raise ValueError(f"Отсутствуют поля памяти: {', '.join(sorted(missing))}.")
    goal_value = payload["goal"]
    goal = None if goal_value is None else _normalize_text(goal_value, field="goal")
    state = TaskState(
        goal=goal,
        constraints=_normalize_list(
            payload["constraints"], field="constraints", limit=_LIST_LIMITS["constraints"]
        ),
        terms=_normalize_terms(payload["terms"]),
        confirmed_facts=_normalize_list(
            payload["confirmed_facts"],
            field="confirmed_facts",
            limit=_LIST_LIMITS["confirmed_facts"],
        ),
        decisions=_normalize_list(
            payload["decisions"], field="decisions", limit=_LIST_LIMITS["decisions"]
        ),
        open_questions=_normalize_list(
            payload["open_questions"],
            field="open_questions",
            limit=_LIST_LIMITS["open_questions"],
        ),
        user_preferences=_normalize_list(
            payload["user_preferences"],
            field="user_preferences",
            limit=_LIST_LIMITS["user_preferences"],
        ),
    )
    protected = json.dumps(
        {"goal": state.goal, "constraints": state.constraints},
        ensure_ascii=False,
    )
    if len(RegexTokenizer().token_spans(protected)) > 800:
        raise ValueError("Цель и ограничения превышают лимит 800 токенов.")
    return state


def _content(state: TaskState) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "constraints": list(state.constraints),
        "terms": dict(state.terms),
        "confirmed_facts": list(state.confirmed_facts),
        "decisions": list(state.decisions),
        "open_questions": list(state.open_questions),
        "user_preferences": list(state.user_preferences),
    }


def diff_task_states(before: TaskState, after: TaskState, *, reason: str | None = None) -> TaskStateChange:
    """Вычислить детерминированную содержательную разницу двух состояний."""
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    changed: dict[str, Any] = {}
    old = _content(before)
    new = _content(after)
    if old["goal"] != new["goal"]:
        if old["goal"] is None:
            added["goal"] = new["goal"]
        elif new["goal"] is None:
            removed["goal"] = old["goal"]
        else:
            changed["goal"] = {"from": old["goal"], "to": new["goal"]}
    for field in _LIST_LIMITS:
        old_values = old[field]
        new_values = new[field]
        new_keys = {item.casefold() for item in new_values}
        old_keys = {item.casefold() for item in old_values}
        additions = [item for item in new_values if item.casefold() not in old_keys]
        removals = [item for item in old_values if item.casefold() not in new_keys]
        if additions:
            added[field] = additions
        if removals:
            removed[field] = removals
    old_terms = old["terms"]
    new_terms = new["terms"]
    term_additions = {key: value for key, value in new_terms.items() if key not in old_terms}
    term_removals = {key: value for key, value in old_terms.items() if key not in new_terms}
    term_changes = {
        key: {"from": old_terms[key], "to": new_terms[key]}
        for key in old_terms.keys() & new_terms.keys()
        if old_terms[key] != new_terms[key]
    }
    if term_additions:
        added["terms"] = term_additions
    if term_removals:
        removed["terms"] = term_removals
    if term_changes:
        changed["terms"] = term_changes
    return TaskStateChange(added=added, removed=removed, changed=changed, reason=reason)


def _terms(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(value)
        if len(token) >= 3 or any(character.isdigit() for character in token)
    }


def _supported_by_message(value: str, user_message: str) -> bool:
    value_terms = _terms(value)
    message_terms = _terms(user_message)
    if not value_terms:
        return False
    overlap = value_terms & message_terms
    return bool(overlap) and (
        len(overlap) >= 2
        or len(value_terms) <= 2
        or any(any(character.isdigit() for character in token) for token in overlap)
    )


def _has_marker(message: str, markers: tuple[str, ...]) -> bool:
    folded = message.casefold()
    return any(marker in folded for marker in markers)


def _protect_state_transitions(
    current: TaskState,
    candidate: TaskState,
    user_message: str,
) -> TaskState:
    """Не позволить LLM молча удалить или воскресить сведения из старого контекста."""
    cancellation = _has_marker(user_message, _CANCEL_MARKERS)
    replacement = _has_marker(user_message, _REPLACE_MARKERS)

    constraints: list[str] = []
    candidate_keys = {value.casefold() for value in candidate.constraints}
    for existing in current.constraints:
        if existing.casefold() in candidate_keys:
            constraints.append(existing)
        elif (cancellation or replacement) and _supported_by_message(existing, user_message):
            continue
        else:
            constraints.append(existing)
    existing_keys = {value.casefold() for value in current.constraints}
    for value in candidate.constraints:
        if value.casefold() in existing_keys:
            continue
        if _supported_by_message(value, user_message):
            constraints.append(value)

    goal = candidate.goal
    if current.goal != candidate.goal:
        if candidate.goal is None:
            if not (cancellation and current.goal and _supported_by_message(current.goal, user_message)):
                goal = current.goal
        elif not _supported_by_message(candidate.goal, user_message):
            goal = current.goal

    terms = dict(current.terms)
    for key, value in candidate.terms.items():
        if key in current.terms and current.terms[key] == value:
            continue
        if _supported_by_message(key, user_message) or _supported_by_message(value, user_message):
            terms[key] = value
    for key in set(current.terms) - set(candidate.terms):
        if (cancellation or replacement) and _supported_by_message(key, user_message):
            terms.pop(key, None)

    protected_lists: dict[str, list[str]] = {}
    for field in ("confirmed_facts", "decisions", "open_questions", "user_preferences"):
        current_values = list(getattr(current, field))
        candidate_values = list(getattr(candidate, field))
        values = list(current_values)
        current_keys = {value.casefold() for value in current_values}
        candidate_keys = {value.casefold() for value in candidate_values}
        if cancellation or replacement:
            values = [
                value
                for value in values
                if value.casefold() in candidate_keys or not _supported_by_message(value, user_message)
            ]
        for value in candidate_values:
            if value.casefold() not in current_keys and _supported_by_message(value, user_message):
                values.append(value)
        protected_lists[field] = values[: _LIST_LIMITS[field]]

    return TaskState(
        goal=goal,
        constraints=constraints[: _LIST_LIMITS["constraints"]],
        terms=dict(list(terms.items())[:30]),
        confirmed_facts=protected_lists["confirmed_facts"],
        decisions=protected_lists["decisions"],
        open_questions=protected_lists["open_questions"],
        user_preferences=protected_lists["user_preferences"],
        version=current.version,
        updated_at=current.updated_at,
    )


def update_task_state(
    current_state: TaskState,
    user_message: str,
    recent_messages: list[ChatMessage],
    llm: LLMProvider,
) -> tuple[TaskState, TaskStateChange]:
    """Обновить память только по сообщению пользователя; при ошибке оставить последнюю версию."""
    normalized_message = user_message.strip()
    if not normalized_message:
        raise ValueError("Сообщение пользователя не может быть пустым.")
    try:
        raw = llm.generate(
            build_memory_update_messages(current_state, normalized_message, recent_messages),
            0.0,
        )
        payload = json.loads(raw)
        candidate = validate_task_state_payload(payload)
        protected = _protect_state_transitions(current_state, candidate, normalized_message)
        change = diff_task_states(current_state, protected)
        if not change.has_changes:
            return replace(current_state), TaskStateChange(reason="Содержательных изменений нет.")
        return protected, change
    except Exception as exc:
        return replace(current_state), TaskStateChange(
            reason=f"Память не обновлена; сохранена последняя корректная версия: {exc}"
        )


def task_state_from_manual_payload(payload: dict[str, Any]) -> TaskState:
    """Проверить состояние из ручного редактора теми же правилами типов и лимитов."""
    return validate_task_state_payload(payload)
