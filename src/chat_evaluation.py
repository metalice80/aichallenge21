"""Запуск и отчёт двух длинных сценариев сохранения памяти RAG-чата."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chat_service import ChatService
from .config import CHAT_REPORT_PATH, CHAT_SCENARIOS_PATH
from .llm import LLMProvider
from .models import RetrievalConfig

_CITATION_RE = re.compile(r"\[(\d+)]")


def load_chat_scenarios(path: Path = CHAT_SCENARIOS_PATH) -> list[dict[str, Any]]:
    """Загрузить и проверить два сценария по 10–15 пользовательских сообщений."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("scenarios"), list):
        raise ValueError("chat_scenarios.json должен иметь version=1 и массив scenarios.")
    scenarios = payload["scenarios"]
    if len(scenarios) != 2:
        raise ValueError("Требуются ровно два длинных чат-сценария.")
    for scenario in scenarios:
        turns = scenario.get("turns")
        if not isinstance(turns, list) or not 10 <= len(turns) <= 15:
            raise ValueError(
                f"Сценарий {scenario.get('id')} должен содержать от 10 до 15 сообщений."
            )
        for turn in turns:
            if not str(turn.get("user_message", "")).strip():
                raise ValueError("Каждый ход должен содержать user_message.")
            if not isinstance(turn.get("expected_state"), dict):
                raise ValueError("Каждый ход должен содержать expected_state.")
    return scenarios


def _contains_all(value: str | None, expected: list[str]) -> bool:
    folded = (value or "").casefold()
    return all(item.casefold() in folded for item in expected)


def _constraints_contain(constraints: list[str], expected: list[str]) -> bool:
    folded = "\n".join(constraints).casefold()
    return all(item.casefold() in folded for item in expected)


def _term_expectation(actual: dict[str, str], expected: dict[str, str]) -> bool:
    folded = {key.casefold(): value.casefold() for key, value in actual.items()}
    return all(
        key.casefold() in folded and value.casefold() in folded[key.casefold()]
        for key, value in expected.items()
    )


def evaluate_turn(turn: dict[str, Any], result) -> dict[str, Any]:
    """Сопоставить фактический ход с проверяемыми ожиданиями сценария."""
    expected_state = turn.get("expected_state", {})
    goal_ok = _contains_all(
        result.task_state.goal,
        list(expected_state.get("goal_contains", [])),
    )
    constraints_ok = _constraints_contain(
        result.task_state.constraints,
        list(expected_state.get("constraints_contain", [])),
    )
    cancelled_absent = not _constraints_contain(
        result.task_state.constraints,
        list(expected_state.get("constraints_absent", [])),
    ) if expected_state.get("constraints_absent") else True
    terms_ok = _term_expectation(
        result.task_state.terms,
        dict(expected_state.get("terms", {})),
    )
    standalone_ok = _contains_all(
        result.standalone_query,
        list(turn.get("expected_standalone_contains", [])),
    )
    actual_sources = [str(source["source"]) for source in result.sources]
    expected_sources = list(turn.get("expected_sources", []))
    if expected_sources:
        source_hit = any(source in actual_sources for source in expected_sources)
    else:
        source_hit = not actual_sources
    valid_numbers = {int(source["number"]) for source in result.sources}
    citations = [int(value) for value in _CITATION_RE.findall(result.assistant_message.content)]
    fabricated = sum(number not in valid_numbers for number in citations)
    answer_elements = list(turn.get("expected_answer_elements", []))
    coverage = (
        sum(
            element.casefold() in result.assistant_message.content.casefold()
            for element in answer_elements
        )
        / len(answer_elements)
        if answer_elements
        else 1.0
    )
    sources_block_present = bool(result.sources) or (
        "источники" in result.assistant_message.content.casefold()
        and "не найдены" in result.assistant_message.content.casefold()
    )
    return {
        "user_message": turn["user_message"],
        "goal_preserved": goal_ok,
        "constraints_preserved": constraints_ok,
        "cancelled_constraints_absent": cancelled_absent,
        "terms_resolved": terms_ok,
        "standalone_query_correct": standalone_ok,
        "expected_source_hit": source_hit,
        "sources_block_present": sources_block_present,
        "fabricated_sources_count": fabricated,
        "answer_elements_coverage": coverage,
        "task_state_update_correct": goal_ok and constraints_ok and cancelled_absent and terms_ok,
        "standalone_query": result.standalone_query,
        "expected_sources": expected_sources,
        "used_sources": actual_sources,
        "task_state": result.task_state.to_dict(),
        "status": result.assistant_message.status,
        "elapsed_seconds": float(result.timings.get("total", 0.0)),
    }


def run_chat_scenarios(
    service: ChatService,
    llm: LLMProvider,
    retrieval_config: RetrievalConfig,
    *,
    model_name: str,
    strategy: str = "structural",
    scenarios_path: Path = CHAT_SCENARIOS_PATH,
) -> list[dict[str, Any]]:
    """Выполнить оба сценария последовательно и сохранить результаты каждого хода."""
    results: list[dict[str, Any]] = []
    for scenario in load_chat_scenarios(scenarios_path):
        conversation = service.storage.create_conversation(
            title=f"Evaluation: {scenario['title']}",
            model_name=model_name,
            strategy=strategy,
            settings={
                "recent_message_count": 8,
                "summary_trigger_messages": 12,
                "automatic_memory": True,
                "automatic_summary": True,
            },
        )
        turn_results = []
        for index, turn in enumerate(scenario["turns"], start=1):
            result = service.process_chat_message(
                conversation.id,
                turn["user_message"],
                llm,
                retrieval_config,
                idempotency_key=f"{scenario['id']}:{index}",
            )
            turn_results.append({"turn": index, **evaluate_turn(turn, result)})
        results.append(
            {
                "scenario_id": scenario["id"],
                "title": scenario["title"],
                "conversation_id": conversation.id,
                "turns": turn_results,
            }
        )
    return results


def aggregate_chat_metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """Рассчитать итоговые метрики сохранения цели, памяти и источников."""
    turns = [turn for scenario in results for turn in scenario["turns"]]
    count = len(turns)
    if not count:
        return {}

    def share(field: str) -> float:
        return sum(bool(turn[field]) for turn in turns) / count

    return {
        "turn_count": count,
        "goal_preserved_share": share("goal_preserved"),
        "constraints_preserved_share": share("constraints_preserved"),
        "cancelled_constraints_absent_share": share("cancelled_constraints_absent"),
        "follow_up_correct_share": share("standalone_query_correct"),
        "source_hit_share": share("expected_source_hit"),
        "sources_block_share": share("sources_block_present"),
        "task_state_correct_share": share("task_state_update_correct"),
        "fabricated_sources_count": sum(
            int(turn["fabricated_sources_count"]) for turn in turns
        ),
        "cancelled_constraint_returns": sum(
            not bool(turn["cancelled_constraints_absent"]) for turn in turns
        ),
        "average_answer_elements_coverage": sum(
            float(turn["answer_elements_coverage"]) for turn in turns
        ) / count,
        "average_elapsed_seconds": sum(
            float(turn["elapsed_seconds"]) for turn in turns
        ) / count,
    }


def write_chat_evaluation_report(
    results: list[dict[str, Any]],
    *,
    model_name: str,
    retrieval_config: RetrievalConfig,
    path: Path = CHAT_REPORT_PATH,
    execution_mode: str,
) -> Path:
    """Атомарно сформировать Markdown-отчёт по каждому ходу двух сценариев."""
    metrics = aggregate_chat_metrics(results)
    lines = [
        "# Оценка RAG-чата и памяти задачи",
        "",
        f"Дата: {datetime.now(timezone.utc).isoformat()}",
        f"Режим проверки: {execution_mode}",
        f"LLM: `{model_name}`",
        f"Retrieval: `{json.dumps(retrieval_config.__dict__ if hasattr(retrieval_config, '__dict__') else {name: getattr(retrieval_config, name) for name in retrieval_config.__slots__}, ensure_ascii=False)}`",
        "",
        "История, task state, conversation summary, recent messages и RAG-контекст "
        "проверяются как отдельные сущности.",
        "",
    ]
    for scenario in results:
        lines.extend(
            [
                f"## {scenario['title']}",
                "",
                "| Ход | Статус | Цель | Ограничения | Follow-up | Source hit | "
                "Источники | Выдумано | Время, с |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for turn in scenario["turns"]:
            lines.append(
                f"| {turn['turn']} | {turn['status']} | "
                f"{int(turn['goal_preserved'])} | {int(turn['constraints_preserved'])} | "
                f"{int(turn['standalone_query_correct'])} | "
                f"{int(turn['expected_source_hit'])} | "
                f"{int(turn['sources_block_present'])} | "
                f"{turn['fabricated_sources_count']} | {turn['elapsed_seconds']:.3f} |"
            )
        lines.extend(["", "### Детали памяти и источников", ""])
        for turn in scenario["turns"]:
            lines.extend(
                [
                    f"#### Ход {turn['turn']}",
                    "",
                    f"- Сообщение: {turn['user_message']}",
                    f"- Standalone query: {turn['standalone_query']}",
                    f"- Ожидаемые источники: {turn['expected_sources'] or 'нет'}",
                    f"- Использованные источники: {turn['used_sources'] or 'нет'}",
                    f"- Состояние памяти: `{json.dumps(turn['task_state'], ensure_ascii=False)}`",
                    "",
                ]
            )
    lines.extend(["## Итоговые метрики", ""])
    for key, value in metrics.items():
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"- `{key}`: {rendered}")
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Успех оценивается по сохранению цели и ограничений, корректности follow-up, "
            "источникам и отсутствию возвращённых отменённых значений — не только по "
            "факту генерации ответа.",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
