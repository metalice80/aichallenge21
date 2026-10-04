"""Контрольный прогон RAG, автоматические метрики и Markdown-отчёт."""

from __future__ import annotations

import json
import statistics
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RAG_QUESTIONS_PATH, RAG_REPORT_PATH
from .llm import LLMProvider
from .models import AnswerResult, RetrievedChunk
from .rag import Retriever, compare_answers
from .result_storage import ResultStorage
from .text_utils import OffsetTokenizer

ProgressCallback = Callable[[int, int, str], None]


def load_questions(path: Path = RAG_QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Загрузить и строго проверить набор контрольных вопросов."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list) or len(questions) < 10:
        raise ValueError("Набор должен содержать минимум 10 контрольных вопросов.")
    identifiers: set[str] = set()
    required = {"id", "question", "expectation", "expected_keywords", "expected_sources"}
    for item in questions:
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError("Каждый вопрос должен содержать id, question, expectation и ожидания.")
        identifier = str(item["id"])
        if identifier in identifiers:
            raise ValueError(f"Повторяющийся идентификатор вопроса: {identifier}")
        identifiers.add(identifier)
        if not str(item["question"]).strip() or not str(item["expectation"]).strip():
            raise ValueError(f"Вопрос {identifier} или его ожидание пусты.")
        if not isinstance(item["expected_keywords"], list) or not isinstance(
            item["expected_sources"], list
        ):
            raise ValueError(f"Ожидания вопроса {identifier} должны быть списками.")
    return questions


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Посчитать долю ключевых фраз без учёта регистра."""
    if not expected_keywords:
        return 1.0
    normalized = answer.casefold()
    hits = sum(keyword.casefold() in normalized for keyword in expected_keywords)
    return hits / len(expected_keywords)


def source_hit_at_k(
    chunks: list[RetrievedChunk],
    expected_sources: list[str],
    k: int,
) -> float | None:
    """Проверить наличие ожидаемого источника среди первых K чанков."""
    if k <= 0:
        raise ValueError("K должен быть больше нуля.")
    if not expected_sources:
        return None
    expected = {source.casefold() for source in expected_sources}
    actual = {chunk.source.casefold() for chunk in chunks[:k]}
    return float(bool(expected & actual))


def _answer_metrics(
    without_rag: AnswerResult,
    with_rag: AnswerResult,
    question: dict[str, Any],
    top_k: int,
) -> dict[str, float | int | None]:
    chunks = with_rag.retrieved_chunks
    return {
        "keyword_coverage_without_rag": keyword_coverage(
            without_rag.answer, question["expected_keywords"]
        ),
        "keyword_coverage_with_rag": keyword_coverage(
            with_rag.answer, question["expected_keywords"]
        ),
        "source_hit_at_1": source_hit_at_k(chunks, question["expected_sources"], 1),
        "source_hit_at_3": source_hit_at_k(chunks, question["expected_sources"], 3),
        "source_hit_at_k": source_hit_at_k(chunks, question["expected_sources"], top_k),
        "mean_similarity": statistics.fmean(chunk.score for chunk in chunks) if chunks else 0.0,
        "retrieval_time": with_rag.retrieval_time,
        "generation_time_without_rag": without_rag.generation_time,
        "generation_time_with_rag": with_rag.generation_time,
        "total_time_without_rag": without_rag.total_time,
        "total_time_with_rag": with_rag.total_time,
        "source_count": len(with_rag.sources),
        "unexpected_sources_for_unanswerable": (
            int(bool(with_rag.sources)) if not question["expected_sources"] else None
        ),
    }


def run_benchmark(
    questions: list[dict[str, Any]],
    llm: LLMProvider,
    storage: ResultStorage,
    *,
    strategy: str = "structural",
    top_k: int = 5,
    max_context_tokens: int = 4000,
    temperature: float = 0.1,
    retriever: Retriever | None = None,
    tokenizer: OffsetTokenizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Выполнить все вопросы, сохраняя каждый результат независимо."""
    batch_id = uuid.uuid4().hex
    results: list[dict[str, Any]] = []
    total = len(questions)
    for position, question in enumerate(questions, start=1):
        payload: dict[str, Any] = {
            "question": question,
            "parameters": {
                "model": getattr(llm, "model_name", llm.__class__.__name__),
                "strategy": strategy,
                "top_k": top_k,
                "max_context_tokens": max_context_tokens,
                "temperature": temperature,
            },
            "error": None,
        }
        try:
            without_rag, with_rag = compare_answers(
                question["question"],
                llm,
                strategy=strategy,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
                temperature=temperature,
                retriever=retriever,
                tokenizer=tokenizer,
            )
            payload["without_rag"] = without_rag.to_dict()
            payload["with_rag"] = with_rag.to_dict()
            payload["metrics"] = _answer_metrics(without_rag, with_rag, question, top_k)
        except Exception as exc:
            payload["error"] = str(exc)
            payload["metrics"] = {}
        storage.save_benchmark_result(batch_id, str(question["id"]), payload)
        results.append(payload)
        if progress_callback:
            progress_callback(position, total, str(question["id"]))
    return batch_id, results


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """Агрегировать только успешно завершённые контрольные вопросы."""
    successful = [result for result in results if not result.get("error")]
    if not successful:
        return {"successful": 0, "failed": len(results)}

    def mean_metric(name: str) -> float:
        values = [
            float(result["metrics"][name])
            for result in successful
            if result["metrics"].get(name) is not None
        ]
        return statistics.fmean(values) if values else 0.0

    return {
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "keyword_coverage_without_rag": mean_metric("keyword_coverage_without_rag"),
        "keyword_coverage_with_rag": mean_metric("keyword_coverage_with_rag"),
        "source_hit_at_1": mean_metric("source_hit_at_1"),
        "source_hit_at_3": mean_metric("source_hit_at_3"),
        "source_hit_at_k": mean_metric("source_hit_at_k"),
        "mean_retrieval_time": mean_metric("retrieval_time"),
        "mean_generation_time_without_rag": mean_metric("generation_time_without_rag"),
        "mean_generation_time_with_rag": mean_metric("generation_time_with_rag"),
        "mean_source_count": mean_metric("source_count"),
        "mean_total_time_without_rag": mean_metric("total_time_without_rag"),
        "mean_total_time_with_rag": mean_metric("total_time_with_rag"),
        "manual_rag_better": 0,
    }


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_rag_report(
    results: list[dict[str, Any]],
    *,
    model_name: str,
    strategy: str,
    top_k: int,
    temperature: float,
    path: Path = RAG_REPORT_PATH,
) -> Path:
    """Сформировать подробный Markdown-отчёт контрольного прогона."""
    metrics = aggregate_metrics(results)
    lines = [
        "# Сравнение ответов с RAG и без RAG",
        "",
        f"- Дата: {datetime.now(timezone.utc).isoformat()}",
        f"- Модель: `{model_name}`",
        f"- Стратегия: `{strategy}`",
        f"- top_k: `{top_k}`",
        f"- temperature: `{temperature}`",
        "",
        "## Сводная таблица",
        "",
        "| ID | Вопрос | Ожидание | Источники | KW без RAG | KW с RAG | Hit@1 | Hit@3 | Hit@K | Ошибка |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        question = result["question"]
        item_metrics = result.get("metrics", {})
        lines.append(
            "| {id} | {question} | {expectation} | {sources} | {kw0} | {kw1} | {h1} | {h3} | {hk} | {error} |".format(
                id=_cell(question["id"]),
                question=_cell(question["question"]),
                expectation=_cell(question["expectation"]),
                sources=_cell(", ".join(question["expected_sources"]) or "нет"),
                kw0=f"{item_metrics.get('keyword_coverage_without_rag', 0):.2f}",
                kw1=f"{item_metrics.get('keyword_coverage_with_rag', 0):.2f}",
                h1=_cell(item_metrics.get("source_hit_at_1", "—")),
                h3=_cell(item_metrics.get("source_hit_at_3", "—")),
                hk=_cell(item_metrics.get("source_hit_at_k", "—")),
                error=_cell(result.get("error") or ""),
            )
        )
    lines.extend(["", "## Подробные результаты", ""])
    for result in results:
        question = result["question"]
        lines.extend(
            [
                f"### {question['id']}: {question['question']}",
                "",
                f"**Ожидание:** {question['expectation']}",
                "",
                f"**Без RAG:** {result.get('without_rag', {}).get('answer', 'ошибка')}",
                "",
                f"**С RAG:** {result.get('with_rag', {}).get('answer', 'ошибка')}",
                "",
                f"**Ожидаемые источники:** {', '.join(question['expected_sources']) or 'нет'}",
                "",
                f"**Найденные источники:** {', '.join(source.get('source', '') for source in result.get('with_rag', {}).get('sources', [])) or 'нет'}",
                "",
                "**Ручная оценка:** ____________________",
                "",
            ]
        )
    lines.extend(
        [
            "## Агрегированные показатели",
            "",
            *[f"- {key}: {value}" for key, value in metrics.items()],
            "",
            "## Вывод",
            "",
            "RAG оценивается по покрытию ожидаемых фактов, попаданию источников и ручной проверке. "
            "Keyword Coverage является вспомогательной метрикой и не заменяет содержательную оценку.",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
