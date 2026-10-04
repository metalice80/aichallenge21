"""Сравнение Baseline/Enhanced RAG, retrieval-метрики и threshold experiment."""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RERANKING_REPORT_PATH
from .llm import LLMProvider
from .models import AnswerResult, RetrievalConfig, RetrievedChunk
from .rag import Retriever, compare_retrieval_modes
from .retrieval import CandidateSearcher, enhanced_retrieve
from .result_storage import ResultStorage
from .text_utils import OffsetTokenizer


def _keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> float:
    if not expected_keywords:
        return 1.0
    normalized = answer.casefold()
    return sum(keyword.casefold() in normalized for keyword in expected_keywords) / len(
        expected_keywords
    )

ProgressCallback = Callable[[int, int, str, str], None]

def _answer_payload(answer: AnswerResult) -> dict[str, Any]:
    """Сериализовать benchmark без дублирования полного текста индексных чанков."""
    payload = answer.to_dict()
    for chunk in payload.get("retrieved_chunks", []):
        chunk.pop("content", None)
    diagnostics = payload.get("retrieval_diagnostics", {})
    for name in (
        "initial_candidates",
        "accepted_candidates",
        "rejected_candidates",
        "final_chunks",
    ):
        for candidate in diagnostics.get(name, []):
            candidate.pop("content", None)
    return payload


def retrieval_metrics(
    chunks: Sequence[RetrievedChunk],
    expected_sources: Sequence[str],
    k: int,
) -> dict[str, float | int | None]:
    """Рассчитать Hit@K, MRR, Precision@K и позицию первого источника."""
    if k < 1:
        raise ValueError("K должен быть больше нуля.")
    if not expected_sources:
        return {
            "source_hit_at_1": None,
            "source_hit_at_3": None,
            "source_hit_at_k": None,
            "mrr": None,
            "precision_at_k": None,
            "first_relevant_rank": None,
        }
    expected = {source.casefold() for source in expected_sources}
    relevance = [chunk.source.casefold() in expected for chunk in chunks[:k]]
    first_rank = next((rank for rank, relevant in enumerate(relevance, start=1) if relevant), None)
    return {
        "source_hit_at_1": float(any(relevance[:1])),
        "source_hit_at_3": float(any(relevance[:3])),
        "source_hit_at_k": float(any(relevance)),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "precision_at_k": sum(relevance) / k,
        "first_relevant_rank": first_rank,
    }


def _mode_metrics(
    answer: AnswerResult,
    expected_sources: list[str],
    k: int,
) -> dict[str, float | int | None]:
    metrics = retrieval_metrics(answer.retrieved_chunks, expected_sources, k)
    metrics.update(
        {
            "retrieval_time": answer.retrieval_time,
            "generation_time": answer.generation_time,
            "total_time": answer.total_time,
            "source_count": len(answer.sources),
            "chunk_count": len(answer.retrieved_chunks),
        }
    )
    return metrics

def comparison_metrics(
    baseline: AnswerResult,
    enhanced: AnswerResult,
    question: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    """Разделить retrieval-метрики и качество ответов двух режимов."""
    expected_sources = question["expected_sources"]
    baseline_metrics = _mode_metrics(baseline, expected_sources, k)
    enhanced_metrics = _mode_metrics(enhanced, expected_sources, k)
    diagnostics = enhanced.retrieval_diagnostics
    precision = enhanced_metrics["precision_at_k"]
    return {
        "baseline": baseline_metrics,
        "enhanced": enhanced_metrics,
        "answer_quality": {
            "baseline_keyword_coverage": _keyword_coverage(
                baseline.answer, question["expected_keywords"]
            ),
            "enhanced_keyword_coverage": _keyword_coverage(
                enhanced.answer, question["expected_keywords"]
            ),
        },
        "candidate_count": int(diagnostics.get("candidate_count", 0)),
        "accepted_candidate_count": int(diagnostics.get("accepted_count", 0)),
        "final_candidate_count": int(diagnostics.get("final_count", 0)),
        "filtered_candidate_count": max(
            0,
            int(diagnostics.get("candidate_count", 0))
            - int(diagnostics.get("accepted_count", 0)),
        ),
        "empty_after_threshold": int(
            int(diagnostics.get("accepted_count", 0)) == 0
        ),
        "irrelevant_final_ratio": (
            None if precision is None else max(0.0, 1.0 - float(precision))
        ),
        "enhanced_better_retrieval": int(
            float(enhanced_metrics["source_hit_at_k"] or 0.0)
            > float(baseline_metrics["source_hit_at_k"] or 0.0)
        ),
        "rewrite_time": enhanced.rewrite_time,
        "filter_time": enhanced.filter_time,
        "rerank_time": enhanced.rerank_time,
        "total_retrieval_time": enhanced.retrieval_time,
    }


def run_reranking_benchmark(
    questions: list[dict[str, Any]],
    llm: LLMProvider,
    storage: ResultStorage,
    *,
    strategy: str = "structural",
    baseline_top_k: int = 5,
    config: RetrievalConfig | None = None,
    max_context_tokens: int = 4000,
    temperature: float = 0.1,
    retriever: Retriever | None = None,
    candidate_searcher: CandidateSearcher | None = None,
    tokenizer: OffsetTokenizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Сравнить два режима на всех вопросах, сохраняя ошибки независимо."""
    active_config = config or RetrievalConfig()
    batch_id = f"reranking-{uuid.uuid4().hex}"
    results: list[dict[str, Any]] = []
    total = len(questions)
    for position, question in enumerate(questions, start=1):
        payload: dict[str, Any] = {
            "question": question,
            "parameters": {
                "model": getattr(llm, "model_name", llm.__class__.__name__),
                "strategy": strategy,
                "baseline_top_k": baseline_top_k,
                "retrieval_config": {
                    "candidate_top_k": active_config.candidate_top_k,
                    "final_top_k": active_config.final_top_k,
                    "similarity_threshold": active_config.similarity_threshold,
                    "rerank_threshold": active_config.rerank_threshold,
                    "reranker_type": active_config.reranker_type,
                    "use_query_rewrite": active_config.use_query_rewrite,
                    "max_alternative_queries": active_config.max_alternative_queries,
                    "max_chunks_per_source": active_config.max_chunks_per_source,
                },
                "max_context_tokens": max_context_tokens,
                "temperature": temperature,
            },
            "error": None,
        }
        if progress_callback:
            progress_callback(position, total, str(question["id"]), "Baseline RAG")
        try:
            baseline, enhanced = compare_retrieval_modes(
                question["question"],
                llm,
                strategy=strategy,
                baseline_top_k=baseline_top_k,
                retrieval_config=active_config,
                max_context_tokens=max_context_tokens,
                temperature=temperature,
                retriever=retriever,
                candidate_searcher=candidate_searcher,
                tokenizer=tokenizer,
            )
            payload["baseline"] = _answer_payload(baseline)
            payload["enhanced"] = _answer_payload(enhanced)
            payload["metrics"] = comparison_metrics(
                baseline,
                enhanced,
                question,
                active_config.final_top_k,
            )
        except Exception as exc:
            payload["error"] = str(exc)
            payload["metrics"] = {}
        storage.save_benchmark_result(batch_id, str(question["id"]), payload)
        results.append(payload)
        if progress_callback:
            progress_callback(position, total, str(question["id"]), "Сохранено")
    return batch_id, results


def _mean(results: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for result in results:
        value: Any = result.get("metrics", {})
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(float(value))
    return statistics.fmean(values) if values else 0.0


def aggregate_reranking_metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """Агрегировать качество retrieval и ответов, не смешивая их."""
    successful = [result for result in results if not result.get("error")]
    return {
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "baseline_hit_at_1": _mean(successful, ("baseline", "source_hit_at_1")),
        "enhanced_hit_at_1": _mean(successful, ("enhanced", "source_hit_at_1")),
        "baseline_hit_at_3": _mean(successful, ("baseline", "source_hit_at_3")),
        "enhanced_hit_at_3": _mean(successful, ("enhanced", "source_hit_at_3")),
        "baseline_hit_at_k": _mean(successful, ("baseline", "source_hit_at_k")),
        "enhanced_hit_at_k": _mean(successful, ("enhanced", "source_hit_at_k")),
        "baseline_mrr": _mean(successful, ("baseline", "mrr")),
        "enhanced_mrr": _mean(successful, ("enhanced", "mrr")),
        "baseline_precision_at_k": _mean(successful, ("baseline", "precision_at_k")),
        "enhanced_precision_at_k": _mean(successful, ("enhanced", "precision_at_k")),
        "baseline_mean_first_relevant_rank": _mean(
            successful, ("baseline", "first_relevant_rank")
        ),
        "enhanced_mean_first_relevant_rank": _mean(
            successful, ("enhanced", "first_relevant_rank")
        ),
        "baseline_keyword_coverage": _mean(
            successful, ("answer_quality", "baseline_keyword_coverage")
        ),
        "enhanced_keyword_coverage": _mean(
            successful, ("answer_quality", "enhanced_keyword_coverage")
        ),
        "enhanced_better_question_count": int(
            sum(
                int(result.get("metrics", {}).get("enhanced_better_retrieval", 0))
                for result in successful
            )
        ),
        "mean_filtered_candidates": _mean(successful, ("filtered_candidate_count",)),
        "mean_candidate_count": _mean(successful, ("candidate_count",)),
        "mean_accepted_candidate_count": _mean(
            successful, ("accepted_candidate_count",)
        ),
        "mean_final_candidate_count": _mean(successful, ("final_candidate_count",)),
        "empty_result_share": _mean(successful, ("empty_after_threshold",)),
        "mean_irrelevant_final_ratio": _mean(successful, ("irrelevant_final_ratio",)),
        "mean_rewrite_time": _mean(successful, ("rewrite_time",)),
        "mean_filter_time": _mean(successful, ("filter_time",)),
        "mean_rerank_time": _mean(successful, ("rerank_time",)),
        "mean_total_retrieval_time": _mean(successful, ("total_retrieval_time",)),
    }


def run_threshold_experiment(
    questions: list[dict[str, Any]],
    thresholds: Sequence[float],
    *,
    strategy: str,
    config: RetrievalConfig,
    rewrite_llm: LLMProvider | None,
    searcher: CandidateSearcher,
) -> list[dict[str, float | int]]:
    """Оценить пороги retrieval без генерации и без изменения индекса."""
    if not thresholds:
        raise ValueError("Нужно указать хотя бы один threshold.")
    output: list[dict[str, float | int]] = []
    for threshold in thresholds:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Threshold должен находиться между 0 и 1.")
        per_question: list[dict[str, float | None]] = []
        started = datetime.now(timezone.utc)
        for question in questions:
            result = enhanced_retrieve(
                question["question"],
                strategy,
                replace(config, similarity_threshold=float(threshold)),
                rewrite_llm,
                searcher=searcher,
            )
            chunks = [
                RetrievedChunk(
                    item.chunk_id,
                    item.source,
                    item.title,
                    item.section,
                    item.page_number,
                    item.content,
                    item.similarity,
                )
                for item in result.final_chunks
            ]
            metrics = retrieval_metrics(
                chunks, question["expected_sources"], config.final_top_k
            )
            per_question.append(
                {
                    "hit": (
                        None
                        if metrics["source_hit_at_k"] is None
                        else float(metrics["source_hit_at_k"])
                    ),
                    "precision": (
                        None
                        if metrics["precision_at_k"] is None
                        else float(metrics["precision_at_k"])
                    ),
                    "accepted": float(result.accepted_count),
                    "empty": float(result.final_count == 0),
                    "time": result.total_retrieval_time,
                }
            )
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        hit_values = [item["hit"] for item in per_question if item["hit"] is not None]
        precision_values = [
            item["precision"]
            for item in per_question
            if item["precision"] is not None
        ]
        output.append(
            {
                "threshold": float(threshold),
                "query_rewrite_enabled": int(config.use_query_rewrite),
                "source_hit_at_k": statistics.fmean(hit_values) if hit_values else 0.0,
                "precision_at_k": (
                    statistics.fmean(precision_values) if precision_values else 0.0
                ),
                "mean_accepted_candidates": statistics.fmean(
                    float(item["accepted"] or 0.0) for item in per_question
                ),
                "empty_result_share": statistics.fmean(
                    float(item["empty"] or 0.0) for item in per_question
                ),
                "mean_processing_time": statistics.fmean(
                    float(item["time"] or 0.0) for item in per_question
                ),
                "total_seconds": elapsed,
            }
        )
    return output


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_reranking_report(
    results: list[dict[str, Any]],
    *,
    model_name: str,
    strategy: str,
    config: RetrievalConfig,
    baseline_top_k: int = 5,
    temperature: float = 0.1,
    threshold_results: list[dict[str, Any]] | None = None,
    path: Path = RERANKING_REPORT_PATH,
) -> Path:
    """Сформировать подробный Markdown-отчёт Baseline против Enhanced."""
    aggregate = aggregate_reranking_metrics(results)
    hit_delta = float(aggregate["enhanced_hit_at_k"]) - float(
        aggregate["baseline_hit_at_k"]
    )
    mrr_delta = float(aggregate["enhanced_mrr"]) - float(aggregate["baseline_mrr"])
    if hit_delta > 0 or (hit_delta == 0 and mrr_delta > 0):
        retrieval_conclusion = "Enhanced Retrieval улучшил итоговые retrieval-метрики."
    elif hit_delta < 0 or (hit_delta == 0 and mrr_delta < 0):
        retrieval_conclusion = "Enhanced Retrieval ухудшил итоговые retrieval-метрики."
    else:
        retrieval_conclusion = (
            "Enhanced Retrieval не изменил итоговые retrieval-метрики."
        )
    threshold_conclusion = ""
    if threshold_results:
        best_hit = max(float(item["source_hit_at_k"]) for item in threshold_results)
        best_precision = max(
            float(item["precision_at_k"])
            for item in threshold_results
            if float(item["source_hit_at_k"]) == best_hit
        )
        quality_thresholds = [
            float(item["threshold"])
            for item in threshold_results
            if float(item["source_hit_at_k"]) == best_hit
            and float(item["precision_at_k"]) == best_precision
        ]
        first_quality = next(
            item
            for item in threshold_results
            if float(item["threshold"]) == quality_thresholds[0]
        )
        last_quality = next(
            item
            for item in reversed(threshold_results)
            if float(item["threshold"]) == quality_thresholds[-1]
        )
        threshold_conclusion = (
            " Максимальные Hit@K/Precision@K показали пороги "
            f"`{', '.join(f'{value:.2f}' for value in quality_thresholds)}`. "
            "В этом диапазоне среднее число прошедших кандидатов изменилось "
            f"с `{first_quality['mean_accepted_candidates']:.2f}` до "
            f"`{last_quality['mean_accepted_candidates']:.2f}`, а доля пустых "
            f"результатов — с `{first_quality['empty_result_share']:.3f}` до "
            f"`{last_quality['empty_result_share']:.3f}`."
        )
    lines = [
        "# Сравнение Baseline RAG и Enhanced RAG",
        "",
        f"- Дата: {datetime.now(timezone.utc).isoformat()}",
        f"- LLM: `{model_name}`",
        f"- Индекс: `{strategy}`",
        f"- Baseline Top-K: `{baseline_top_k}`",
        f"- Enhanced Candidate Top-K: `{config.candidate_top_k}`",
        f"- Similarity threshold: `{config.similarity_threshold}`",
        f"- Reranker: `{config.reranker_type}`",
        f"- Final Top-K: `{config.final_top_k}`",
        f"- Query Rewrite: `{config.use_query_rewrite}`",
        f"- Temperature: `{temperature}`",
        "",
        "## Результаты контрольных вопросов",
        "",
        "| ID | Вопрос | Rewritten query | Baseline Hit@K | Enhanced Hit@K | Baseline MRR | Enhanced MRR | Кандидаты | После threshold | Final | Ошибка |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        question = result["question"]
        metrics = result.get("metrics", {})
        enhanced = result.get("enhanced", {})
        diagnostics = enhanced.get("retrieval_diagnostics", {})
        lines.append(
            "| {id} | {question} | {rewrite} | {bh} | {eh} | {bm} | {em} | {c} | {a} | {f} | {error} |".format(
                id=_cell(question["id"]),
                question=_cell(question["question"]),
                rewrite=_cell(enhanced.get("rewritten_query") or "—"),
                bh=_cell(metrics.get("baseline", {}).get("source_hit_at_k", "—")),
                eh=_cell(metrics.get("enhanced", {}).get("source_hit_at_k", "—")),
                bm=_cell(metrics.get("baseline", {}).get("mrr", "—")),
                em=_cell(metrics.get("enhanced", {}).get("mrr", "—")),
                c=diagnostics.get("candidate_count", 0),
                a=diagnostics.get("accepted_count", 0),
                f=diagnostics.get("final_count", 0),
                error=_cell(result.get("error") or ""),
            )
        )
    lines.extend(["", "## Подробные ответы", ""])
    for result in results:
        question = result["question"]
        baseline = result.get("baseline", {})
        enhanced = result.get("enhanced", {})
        lines.extend(
            [
                f"### {question['id']}: {question['question']}",
                "",
                f"**Ожидание:** {question['expectation']}",
                "",
                f"**Ожидаемые источники:** {', '.join(question['expected_sources']) or 'нет'}",
                "",
                f"**Исходный запрос:** {enhanced.get('original_query', question['question'])}",
                "",
                f"**Rewritten query:** {enhanced.get('rewritten_query') or 'не использован'}",
                "",
                f"**Baseline:** {baseline.get('answer', result.get('error') or 'ошибка')}",
                "",
                f"**Enhanced:** {enhanced.get('answer', result.get('error') or 'ошибка')}",
                "",
                f"**Baseline источники:** {', '.join(item.get('source', '') for item in baseline.get('sources', [])) or 'нет'}",
                "",
                f"**Enhanced источники:** {', '.join(item.get('source', '') for item in enhanced.get('sources', [])) or 'нет'}",
                "",
                "**Ручная оценка:** ____________________",
                "",
            ]
        )
    if threshold_results:
        lines.extend(
            [
                "## Эксперимент с similarity threshold",
                (
                    "Query Rewrite: `"
                    f"{bool(threshold_results[0].get('query_rewrite_enabled', 0))}` "
                    "(фиксировано для чистого сравнения порогов)."
                ),
                "",
                "",
                "| Threshold | Hit@K | Precision@K | Среднее прошедших | Пустые результаты | Время, с |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in threshold_results:
            lines.append(
                f"| {item['threshold']:.2f} | {item['source_hit_at_k']:.3f} | "
                f"{item['precision_at_k']:.3f} | {item['mean_accepted_candidates']:.2f} | "
                f"{item['empty_result_share']:.3f} | {item['mean_processing_time']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Агрегированные метрики",
            "",
            *[f"- {name}: {value}" for name, value in aggregate.items()],
            "",
            "## Технический вывод",
            "",
            (
                f"{retrieval_conclusion} ΔHit@K = `{hit_delta:+.3f}`, "
                f"ΔMRR = `{mrr_delta:+.3f}`. "
                f"Среднее число кандидатов после threshold: "
                f"`{aggregate['mean_accepted_candidate_count']:.2f}`; "
                f"доля нерелевантных финальных чанков: "
                f"`{aggregate['mean_irrelevant_final_ratio']:.3f}`."
                f"{threshold_conclusion}"
            ),
            "",
            "Ответы требуют отдельной ручной оценки: retrieval-метрики не подменяют качество генерации.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
