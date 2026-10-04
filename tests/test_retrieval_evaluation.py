from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import RetrievalConfig, RetrievedChunk
from src.retrieval_evaluation import (
    aggregate_reranking_metrics,
    retrieval_metrics,
    run_reranking_benchmark,
    run_threshold_experiment,
    write_reranking_report,
)
from src.result_storage import ResultStorage
from src.text_utils import RegexTokenizer


class EvaluationLLM:
    model_name = "evaluation-fake"

    def generate(self, messages, temperature=0.1):
        if "Преобразуй вопрос" in messages[0]["content"]:
            return json.dumps(
                {
                    "search_query": "установка streamlit",
                    "alternative_queries": ["команда запуска"],
                }
            )
        return "venv pip install streamlit run [1]"


class EvaluationRetriever:
    tokenizer = RegexTokenizer()

    def retrieve(self, query: str, strategy: str, top_k: int):
        if "ошиб" in query:
            raise RuntimeError("ошибка одного вопроса")
        return [
            RetrievedChunk(
                "baseline",
                "guide.md",
                "Guide",
                "Установка",
                None,
                "venv pip install streamlit run",
                0.70,
                8,
            )
        ]


class EvaluationSearcher:
    tokenizer = RegexTokenizer()

    def search(self, query: str, strategy: str, top_k: int):
        source = "guide.md" if "ошиб" not in query else "wrong.txt"
        return [
            {
                "chunk_id": f"{source}-{query}",
                "source": source,
                "title": "Guide",
                "section": "Установка",
                "page_number": None,
                "content": "venv pip install streamlit run",
                "score": 0.82,
                "token_count": 8,
            },
            {
                "chunk_id": f"low-{query}",
                "source": "wrong.txt",
                "title": "Wrong",
                "section": None,
                "page_number": None,
                "content": "нерелевантный текст",
                "score": 0.20,
                "token_count": 3,
            },
        ][:top_k]


def questions():
    return [
        {
            "id": "q1",
            "question": "Как запустить приложение?",
            "expectation": "Команда запуска",
            "expected_keywords": ["streamlit run"],
            "expected_sources": ["guide.md"],
        },
        {
            "id": "q2",
            "question": "вопрос с ошибкой",
            "expectation": "Контролируемая ошибка",
            "expected_keywords": [],
            "expected_sources": ["guide.md"],
        },
    ]


def test_retrieval_metrics_calculates_hits_mrr_and_precision() -> None:
    chunks = [
        RetrievedChunk("1", "wrong.txt", None, None, None, "", 0.9),
        RetrievedChunk("2", "guide.md", None, None, None, "", 0.8),
    ]

    metrics = retrieval_metrics(chunks, ["guide.md"], 3)

    assert metrics["source_hit_at_1"] == 0.0
    assert metrics["source_hit_at_3"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["first_relevant_rank"] == 2
    assert retrieval_metrics(chunks, [], 3)["mrr"] is None


def test_benchmark_compares_modes_saves_each_question_and_report(tmp_path: Path) -> None:
    storage = ResultStorage(tmp_path / "results.sqlite")
    config = RetrievalConfig(candidate_top_k=4, final_top_k=2, similarity_threshold=0.35)

    batch_id, results = run_reranking_benchmark(
        questions(),
        EvaluationLLM(),
        storage,
        config=config,
        retriever=EvaluationRetriever(),
        candidate_searcher=EvaluationSearcher(),
        tokenizer=RegexTokenizer(),
    )
    aggregate = aggregate_reranking_metrics(results)
    report = write_reranking_report(
        results,
        model_name="evaluation-fake",
        strategy="structural",
        config=config,
        path=tmp_path / "reranking_comparison.md",
    )

    assert len(results) == 2
    assert sum(bool(item["error"]) for item in results) == 1
    assert len(storage.benchmark_results(batch_id)) == 2
    restored_rows = ResultStorage(tmp_path / "results.sqlite").latest_reranking_results()
    assert len(restored_rows) == 2
    successful_payload = storage.benchmark_results(batch_id)[0]["result"]
    assert "content" not in successful_payload["enhanced"]["retrieved_chunks"][0]
    assert (
        "content"
        not in successful_payload["enhanced"]["retrieval_diagnostics"][
            "initial_candidates"
        ][0]
    )
    storage.save_benchmark_result(
        "threshold-batch",
        "threshold-0.35",
        {
            "type": "threshold_experiment",
            "metrics": {"threshold": 0.35, "source_hit_at_k": 1.0},
        },
    )
    assert len(storage.latest_reranking_results()) == 2
    assert storage.latest_threshold_results() == [
        {"threshold": 0.35, "source_hit_at_k": 1.0}
    ]
    assert aggregate["successful"] == 1
    assert aggregate["enhanced_hit_at_k"] == 1.0
    text = report.read_text(encoding="utf-8")
    assert "Baseline RAG и Enhanced RAG" in text
    assert "Rewritten query" in text
    assert "Как запустить приложение?" in text
    assert "Технический вывод" in text


def test_threshold_experiment_uses_all_values_without_generation() -> None:
    config = RetrievalConfig(
        candidate_top_k=4,
        final_top_k=2,
        use_query_rewrite=False,
    )

    results = run_threshold_experiment(
        questions()[:1],
        [0.20, 0.35, 0.50],
        strategy="structural",
        config=config,
        rewrite_llm=None,
        searcher=EvaluationSearcher(),
    )
    assert all(item["query_rewrite_enabled"] == 0 for item in results)

    assert [item["threshold"] for item in results] == [0.20, 0.35, 0.50]
    assert results[0]["mean_accepted_candidates"] == 2.0
    assert results[1]["mean_accepted_candidates"] == 1.0
    assert results[2]["empty_result_share"] == 0.0
