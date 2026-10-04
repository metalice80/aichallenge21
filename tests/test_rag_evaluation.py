from __future__ import annotations

from pathlib import Path

import pytest

from src.models import RetrievedChunk
from src.rag_evaluation import (
    aggregate_metrics,
    keyword_coverage,
    load_questions,
    run_benchmark,
    source_hit_at_k,
    write_rag_report,
)
from src.result_storage import ResultStorage
from src.text_utils import RegexTokenizer


class BenchmarkLLM:
    model_name = "benchmark-fake"

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def generate(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("контрольная ошибка")
        return "venv pip install streamlit run FAISS SQLite норм нулев нет информации [1]"


class BenchmarkRetriever:
    tokenizer = RegexTokenizer()

    def retrieve(self, query: str, strategy: str, top_k: int):
        source = "sample_module.py" if "VectorCatalog" in query or "normalize_vector" in query else "guide.md"
        return [
            RetrievedChunk(
                chunk_id=f"chunk-{source}",
                source=source,
                title="Demo",
                section="Section",
                page_number=None,
                content="venv pip install streamlit run FAISS SQLite normalize vector",
                score=0.91,
                token_count=10,
            )
        ]


def questions(count: int = 3):
    return [
        {
            "id": f"q{index}",
            "question": f"Вопрос {index}",
            "expectation": "Проверяемый ответ",
            "expected_keywords": ["FAISS", "SQLite"],
            "expected_sources": ["guide.md"],
        }
        for index in range(1, count + 1)
    ]


def test_keyword_coverage_is_case_insensitive() -> None:
    assert keyword_coverage("FaIsS связан с sqlite", ["FAISS", "SQLite"]) == 1.0
    assert keyword_coverage("Только FAISS", ["FAISS", "SQLite"]) == 0.5


def test_source_hit_at_k_handles_expected_and_unanswerable_questions() -> None:
    chunks = [
        RetrievedChunk("1", "wrong.txt", None, None, None, "", 0.9),
        RetrievedChunk("2", "guide.md", None, None, None, "", 0.8),
    ]

    assert source_hit_at_k(chunks, ["guide.md"], 1) == 0.0
    assert source_hit_at_k(chunks, ["guide.md"], 3) == 1.0
    assert source_hit_at_k(chunks, [], 3) is None


def test_questions_file_contains_real_requirements() -> None:
    loaded = load_questions()

    assert len(loaded) >= 10
    assert any("sample_module.py" in item["expected_sources"] for item in loaded)
    assert any(not item["expected_sources"] for item in loaded)
    assert all(item["expectation"] for item in loaded)


def test_benchmark_continues_after_one_question_fails(tmp_path: Path) -> None:
    storage = ResultStorage(tmp_path / "results.sqlite")
    progress: list[int] = []

    batch_id, results = run_benchmark(
        questions(),
        BenchmarkLLM(fail_on_call=3),
        storage,
        retriever=BenchmarkRetriever(),
        progress_callback=lambda position, total, question_id: progress.append(position),
    )

    assert len(results) == 3
    assert sum(bool(result["error"]) for result in results) == 1
    assert storage.counts()["benchmark_results"] == 3
    assert len(storage.benchmark_results(batch_id)) == 3
    assert progress == [1, 2, 3]


def test_benchmark_metrics_and_report_are_generated(tmp_path: Path) -> None:
    storage = ResultStorage(tmp_path / "results.sqlite")
    _, results = run_benchmark(
        questions(),
        BenchmarkLLM(),
        storage,
        retriever=BenchmarkRetriever(),
    )
    metrics = aggregate_metrics(results)
    report = write_rag_report(
        results,
        model_name="benchmark-fake",
        strategy="structural",
        top_k=5,
        temperature=0.1,
        path=tmp_path / "rag_comparison.md",
    )
    report_text = report.read_text(encoding="utf-8")

    assert metrics["successful"] == 3
    assert metrics["keyword_coverage_with_rag"] == 1.0
    assert "Сравнение ответов с RAG" in report_text
    assert "Ручная оценка" in report_text
    assert "Вопрос 1" in report_text
