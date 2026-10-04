from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.models import AnswerResult, RetrievedChunk
from src.result_storage import (
    ResultStorage,
    clear_json_results,
    clear_sqlite_results,
)


def answers() -> tuple[AnswerResult, AnswerResult]:
    no_rag = AnswerResult("Вопрос", "Общий ответ", "no_rag", "fake", None, None)
    rag = AnswerResult(
        "Вопрос",
        "Ответ [1]",
        "rag",
        "fake",
        "structural",
        5,
        retrieved_chunks=[
            RetrievedChunk("chunk-1", "guide.md", "Guide", "Установка", None, "текст", 0.9)
        ],
        sources=[{"number": 1, "source": "guide.md"}],
        context_token_count=20,
    )
    return no_rag, rag


def save(storage: ResultStorage) -> int:
    no_rag, rag = answers()
    return storage.save_comparison(
        no_rag,
        rag,
        strategy="structural",
        top_k=5,
        max_context_tokens=4000,
        temperature=0.1,
    )


def test_comparison_and_both_answers_are_saved_together(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite"
    storage = ResultStorage(database)

    run_id = save(storage)
    restored = ResultStorage(database).get_run(run_id)

    assert restored is not None
    assert {item["mode"] for item in restored["answers"]} == {"rag", "no_rag"}
    assert restored["answers"][1]["sources"] or restored["answers"][0]["sources"]
    assert ResultStorage(database).counts()["answers"] == 2


def test_failed_answer_serialization_rolls_back_entire_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = ResultStorage(tmp_path / "results.sqlite")
    calls = 0

    def failing_json(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TypeError("serialization failed")
        return "[]"

    monkeypatch.setattr("src.result_storage._json", failing_json)
    no_rag, rag = answers()
    with pytest.raises(TypeError, match="serialization"):
        storage.save_comparison(
            no_rag,
            rag,
            strategy="structural",
            top_k=5,
            max_context_tokens=4000,
            temperature=0.1,
        )

    assert storage.counts()["comparisons"] == 0
    assert storage.counts()["answers"] == 0


def test_manual_evaluation_is_unique_and_history_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite"
    storage = ResultStorage(database)
    run_id = save(storage)

    storage.save_manual_evaluation(
        run_id,
        preferred_mode="rag",
        score_without_rag=1,
        score_with_rag=2,
        has_unsupported_claims=False,
        comment="Первичная оценка",
    )
    storage.save_manual_evaluation(
        run_id,
        preferred_mode="equal",
        score_without_rag=2,
        score_with_rag=2,
        has_unsupported_claims=False,
        comment="Обновлено",
    )
    history = ResultStorage(database).list_history()

    assert len(history) == 1
    assert history[0]["preferred_mode"] == "equal"
    assert ResultStorage(database).counts()["manual_evaluations"] == 1


def test_scoped_clear_uses_allowlist_and_preserves_unrelated_data(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite"
    storage = ResultStorage(database)
    save(storage)
    storage.save_benchmark_result("batch", "q1", {"ok": True})
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE chunks (vector_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO chunks VALUES (7)")

    deleted = clear_sqlite_results(database, "history")

    assert deleted == 3
    assert storage.counts()["comparisons"] == 0
    assert storage.counts()["benchmark_results"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT vector_id FROM chunks").fetchone()[0] == 7
    with pytest.raises(ValueError, match="область"):
        clear_sqlite_results(database, "rag_runs")


def test_json_export_and_clear_are_atomic_and_keep_questions(tmp_path: Path) -> None:
    storage = ResultStorage(tmp_path / "results.sqlite")
    save(storage)
    storage.save_benchmark_result("batch", "q1", {"metric": 1.0})
    output = tmp_path / "evaluation" / "results.json"
    questions = tmp_path / "evaluation" / "questions.json"
    questions.parent.mkdir(parents=True, exist_ok=True)
    questions.write_text('{"questions":[{"id":"q1"}]}', encoding="utf-8")
    original_questions = questions.read_bytes()

    storage.export_json(output)
    exported = json.loads(output.read_text(encoding="utf-8"))
    output.write_text("{повреждённый json", encoding="utf-8")
    clear_json_results(output)
    cleared = json.loads(output.read_text(encoding="utf-8"))

    assert exported["version"] == 1
    assert len(exported["results"]) == 2
    assert cleared == {"version": 1, "results": []}
    assert questions.read_bytes() == original_questions
