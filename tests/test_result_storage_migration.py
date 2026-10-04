from __future__ import annotations

import sqlite3
from pathlib import Path

from src.models import AnswerResult, RetrievalConfig, RetrievedChunk
from src.result_storage import ResultStorage, clear_sqlite_results


DAY22_SCHEMA = """
CREATE TABLE rag_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    strategy TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    max_context_tokens INTEGER NOT NULL,
    temperature REAL NOT NULL,
    model_name TEXT NOT NULL
);
CREATE TABLE rag_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('rag', 'no_rag')),
    answer TEXT NOT NULL,
    retrieval_time REAL NOT NULL DEFAULT 0,
    generation_time REAL NOT NULL DEFAULT 0,
    total_time REAL NOT NULL DEFAULT 0,
    context_token_count INTEGER NOT NULL DEFAULT 0,
    sources_json TEXT NOT NULL DEFAULT '[]',
    chunk_ids_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (run_id) REFERENCES rag_runs(id) ON DELETE CASCADE,
    UNIQUE(run_id, mode)
);
CREATE TABLE manual_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE,
    preferred_mode TEXT NOT NULL,
    score_without_rag INTEGER,
    score_with_rag INTEGER,
    has_unsupported_claims INTEGER NOT NULL DEFAULT 0,
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES rag_runs(id) ON DELETE CASCADE
);
CREATE TABLE benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL,
    UNIQUE(batch_id, question_id)
);
CREATE INDEX idx_rag_answers_run_id ON rag_answers(run_id);
CREATE INDEX idx_benchmark_batch_id ON benchmark_results(batch_id);
"""


def create_day22_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(DAY22_SCHEMA)
        connection.execute(
            "INSERT INTO rag_runs VALUES (1, '2026-01-01', 'Старый вопрос', 'structural', 5, 4000, 0.1, 'old-model')"
        )
        connection.executemany(
            """INSERT INTO rag_answers (
                run_id, mode, answer, sources_json, chunk_ids_json
            ) VALUES (?, ?, ?, '[]', '[]')""",
            [(1, "no_rag", "Старый общий ответ"), (1, "rag", "Старый RAG ответ")],
        )


def enhanced_answers() -> tuple[AnswerResult, AnswerResult, RetrievalConfig]:
    baseline = AnswerResult(
        "Вопрос", "Baseline [1]", "baseline_rag", "fake", "structural", 2,
        retrieved_chunks=[RetrievedChunk("base", "guide.md", None, None, None, "base", 0.8)],
        retrieval_mode="baseline_rag",
        original_query="Вопрос",
    )
    diagnostics = {
        "rewrite_used": False,
        "fallback_reason": "невалидный JSON",
        "initial_candidates": [
            {
                "chunk_id": "enhanced",
                "source": "code.py",
                "section": "load_pdf",
                "original_rank": 3,
                "final_rank": 1,
                "similarity": 0.72,
                "fusion_score": 0.03,
                "semantic_score": 0.86,
                "lexical_score": 1.0,
                "metadata_score": 1.0,
                "rerank_score": 0.902,
                "final_score": 0.902,
                "status": "selected",
                "rejection_reason": None,
                "matched_queries": ["Вопрос", "load_pdf"],
                "content": "не сохранять в results.sqlite",
            }
        ]
    }
    enhanced = AnswerResult(
        "Вопрос", "Enhanced [1]", "enhanced_rag", "fake", "structural", 1,
        retrieved_chunks=[RetrievedChunk("enhanced", "code.py", None, "load_pdf", None, "text", 0.72)],
        retrieval_mode="enhanced_rag",
        original_query="Вопрос",
        rewritten_query="load_pdf",
        alternative_queries=["PDF loader"],
        candidate_count=1,
        accepted_count=1,
        final_count=1,
        retrieval_diagnostics=diagnostics,
    )
    return baseline, enhanced, RetrievalConfig(candidate_top_k=5, final_top_k=2)


def test_day22_database_migrates_once_and_keeps_old_results(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite"
    create_day22_database(database)

    storage = ResultStorage(database)
    storage_again = ResultStorage(database)
    old = storage_again.get_run(1)

    assert storage.schema_version() == 2
    assert storage_again.schema_version() == 2
    assert old is not None
    assert old["question"] == "Старый вопрос"
    assert {answer["mode"] for answer in old["answers"]} == {"rag", "no_rag"}
    assert old["retrieval_candidates"] == []
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(rag_runs)")}
        assert {
            "rewritten_query",
            "candidate_top_k",
            "rerank_time",
            "rewrite_used",
            "rewrite_fallback_reason",
        } <= columns


def test_enhanced_diagnostics_save_restore_export_and_clear(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite"
    storage = ResultStorage(database)
    baseline, enhanced, config = enhanced_answers()

    run_id = storage.save_mode_comparison(
        (baseline, enhanced),
        strategy="structural",
        baseline_top_k=5,
        max_context_tokens=4000,
        temperature=0.1,
        retrieval_config=config,
    )
    restored = ResultStorage(database).get_run(run_id)

    assert restored is not None
    assert restored["rewritten_query"] == "load_pdf"
    assert restored["alternative_queries"] == ["PDF loader"]
    assert restored["rewrite_used"] == 0
    assert restored["rewrite_fallback_reason"] == "невалидный JSON"
    assert restored["retrieval_candidates"][0]["semantic_score"] == 0.86
    assert restored["retrieval_candidates"][0]["matched_queries"] == ["Вопрос", "load_pdf"]
    assert "content" not in restored["retrieval_candidates"][0]
    assert storage.retrieval_candidate_count() == 1

    deleted = clear_sqlite_results(database, "history")
    assert deleted == 4
    assert storage.retrieval_candidate_count() == 0
