"""Транзакционное хранилище RAG-результатов с безопасной миграцией Day 23."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import RAG_RESULTS_PATH, RESULTS_DATABASE
from .models import AnswerResult, RetrievalConfig

SCHEMA_VERSION = 2

_CREATE_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS rag_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        question TEXT NOT NULL,
        strategy TEXT NOT NULL,
        top_k INTEGER NOT NULL,
        max_context_tokens INTEGER NOT NULL,
        temperature REAL NOT NULL,
        model_name TEXT NOT NULL,
        retrieval_mode TEXT,
        original_query TEXT,
        rewritten_query TEXT,
        alternative_queries_json TEXT NOT NULL DEFAULT '[]',
        candidate_top_k INTEGER,
        final_top_k INTEGER,
        similarity_threshold REAL,
        rerank_threshold REAL,
        reranker_type TEXT,
        query_rewrite_enabled INTEGER NOT NULL DEFAULT 0,
        rewrite_used INTEGER NOT NULL DEFAULT 0,
        rewrite_fallback_reason TEXT,
        initial_candidate_count INTEGER NOT NULL DEFAULT 0,
        accepted_candidate_count INTEGER NOT NULL DEFAULT 0,
        final_candidate_count INTEGER NOT NULL DEFAULT 0,
        rewrite_time REAL NOT NULL DEFAULT 0,
        filter_time REAL NOT NULL DEFAULT 0,
        rerank_time REAL NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS rag_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        mode TEXT NOT NULL,
        answer TEXT NOT NULL,
        retrieval_time REAL NOT NULL DEFAULT 0,
        generation_time REAL NOT NULL DEFAULT 0,
        total_time REAL NOT NULL DEFAULT 0,
        context_token_count INTEGER NOT NULL DEFAULT 0,
        sources_json TEXT NOT NULL DEFAULT '[]',
        chunk_ids_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY (run_id) REFERENCES rag_runs(id) ON DELETE CASCADE,
        UNIQUE(run_id, mode)
    )""",
    """CREATE TABLE IF NOT EXISTS manual_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL UNIQUE,
        preferred_mode TEXT NOT NULL,
        score_without_rag INTEGER,
        score_with_rag INTEGER,
        score_baseline_rag INTEGER,
        score_enhanced_rag INTEGER,
        correct_sources INTEGER,
        has_irrelevant_sources INTEGER NOT NULL DEFAULT 0,
        has_unsupported_claims INTEGER NOT NULL DEFAULT 0,
        rewrite_improved INTEGER,
        comment TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES rag_runs(id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS benchmark_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        question_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        result_json TEXT NOT NULL,
        UNIQUE(batch_id, question_id)
    )""",
    """CREATE TABLE IF NOT EXISTS retrieval_candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        chunk_id TEXT NOT NULL,
        source TEXT NOT NULL,
        section TEXT,
        original_rank INTEGER,
        final_rank INTEGER,
        similarity REAL NOT NULL,
        fusion_score REAL,
        semantic_score REAL,
        lexical_score REAL,
        metadata_score REAL,
        rerank_score REAL,
        final_score REAL,
        status TEXT NOT NULL,
        rejection_reason TEXT,
        matched_queries_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY (run_id) REFERENCES rag_runs(id) ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_rag_answers_run_id ON rag_answers(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_batch_id ON benchmark_results(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_run ON retrieval_candidates(run_id)",
)

_RUN_MIGRATION_COLUMNS = {
    "retrieval_mode": "TEXT",
    "original_query": "TEXT",
    "rewritten_query": "TEXT",
    "alternative_queries_json": "TEXT NOT NULL DEFAULT '[]'",
    "candidate_top_k": "INTEGER",
    "final_top_k": "INTEGER",
    "similarity_threshold": "REAL",
    "rerank_threshold": "REAL",
    "reranker_type": "TEXT",
    "query_rewrite_enabled": "INTEGER NOT NULL DEFAULT 0",
    "rewrite_used": "INTEGER NOT NULL DEFAULT 0",
    "rewrite_fallback_reason": "TEXT",
    "initial_candidate_count": "INTEGER NOT NULL DEFAULT 0",
    "accepted_candidate_count": "INTEGER NOT NULL DEFAULT 0",
    "final_candidate_count": "INTEGER NOT NULL DEFAULT 0",
    "rewrite_time": "REAL NOT NULL DEFAULT 0",
    "filter_time": "REAL NOT NULL DEFAULT 0",
    "rerank_time": "REAL NOT NULL DEFAULT 0",
}

_EVALUATION_MIGRATION_COLUMNS = {
    "score_baseline_rag": "INTEGER",
    "score_enhanced_rag": "INTEGER",
    "correct_sources": "INTEGER",
    "has_irrelevant_sources": "INTEGER NOT NULL DEFAULT 0",
    "rewrite_improved": "INTEGER",
}

RESULT_TABLES: dict[str, tuple[str, ...]] = {
    "history": ("retrieval_candidates", "manual_evaluations", "rag_answers", "rag_runs"),
    "benchmark": ("benchmark_results",),
    "all_results": (
        "retrieval_candidates",
        "manual_evaluations",
        "rag_answers",
        "rag_runs",
        "benchmark_results",
    ),
}

_ALLOWED_PREFERENCES = {
    "no_rag",
    "rag",
    "baseline_rag",
    "filtered_rag",
    "enhanced_rag",
    "equal",
    "both_wrong",
    "all_incorrect",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _migrate(connection: sqlite3.Connection) -> None:
    """Транзакционно привести существующую Day 22 схему к версии 2."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        if not _table_exists(connection, "rag_runs"):
            for statement in _CREATE_STATEMENTS:
                connection.execute(statement)
        else:
            run_columns = _columns(connection, "rag_runs")
            for name, definition in _RUN_MIGRATION_COLUMNS.items():
                if name not in run_columns:
                    connection.execute(f"ALTER TABLE rag_runs ADD COLUMN {name} {definition}")

            evaluation_columns = _columns(connection, "manual_evaluations")
            for name, definition in _EVALUATION_MIGRATION_COLUMNS.items():
                if name not in evaluation_columns:
                    connection.execute(
                        f"ALTER TABLE manual_evaluations ADD COLUMN {name} {definition}"
                    )

            answer_sql_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'rag_answers'"
            ).fetchone()
            answer_sql = str(answer_sql_row[0] or "") if answer_sql_row else ""
            if "CHECK" in answer_sql.upper() and "baseline_rag" not in answer_sql:
                connection.execute("DROP INDEX IF EXISTS idx_rag_answers_run_id")
                connection.execute("ALTER TABLE rag_answers RENAME TO rag_answers_day22")
                connection.execute(_CREATE_STATEMENTS[1])
                connection.execute(
                    """INSERT INTO rag_answers (
                        id, run_id, mode, answer, retrieval_time, generation_time,
                        total_time, context_token_count, sources_json, chunk_ids_json
                    ) SELECT id, run_id, mode, answer, retrieval_time, generation_time,
                        total_time, context_token_count, sources_json, chunk_ids_json
                    FROM rag_answers_day22"""
                )
                connection.execute("DROP TABLE rag_answers_day22")

            for statement in _CREATE_STATEMENTS[2:]:
                connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


class ResultStorage:
    """SQLite-хранилище сравнений, диагностик, оценок и benchmark-результатов."""

    def __init__(self, database_path: Path = RESULTS_DATABASE) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            _migrate(connection)
        finally:
            connection.close()

    def schema_version(self) -> int:
        """Вернуть текущую версию схемы результатов."""
        with self._connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def save_mode_comparison(
        self,
        answers: Iterable[AnswerResult],
        *,
        strategy: str,
        baseline_top_k: int,
        max_context_tokens: int,
        temperature: float,
        retrieval_config: RetrievalConfig | None = None,
    ) -> int:
        """Атомарно сохранить запуск, все ответы и retrieval-диагностику."""
        answer_list = list(answers)
        if len(answer_list) < 2:
            raise ValueError("Сравнение должно содержать минимум два ответа.")
        if len({answer.mode for answer in answer_list}) != len(answer_list):
            raise ValueError("Режимы ответов одного сравнения не должны повторяться.")
        if len({answer.question for answer in answer_list}) != 1:
            raise ValueError("Нельзя сохранить ответы на разные вопросы одним запуском.")
        if len({answer.model_name for answer in answer_list}) != 1:
            raise ValueError("Для честного сравнения требуется одна модель.")

        enhanced = next(
            (answer for answer in answer_list if answer.retrieval_mode == "enhanced_rag"),
            answer_list[-1],
        )
        config = retrieval_config
        answer_rows = [
            (
                answer.mode,
                answer.answer,
                answer.retrieval_time,
                answer.generation_time,
                answer.total_time,
                answer.context_token_count,
                _json(answer.sources),
                _json([chunk.chunk_id for chunk in answer.retrieved_chunks]),
            )
            for answer in answer_list
        ]
        raw_candidates = enhanced.retrieval_diagnostics.get("initial_candidates", [])
        candidate_rows = [
            (
                str(item["chunk_id"]),
                str(item["source"]),
                item.get("section"),
                item.get("original_rank"),
                item.get("final_rank"),
                float(item.get("similarity", 0.0)),
                item.get("fusion_score"),
                item.get("semantic_score"),
                item.get("lexical_score"),
                item.get("metadata_score"),
                item.get("rerank_score"),
                item.get("final_score"),
                str(item.get("status", "candidate")),
                item.get("rejection_reason"),
                _json(item.get("matched_queries", [])),
            )
            for item in raw_candidates
        ]

        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO rag_runs (
                    created_at, question, strategy, top_k, max_context_tokens,
                    temperature, model_name, retrieval_mode, original_query,
                    rewritten_query, alternative_queries_json, candidate_top_k,
                    final_top_k, similarity_threshold, rerank_threshold,
                    reranker_type, query_rewrite_enabled, rewrite_used,
                    rewrite_fallback_reason, initial_candidate_count,
                    accepted_candidate_count, final_candidate_count, rewrite_time,
                    filter_time, rerank_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _now(),
                    answer_list[0].question,
                    strategy,
                    baseline_top_k,
                    max_context_tokens,
                    temperature,
                    answer_list[0].model_name,
                    "_vs_".join(answer.mode for answer in answer_list),
                    enhanced.original_query or answer_list[0].question,
                    enhanced.rewritten_query,
                    _json(enhanced.alternative_queries),
                    config.candidate_top_k if config else None,
                    config.final_top_k if config else enhanced.top_k,
                    config.similarity_threshold if config else None,
                    config.rerank_threshold if config else None,
                    config.reranker_type if config else None,
                    int(config.use_query_rewrite) if config else 0,
                    int(
                        bool(
                            enhanced.retrieval_diagnostics.get("rewrite_used", False)
                        )
                    ),
                    enhanced.retrieval_diagnostics.get("fallback_reason"),
                    enhanced.candidate_count,
                    enhanced.accepted_count,
                    enhanced.final_count,
                    enhanced.rewrite_time,
                    enhanced.filter_time,
                    enhanced.rerank_time,
                ),
            )
            run_id = int(cursor.lastrowid)
            connection.executemany(
                """INSERT INTO rag_answers (
                    run_id, mode, answer, retrieval_time, generation_time,
                    total_time, context_token_count, sources_json, chunk_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(run_id, *row) for row in answer_rows],
            )
            connection.executemany(
                """INSERT INTO retrieval_candidates (
                    run_id, chunk_id, source, section, original_rank, final_rank,
                    similarity, fusion_score, semantic_score, lexical_score,
                    metadata_score, rerank_score, final_score, status,
                    rejection_reason, matched_queries_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(run_id, *row) for row in candidate_rows],
            )
        return run_id

    def save_comparison(
        self,
        without_rag: AnswerResult,
        with_rag: AnswerResult,
        *,
        strategy: str,
        top_k: int,
        max_context_tokens: int,
        temperature: float,
    ) -> int:
        """Совместимый Day 22 wrapper сохранения пары no-RAG/RAG."""
        if without_rag.mode != "no_rag" or with_rag.mode != "rag":
            raise ValueError("Ожидаются ответы no_rag и rag соответственно.")
        return self.save_mode_comparison(
            (without_rag, with_rag),
            strategy=strategy,
            baseline_top_k=top_k,
            max_context_tokens=max_context_tokens,
            temperature=temperature,
        )

    def save_manual_evaluation(
        self,
        run_id: int,
        *,
        preferred_mode: str,
        score_without_rag: int | None = None,
        score_with_rag: int | None = None,
        score_baseline_rag: int | None = None,
        score_enhanced_rag: int | None = None,
        correct_sources: bool | None = None,
        has_irrelevant_sources: bool = False,
        has_unsupported_claims: bool = False,
        rewrite_improved: bool | None = None,
        comment: str = "",
    ) -> None:
        """Создать или обновить единственную расширенную оценку запуска."""
        if preferred_mode not in _ALLOWED_PREFERENCES:
            raise ValueError("Некорректный выбор лучшего ответа.")
        scores = (
            score_without_rag,
            score_with_rag,
            score_baseline_rag,
            score_enhanced_rag,
        )
        if any(score is not None and score not in (0, 1, 2) for score in scores):
            raise ValueError("Оценки должны находиться в диапазоне от 0 до 2.")
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM rag_runs WHERE id = ?", (run_id,)
            ).fetchone() is None:
                raise ValueError("Сравнение для оценки не найдено.")
            connection.execute(
                """INSERT INTO manual_evaluations (
                    run_id, preferred_mode, score_without_rag, score_with_rag,
                    score_baseline_rag, score_enhanced_rag, correct_sources,
                    has_irrelevant_sources, has_unsupported_claims,
                    rewrite_improved, comment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    preferred_mode = excluded.preferred_mode,
                    score_without_rag = excluded.score_without_rag,
                    score_with_rag = excluded.score_with_rag,
                    score_baseline_rag = excluded.score_baseline_rag,
                    score_enhanced_rag = excluded.score_enhanced_rag,
                    correct_sources = excluded.correct_sources,
                    has_irrelevant_sources = excluded.has_irrelevant_sources,
                    has_unsupported_claims = excluded.has_unsupported_claims,
                    rewrite_improved = excluded.rewrite_improved,
                    comment = excluded.comment,
                    created_at = excluded.created_at""",
                (
                    run_id,
                    preferred_mode,
                    score_without_rag,
                    score_with_rag,
                    score_baseline_rag,
                    score_enhanced_rag,
                    None if correct_sources is None else int(correct_sources),
                    int(has_irrelevant_sources),
                    int(has_unsupported_claims),
                    None if rewrite_improved is None else int(rewrite_improved),
                    comment.strip(),
                    _now(),
                ),
            )

    def save_benchmark_result(
        self,
        batch_id: str,
        question_id: str,
        result: dict[str, Any],
    ) -> None:
        """Надёжно сохранить результат одного контрольного вопроса."""
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO benchmark_results (
                    batch_id, question_id, created_at, result_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_id, question_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    result_json = excluded.result_json""",
                (batch_id, question_id, _now(), _json(result)),
            )

    def _answer_map(self, run_ids: list[int]) -> dict[int, dict[str, str]]:
        if not run_ids:
            return {}
        placeholders = ",".join("?" for _ in run_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT run_id, mode, answer FROM rag_answers WHERE run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
        output: dict[int, dict[str, str]] = defaultdict(dict)
        for row in rows:
            output[int(row["run_id"])][str(row["mode"])] = str(row["answer"])
        return dict(output)

    def list_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Прочитать старую и новую историю независимо от session_state."""
        if limit <= 0:
            raise ValueError("Лимит истории должен быть больше нуля.")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT r.*, e.preferred_mode, e.score_without_rag,
                    e.score_with_rag, e.score_baseline_rag, e.score_enhanced_rag,
                    e.correct_sources, e.has_irrelevant_sources,
                    e.has_unsupported_claims, e.rewrite_improved,
                    e.comment AS evaluation_comment
                FROM rag_runs r
                LEFT JOIN manual_evaluations e ON e.run_id = r.id
                ORDER BY r.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        output = [dict(row) for row in rows]
        answers = self._answer_map([int(row["id"]) for row in output])
        for row in output:
            by_mode = answers.get(int(row["id"]), {})
            row["answers_by_mode"] = by_mode
            row["answer_without_rag"] = by_mode.get("no_rag", "")
            row["answer_with_rag"] = by_mode.get("rag", "")
            row["answer_baseline_rag"] = by_mode.get("baseline_rag", by_mode.get("rag", ""))
            row["answer_enhanced_rag"] = by_mode.get("enhanced_rag", "")
        return output

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Получить запуск, ответы, оценки и retrieval-диагностику."""
        with self._connect() as connection:
            run = connection.execute(
                "SELECT * FROM rag_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                return None
            answers = connection.execute(
                "SELECT * FROM rag_answers WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
            evaluation = connection.execute(
                "SELECT * FROM manual_evaluations WHERE run_id = ?", (run_id,)
            ).fetchone()
            candidates = connection.execute(
                "SELECT * FROM retrieval_candidates WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        payload = dict(run)
        payload["alternative_queries"] = json.loads(
            payload.pop("alternative_queries_json") or "[]"
        )
        payload["answers"] = []
        for row in answers:
            answer = dict(row)
            answer["sources"] = json.loads(answer.pop("sources_json"))
            answer["chunk_ids"] = json.loads(answer.pop("chunk_ids_json"))
            payload["answers"].append(answer)
        payload["retrieval_candidates"] = []
        for row in candidates:
            candidate = dict(row)
            candidate["matched_queries"] = json.loads(
                candidate.pop("matched_queries_json") or "[]"
            )
            payload["retrieval_candidates"].append(candidate)
        payload["manual_evaluation"] = dict(evaluation) if evaluation else None
        return payload

    def benchmark_results(self, batch_id: str | None = None) -> list[dict[str, Any]]:
        """Прочитать результаты последнего или выбранного benchmark-пакета."""
        with self._connect() as connection:
            selected_batch = batch_id
            if selected_batch is None:
                row = connection.execute(
                    "SELECT batch_id FROM benchmark_results ORDER BY id DESC LIMIT 1"
                ).fetchone()
                selected_batch = str(row[0]) if row else None
            if selected_batch is None:
                return []
            rows = connection.execute(
                "SELECT * FROM benchmark_results WHERE batch_id = ? ORDER BY id",
                (selected_batch,),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json"))
            output.append(item)
        return output

    def benchmark_results_all(self) -> list[dict[str, Any]]:
        """Прочитать все результаты контрольных прогонов."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM benchmark_results ORDER BY id"
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json"))
            output.append(item)
        return output

    def latest_reranking_results(self) -> list[dict[str, Any]]:
        """Восстановить последний пакет Baseline/Enhanced после перезапуска."""
        rows = self.benchmark_results_all()
        for row in reversed(rows):
            payload = row["result"]
            if "question" in payload and (
                "baseline" in payload or "enhanced" in payload or payload.get("error")
            ):
                return self.benchmark_results(str(row["batch_id"]))
        return []

    def latest_threshold_results(self) -> list[dict[str, Any]]:
        """Восстановить метрики последнего threshold experiment."""
        rows = self.benchmark_results_all()
        for row in reversed(rows):
            payload = row["result"]
            if payload.get("type") == "threshold_experiment":
                batch = self.benchmark_results(str(row["batch_id"]))
                return [item["result"]["metrics"] for item in batch]
        return []

    def counts(self) -> dict[str, int]:
        """Вернуть счётчики фиксированного набора таблиц."""
        with self._connect() as connection:
            return {
                "comparisons": int(connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0]),
                "answers": int(connection.execute("SELECT COUNT(*) FROM rag_answers").fetchone()[0]),
                "manual_evaluations": int(connection.execute("SELECT COUNT(*) FROM manual_evaluations").fetchone()[0]),
                "benchmark_results": int(connection.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0]),
            }

    def retrieval_candidate_count(self) -> int:
        """Вернуть число сохранённых диагностических кандидатов."""
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM retrieval_candidates"
                ).fetchone()[0]
            )

    def export_json(self, path: Path = RAG_RESULTS_PATH) -> Path:
        """Атомарно экспортировать результаты и диагностику в совместимый JSON."""
        history = [self.get_run(int(row["id"])) for row in self.list_history(limit=1_000_000)]
        payload = {
            "version": 1,
            "exported_at": _now(),
            "results": [
                *({"type": "comparison", "data": item} for item in history if item),
                *({"type": "benchmark", "data": item} for item in self.benchmark_results_all()),
            ],
        }
        return _atomic_json_write(Path(path), payload)


def clear_sqlite_results(database_path: Path, scope: str) -> int:
    """Транзакционно очистить только таблицы из фиксированного allowlist."""
    tables = RESULT_TABLES.get(scope)
    if tables is None:
        raise ValueError("Неизвестная область сброса результатов.")
    storage = ResultStorage(Path(database_path))
    deleted = 0
    with storage._connect() as connection:
        for table in tables:
            cursor = connection.execute(f"DELETE FROM {table}")
            deleted += max(0, int(cursor.rowcount))
    return deleted


def clear_json_results(path: Path = RAG_RESULTS_PATH) -> None:
    """Атомарно заменить JSON результатов корректной совместимой структурой."""
    _atomic_json_write(Path(path), {"version": 1, "results": []})
