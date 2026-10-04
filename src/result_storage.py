"""Отдельное транзакционное хранилище результатов RAG."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RAG_RESULTS_PATH, RESULTS_DATABASE
from .models import AnswerResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    strategy TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    max_context_tokens INTEGER NOT NULL,
    temperature REAL NOT NULL,
    model_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_answers (
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
CREATE TABLE IF NOT EXISTS manual_evaluations (
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
CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL,
    UNIQUE(batch_id, question_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_answers_run_id ON rag_answers(run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_batch_id ON benchmark_results(batch_id);
"""

RESULT_TABLES: dict[str, tuple[str, ...]] = {
    "history": ("manual_evaluations", "rag_answers", "rag_runs"),
    "benchmark": ("benchmark_results",),
    "all_results": (
        "manual_evaluations",
        "rag_answers",
        "rag_runs",
        "benchmark_results",
    ),
}

_ALLOWED_PREFERENCES = {"no_rag", "rag", "equal", "both_wrong"}


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


class ResultStorage:
    """SQLite-хранилище сравнений, оценок и benchmark-результатов."""

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
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

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
        """Сохранить запуск и два ответа атомарно одной транзакцией."""
        if without_rag.mode != "no_rag" or with_rag.mode != "rag":
            raise ValueError("Ожидаются ответы no_rag и rag соответственно.")
        if without_rag.question != with_rag.question:
            raise ValueError("Нельзя сохранить ответы на разные вопросы одним запуском.")
        if without_rag.model_name != with_rag.model_name:
            raise ValueError("Для честного сравнения требуется одна модель.")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO rag_runs (
                    created_at, question, strategy, top_k,
                    max_context_tokens, temperature, model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    without_rag.question,
                    strategy,
                    top_k,
                    max_context_tokens,
                    temperature,
                    without_rag.model_name,
                ),
            )
            run_id = int(cursor.lastrowid)
            answers = (without_rag, with_rag)
            connection.executemany(
                """
                INSERT INTO rag_answers (
                    run_id, mode, answer, retrieval_time, generation_time,
                    total_time, context_token_count, sources_json, chunk_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        answer.mode,
                        answer.answer,
                        answer.retrieval_time,
                        answer.generation_time,
                        answer.total_time,
                        answer.context_token_count,
                        _json(answer.sources),
                        _json([chunk.chunk_id for chunk in answer.retrieved_chunks]),
                    )
                    for answer in answers
                ],
            )
        return run_id

    def save_manual_evaluation(
        self,
        run_id: int,
        *,
        preferred_mode: str,
        score_without_rag: int,
        score_with_rag: int,
        has_unsupported_claims: bool,
        comment: str = "",
    ) -> None:
        """Создать или обновить единственную ручную оценку запуска."""
        if preferred_mode not in _ALLOWED_PREFERENCES:
            raise ValueError("Некорректный выбор лучшего ответа.")
        if score_without_rag not in (0, 1, 2) or score_with_rag not in (0, 1, 2):
            raise ValueError("Оценки должны находиться в диапазоне от 0 до 2.")
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM rag_runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise ValueError("Сравнение для оценки не найдено.")
            connection.execute(
                """
                INSERT INTO manual_evaluations (
                    run_id, preferred_mode, score_without_rag, score_with_rag,
                    has_unsupported_claims, comment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    preferred_mode = excluded.preferred_mode,
                    score_without_rag = excluded.score_without_rag,
                    score_with_rag = excluded.score_with_rag,
                    has_unsupported_claims = excluded.has_unsupported_claims,
                    comment = excluded.comment,
                    created_at = excluded.created_at
                """,
                (
                    run_id,
                    preferred_mode,
                    score_without_rag,
                    score_with_rag,
                    int(has_unsupported_claims),
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
                """
                INSERT INTO benchmark_results (
                    batch_id, question_id, created_at, result_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_id, question_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    result_json = excluded.result_json
                """,
                (batch_id, question_id, _now(), _json(result)),
            )

    def list_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Прочитать сохранённую историю независимо от session_state."""
        if limit <= 0:
            raise ValueError("Лимит истории должен быть больше нуля.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, nr.answer AS answer_without_rag, wr.answer AS answer_with_rag,
                       e.preferred_mode, e.score_without_rag, e.score_with_rag,
                       e.has_unsupported_claims, e.comment AS evaluation_comment
                FROM rag_runs r
                JOIN rag_answers nr ON nr.run_id = r.id AND nr.mode = 'no_rag'
                JOIN rag_answers wr ON wr.run_id = r.id AND wr.mode = 'rag'
                LEFT JOIN manual_evaluations e ON e.run_id = r.id
                ORDER BY r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Получить запуск с ответами, источниками и ручной оценкой."""
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM rag_runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            answers = connection.execute(
                "SELECT * FROM rag_answers WHERE run_id = ? ORDER BY mode", (run_id,)
            ).fetchall()
            evaluation = connection.execute(
                "SELECT * FROM manual_evaluations WHERE run_id = ?", (run_id,)
            ).fetchone()
        payload = dict(run)
        payload["answers"] = []
        for row in answers:
            answer = dict(row)
            answer["sources"] = json.loads(answer.pop("sources_json"))
            answer["chunk_ids"] = json.loads(answer.pop("chunk_ids_json"))
            payload["answers"].append(answer)
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

    def counts(self) -> dict[str, int]:
        """Вернуть безопасные счётчики фиксированного набора таблиц."""
        with self._connect() as connection:
            return {
                "comparisons": int(connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0]),
                "answers": int(connection.execute("SELECT COUNT(*) FROM rag_answers").fetchone()[0]),
                "manual_evaluations": int(
                    connection.execute("SELECT COUNT(*) FROM manual_evaluations").fetchone()[0]
                ),
                "benchmark_results": int(
                    connection.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0]
                ),
            }

    def export_json(self, path: Path = RAG_RESULTS_PATH) -> Path:
        """Атомарно экспортировать все результаты SQLite в JSON."""
        history = [self.get_run(int(row["id"])) for row in self.list_history(limit=1_000_000)]
        benchmarks = self.benchmark_results_all()
        payload = {
            "version": 1,
            "exported_at": _now(),
            "results": [
                *({"type": "comparison", "data": item} for item in history if item),
                *({"type": "benchmark", "data": item} for item in benchmarks),
            ],
        }
        return _atomic_json_write(Path(path), payload)

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
    """Атомарно заменить JSON результатов корректной пустой структурой."""
    _atomic_json_write(Path(path), {"version": 1, "results": []})
