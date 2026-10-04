from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import Chunk
from src.rag import SemanticRetriever, answer_question, compare_answers
from src.result_storage import ResultStorage, clear_sqlite_results
from src.storage import IndexNotFoundError, IndexStorage


class IntegrationLLM:
    model_name = "integration-fake"

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, temperature=0.1):
        self.messages.append(messages)
        if "КОНТЕКСТ:" in messages[-1]["content"]:
            return "Создайте окружение и выполните streamlit run app.py [1]"
        return "Используйте стандартную команду запуска."


def test_full_rag_path_persists_exports_and_clears_safely(
    tmp_path: Path, fake_embeddings
) -> None:
    index_root = tmp_path / "indexes"
    contents = [
        "Для установки создайте окружение и выполните streamlit run app.py",
        "FAISS хранит нормализованные векторы",
    ]
    chunks = [
        Chunk(
            chunk_id=f"chunk-{index}",
            document_id="doc",
            source="guide.md",
            title="Guide",
            section="Установка" if index == 0 else "Поиск",
            strategy="structural",
            content=content,
            token_count=len(content.split()),
        )
        for index, content in enumerate(contents)
    ]
    vectors = np.asarray(fake_embeddings.encode_documents(contents), dtype=np.float32)
    index_storage = IndexStorage("structural", index_root)
    index_storage.save(vectors, chunks, {"strategy": "structural", "chunk_count": 2})
    retriever = SemanticRetriever(fake_embeddings, min_score=-1.0, index_root=index_root)
    llm = IntegrationLLM()

    without_rag, with_rag = compare_answers(
        "Как запустить приложение?",
        llm,
        strategy="structural",
        retriever=retriever,
        tokenizer=fake_embeddings.tokenizer,
    )
    result_database = tmp_path / "data" / "results.sqlite"
    results = ResultStorage(result_database)
    run_id = results.save_comparison(
        without_rag,
        with_rag,
        strategy="structural",
        top_k=5,
        max_context_tokens=4000,
        temperature=0.1,
    )
    results.save_manual_evaluation(
        run_id,
        preferred_mode="rag",
        score_without_rag=1,
        score_with_rag=2,
        has_unsupported_claims=False,
    )
    exported = results.export_json(tmp_path / "evaluation" / "results.json")

    assert len(llm.messages) == 2
    assert without_rag.sources == []
    assert with_rag.sources[0]["source"] == "guide.md"
    assert "[1]" in with_rag.answer
    assert len(ResultStorage(result_database).list_history()) == 1
    assert exported.is_file()
    assert ResultStorage(result_database).counts()["manual_evaluations"] == 1

    clear_sqlite_results(result_database, "all_results")

    assert ResultStorage(result_database).counts() == {
        "comparisons": 0,
        "answers": 0,
        "manual_evaluations": 0,
        "benchmark_results": 0,
    }
    assert index_storage.exists
    index_storage.validate()
    assert index_storage.load_index().ntotal == 2


def test_rag_reports_missing_index_before_llm_call(tmp_path: Path, fake_embeddings) -> None:
    llm = IntegrationLLM()
    retriever = SemanticRetriever(fake_embeddings, index_root=tmp_path / "missing")

    with pytest.raises(IndexNotFoundError, match="не создан"):
        answer_question("Вопрос", "rag", llm, retriever=retriever)

    assert llm.messages == []
