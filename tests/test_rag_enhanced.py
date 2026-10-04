from __future__ import annotations

from dataclasses import replace

import json

from src.models import RetrievalConfig, RetrievedChunk
from src.rag import answer_question, compare_retrieval_modes
from src.text_utils import RegexTokenizer


class PipelineLLM:
    model_name = "pipeline-fake"

    def __init__(self, rewrite_response: str | None = None) -> None:
        self.rewrite_response = rewrite_response or json.dumps(
            {"search_query": "функция load_pdf", "alternative_queries": []}
        )
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages, temperature=0.1):
        self.calls.append(messages)
        if "Преобразуй вопрос" in messages[0]["content"]:
            return self.rewrite_response
        return "Ответ по переданному контексту [1]"


class BaselineRetriever:
    tokenizer = RegexTokenizer()

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str, strategy: str, top_k: int):
        self.calls.append(query)
        return [
            RetrievedChunk(
                "baseline",
                "guide.md",
                "Guide",
                "Установка",
                None,
                "baseline context",
                0.10,
                3,
            )
        ][:top_k]


class PipelineSearcher:
    tokenizer = RegexTokenizer()

    def __init__(self, only_low: bool = False) -> None:
        self.only_low = only_low
        self.calls: list[str] = []

    def search(self, query: str, strategy: str, top_k: int):
        self.calls.append(query)
        rows = [
            {
                "chunk_id": "low",
                "source": "wrong.txt",
                "title": "Wrong",
                "section": None,
                "page_number": None,
                "content": "СЕКРЕТНЫЙ НЕРЕЛЕВАНТНЫЙ ТЕКСТ",
                "score": 0.20,
                "token_count": 4,
            }
        ]
        if not self.only_low:
            rows.insert(
                0,
                {
                    "chunk_id": "high",
                    "source": "sample_module.py",
                    "title": "Code",
                    "section": "load_pdf",
                    "page_number": None,
                    "content": "def load_pdf(path): return pages",
                    "score": 0.85,
                    "token_count": 8,
                },
            )
        return rows[:top_k]


def config() -> RetrievalConfig:
    return RetrievalConfig(
        candidate_top_k=5,
        final_top_k=3,
        similarity_threshold=0.35,
        max_chunks_per_source=3,
    )


def test_baseline_does_not_rewrite_filter_or_rerank() -> None:
    llm = PipelineLLM()
    retriever = BaselineRetriever()

    result = answer_question(
        "Как запустить?",
        "baseline_rag",
        llm,
        retriever=retriever,
        tokenizer=RegexTokenizer(),
    )

    assert retriever.calls == ["Как запустить?"]
    assert len(llm.calls) == 1
    assert "Преобразуй вопрос" not in llm.calls[0][0]["content"]
    assert result.retrieved_chunks[0].score == 0.10
    assert result.retrieval_diagnostics["filter_time"] == 0.0


def test_enhanced_never_sends_below_threshold_chunk_to_llm() -> None:
    llm = PipelineLLM()
    searcher = PipelineSearcher()

    result = answer_question(
        "Где загружается PDF?",
        "enhanced_rag",
        llm,
        retrieval_config=config(),
        candidate_searcher=searcher,
        tokenizer=RegexTokenizer(),
    )
    generation_prompt = "\n".join(message["content"] for message in llm.calls[-1])

    assert searcher.calls[0] == "Где загружается PDF?"
    assert "функция load_pdf" in searcher.calls
    assert "def load_pdf" in generation_prompt
    assert "СЕКРЕТНЫЙ" not in generation_prompt
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == ["high"]
    low = next(
        item
        for item in result.retrieval_diagnostics["initial_candidates"]
        if item["chunk_id"] == "low"
    )
    assert low["status"] == "rejected_similarity"
    assert low["rejection_reason"]


def test_filtered_rag_filters_without_query_rewrite_or_heuristic_reranking() -> None:
    llm = PipelineLLM()
    result = answer_question(
        "Где загружается PDF?",
        "filtered_rag",
        llm,
        retrieval_config=replace(config(), rerank_threshold=0.99),
        candidate_searcher=PipelineSearcher(),
        tokenizer=RegexTokenizer(),
    )

    assert len(llm.calls) == 1
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == ["high"]
    high = next(
        item
        for item in result.retrieval_diagnostics["initial_candidates"]
        if item["chunk_id"] == "high"
    )
    assert high["lexical_score"] is None
    assert result.rewritten_query is None


def test_empty_filtered_result_skips_answer_generation_and_keeps_diagnostics() -> None:
    llm = PipelineLLM("невалидный JSON")
    searcher = PipelineSearcher(only_low=True)

    result = answer_question(
        "Нет ответа",
        "enhanced_rag",
        llm,
        retrieval_config=config(),
        candidate_searcher=searcher,
        tokenizer=RegexTokenizer(),
    )

    assert searcher.calls == ["Нет ответа"]
    assert len(llm.calls) == 1
    assert "не найден" in result.answer
    assert result.sources == []
    assert result.retrieval_diagnostics["fallback_reason"]
    assert result.retrieval_diagnostics["final_count"] == 0


def test_comparison_uses_same_generation_settings() -> None:
    llm = PipelineLLM()
    baseline, enhanced = compare_retrieval_modes(
        "Где загружается PDF?",
        llm,
        temperature=0.25,
        retriever=BaselineRetriever(),
        candidate_searcher=PipelineSearcher(),
        tokenizer=RegexTokenizer(),
        retrieval_config=config(),
    )

    assert baseline.question == enhanced.question
    assert baseline.model_name == enhanced.model_name
    assert baseline.retrieval_mode == "baseline_rag"
    assert enhanced.retrieval_mode == "enhanced_rag"
