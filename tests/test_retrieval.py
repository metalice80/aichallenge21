from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.models import RetrievalCandidate, RetrievalConfig
from src.retrieval import (
    HeuristicReranker,
    NoOpReranker,
    enhanced_retrieve,
    filter_by_similarity,
    limit_chunks_per_source,
    reciprocal_rank_fusion,
    rewrite_query,
)


class RewriteLLM:
    model_name = "rewrite-fake"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def generate(self, messages, temperature=0.1):
        self.calls.append((messages, temperature))
        return self.response


class FakeSearcher:
    def __init__(self, results: dict[str, list[dict]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, strategy: str, top_k: int):
        self.queries.append(query)
        return self.results.get(query, [])[:top_k]


class FakeReranker:
    name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        self.queries.append(query)
        return [
            replace(item, final_score=1.0, final_rank=rank)
            for rank, item in enumerate(reversed(candidates), start=1)
        ]

def row(chunk_id: str, score: float, *, source: str = "guide.md", section: str = "Раздел"):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "title": "Guide",
        "section": section,
        "page_number": None,
        "content": f"текст {chunk_id} функция load_pdf IndexFlatIP",
        "score": score,
        "token_count": 8,
    }


def candidate(chunk_id: str, score: float, rank: int = 1, *, source: str = "guide.md", section: str = "Раздел"):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source=source,
        title="Guide",
        section=section,
        page_number=None,
        content=f"текст {chunk_id} функция load_pdf IndexFlatIP",
        similarity=score,
        original_rank=rank,
        matched_queries=["вопрос"],
    )


def test_query_rewrite_validates_json_and_deduplicates_variants() -> None:
    llm = RewriteLLM(
        json.dumps(
            {
                "search_query": "функция load_pdf",
                "alternative_queries": ["", "функция load_pdf", "загрузка PDF", "ЗАГРУЗКА pdf"],
            }
        )
    )

    result = rewrite_query("Где загружается PDF?", llm, 3)

    assert result.search_query == "функция load_pdf"
    assert result.alternative_queries == ["загрузка PDF"]
    assert result.rewrite_used is True
    assert result.fallback_reason is None
    assert llm.calls[0][1] == 0.0
    assert "Не отвечай" in llm.calls[0][0][0]["content"]


@pytest.mark.parametrize("response", ["не json", "{}", '{"search_query":""}'])
def test_invalid_rewrite_falls_back_to_original_question(response: str) -> None:
    result = rewrite_query("Исходный вопрос", RewriteLLM(response))

    assert result.search_query == "Исходный вопрос"
    assert result.rewrite_used is False
    assert result.fallback_reason and "не применён" in result.fallback_reason


def test_rrf_deduplicates_and_is_deterministic() -> None:
    first = [candidate("a", 0.8, 1), candidate("b", 0.7, 2)]
    second = [candidate("b", 0.9, 1), candidate("a", 0.8, 2)]

    fused = reciprocal_rank_fusion([first, second])

    assert [item.chunk_id for item in fused] == ["b", "a"]
    assert fused[0].fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].similarity == 0.9


def test_similarity_filter_accepts_boundary_and_explains_rejection() -> None:
    accepted, rejected = filter_by_similarity(
        [candidate("equal", 0.35), candidate("low", 0.349)], 0.35
    )

    assert [item.chunk_id for item in accepted] == ["equal"]
    assert rejected[0].status == "rejected_similarity"
    assert "0.349" in (rejected[0].rejection_reason or "")


def test_noop_preserves_order_and_heuristic_saves_score_components() -> None:
    items = [
        candidate("plain", 0.90, section="Общее"),
        candidate("function", 0.70, section="load_pdf"),
    ]

    unchanged = NoOpReranker().rerank("load_pdf", items)
    reranked = HeuristicReranker().rerank("Как работает load_pdf?", items)

    assert [item.chunk_id for item in unchanged] == ["plain", "function"]
    assert all(item.final_rank for item in unchanged)
    assert unchanged[0].final_score == pytest.approx(0.95)
    function = next(item for item in reranked if item.chunk_id == "function")
    assert function.semantic_score is not None
    assert function.lexical_score is not None
    assert function.metadata_score and function.metadata_score > 0
    assert function.final_score == pytest.approx(
        0.7 * function.semantic_score
        + 0.2 * function.lexical_score
        + 0.1 * function.metadata_score
    )


def test_source_limit_keeps_best_ranked_candidates() -> None:
    accepted, rejected = limit_chunks_per_source(
        [candidate("a", 0.9), candidate("b", 0.8), candidate("c", 0.7, source="code.py")],
        1,
    )

    assert [item.chunk_id for item in accepted] == ["a", "c"]
    assert rejected[0].chunk_id == "b"
    assert rejected[0].status == "rejected_source_limit"


def test_enhanced_retrieval_always_searches_original_and_assigns_statuses() -> None:
    question = "Где загружается PDF?"
    rewritten = "функция load_pdf"
    llm = RewriteLLM(json.dumps({"search_query": rewritten, "alternative_queries": ["PDF loader"]}))
    searcher = FakeSearcher(
        {
            question: [row("original", 0.75), row("duplicate", 0.34)],
            rewritten: [row("rewritten", 0.90), row("original", 0.70)],
            "PDF loader": [row("third", 0.65, source="code.py")],
        }
    )
    config = RetrievalConfig(
        candidate_top_k=4,
        final_top_k=2,
        similarity_threshold=0.35,
        max_chunks_per_source=2,
    )

    result = enhanced_retrieve(question, "structural", config, llm, searcher=searcher)

    assert searcher.queries[0] == question
    assert set(searcher.queries) == {question, rewritten, "PDF loader"}
    assert result.candidate_count == 4
    assert result.accepted_count == 3
    assert result.final_count == 2
    assert {item.status for item in result.initial_candidates} >= {
        "selected",
        "accepted_not_selected",
        "rejected_similarity",
    }
    assert next(item for item in result.initial_candidates if item.chunk_id == "duplicate").rejection_reason
    assert len({item.chunk_id for item in result.initial_candidates}) == 4

def test_enhanced_retrieval_accepts_injected_reranker() -> None:
    searcher = FakeSearcher(
        {"вопрос": [row("first", 0.9), row("second", 0.8)]}
    )
    reranker = FakeReranker()
    config = RetrievalConfig(
        candidate_top_k=2,
        final_top_k=2,
        similarity_threshold=0.0,
        use_query_rewrite=False,
        reranker_type="heuristic",
    )

    result = enhanced_retrieve(
        "вопрос",
        "structural",
        config,
        None,
        reranker=reranker,
        searcher=searcher,
    )

    assert reranker.queries == ["вопрос"]
    assert [item.chunk_id for item in result.final_chunks] == ["second", "first"]


def test_candidate_top_k_limits_fused_multi_query_set() -> None:
    llm = RewriteLLM(
        json.dumps(
            {
                "search_query": "переписанный",
                "alternative_queries": ["альтернатива"],
            }
        )
    )
    searcher = FakeSearcher(
        {
            "исходный": [row("original", 0.7)],
            "переписанный": [row("rewritten", 0.9)],
            "альтернатива": [row("alternative", 0.8)],
        }
    )
    config = RetrievalConfig(
        candidate_top_k=2,
        final_top_k=2,
        similarity_threshold=0.0,
        reranker_type="none",
    )

    result = enhanced_retrieve(
        "исходный",
        "structural",
        config,
        llm,
        searcher=searcher,
    )

    assert searcher.queries == ["исходный", "переписанный", "альтернатива"]
    assert result.candidate_count == 2


def test_enhanced_empty_after_threshold_never_invents_top_one() -> None:
    searcher = FakeSearcher({"вопрос": [row("low", 0.2)]})
    config = RetrievalConfig(
        candidate_top_k=5,
        final_top_k=3,
        similarity_threshold=0.5,
        use_query_rewrite=False,
    )

    result = enhanced_retrieve("вопрос", "structural", config, None, searcher=searcher)

    assert result.final_chunks == []
    assert result.final_count == 0
    assert result.rejected_candidates[0].chunk_id == "low"
