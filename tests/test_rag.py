from __future__ import annotations

import pytest

from src.llm import LLMConnectionError
from src.models import RetrievedChunk
from src.rag import answer_question, build_context, compare_answers
from src.text_utils import RegexTokenizer


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self, answer: str = "Тестовый ответ") -> None:
        self.answer = answer
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        self.calls.append((messages, temperature))
        return self.answer


class FailingLLM(FakeLLM):
    def generate(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        raise LLMConnectionError("Ollama недоступна")


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls = 0
        self.tokenizer = RegexTokenizer()

    def retrieve(self, query: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        self.calls += 1
        return self.chunks[:top_k]


def chunk(
    chunk_id: str,
    content: str,
    *,
    source: str = "guide.md",
    section: str | None = "Установка",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source=source,
        title="Guide",
        section=section,
        page_number=None,
        content=content,
        score=score,
        token_count=len(content.split()),
    )


def test_no_rag_never_calls_retrieval_or_includes_documents() -> None:
    llm = FakeLLM()
    retriever = FakeRetriever([chunk("a", "секретный локальный документ")])

    result = answer_question("Общий вопрос", "no_rag", llm, retriever=retriever)

    assert retriever.calls == 0
    assert result.retrieved_chunks == []
    assert result.sources == []
    assert "секретный" not in str(llm.calls[0][0])


def test_rag_calls_retrieval_once_and_passes_context() -> None:
    llm = FakeLLM("Ответ на основании источника")
    retriever = FakeRetriever([chunk("a", "Запуск выполняется командой streamlit run app.py")])

    result = answer_question("Как запустить?", "rag", llm, retriever=retriever)
    prompt = "\n".join(message["content"] for message in llm.calls[0][0])

    assert retriever.calls == 1
    assert "Как запустить?" in prompt
    assert "streamlit run app.py" in prompt
    assert result.sources[0]["number"] == 1
    assert "[1]" in result.answer
    assert result.retrieval_time >= 0
    assert result.generation_time >= 0


def test_context_deduplicates_chunks_and_numbers_sources_deterministically() -> None:
    chunks = [
        chunk("a", "первый текст"),
        chunk("a", "дубликат"),
        chunk("b", "второй текст", source="code.py", section="run"),
        chunk("c", "ещё один фрагмент"),
    ]

    context, used = build_context(chunks, 500, tokenizer=RegexTokenizer())

    assert [item.chunk_id for item in used] == ["a", "b", "c"]
    assert [item.source_number for item in used] == [1, 2, 1]
    assert context.count("Chunk ID: a") == 1
    assert "[Источник 1]" in context
    assert "[Источник 2]" in context


def test_context_budget_never_splits_chunk() -> None:
    tokenizer = RegexTokenizer()
    chunks = [chunk("a", "короткий текст"), chunk("b", " ".join(["длинный"] * 100))]
    context, used = build_context(chunks, 60, tokenizer=tokenizer)

    assert [item.chunk_id for item in used] == ["a"]
    assert len(tokenizer.token_spans(context)) <= 60
    assert "длинный длинный" not in context


def test_missing_context_is_reported_without_llm_call() -> None:
    llm = FakeLLM()
    result = answer_question("Неизвестный факт", "rag", llm, retriever=FakeRetriever([]))

    assert "не найден" in result.answer
    assert llm.calls == []


def test_invalid_question_and_top_k_are_rejected() -> None:
    with pytest.raises(ValueError, match="пустым"):
        answer_question(" ", "no_rag", FakeLLM())
    with pytest.raises(ValueError, match="top_k"):
        answer_question("вопрос", "no_rag", FakeLLM(), top_k=0)


def test_llm_connection_error_stays_user_friendly() -> None:
    with pytest.raises(LLMConnectionError, match="недоступна"):
        answer_question("вопрос", "no_rag", FailingLLM())


def test_comparison_uses_same_llm_question_and_temperature() -> None:
    llm = FakeLLM()
    retriever = FakeRetriever([chunk("a", "контекст")])

    no_rag, rag = compare_answers(
        "Один вопрос", llm, temperature=0.25, retriever=retriever
    )

    assert no_rag.question == rag.question == "Один вопрос"
    assert [call[1] for call in llm.calls] == [0.25, 0.25]
    assert no_rag.model_name == rag.model_name == "fake-llm"
