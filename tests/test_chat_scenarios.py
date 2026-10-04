from __future__ import annotations

import json
from pathlib import Path

from src.chat_evaluation import (
    aggregate_chat_metrics,
    load_chat_scenarios,
    run_chat_scenarios,
    write_chat_evaluation_report,
)
from src.chat_service import ChatService
from src.chat_storage import ChatStorage
from src.models import RetrievalConfig
from src.text_utils import RegexTokenizer


class ScenarioLLM:
    model_name = "deterministic-offline-fake"

    def generate(self, messages, temperature=0.1):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Кратко обнови резюме" in system:
            return "Сохранены цель, активные ограничения и принятые решения."
        if "структурированную память" in system:
            current = json.loads(
                user.split("<current_state>\n", 1)[1].split("\n</current_state>", 1)[0]
            )
            message = user.split("<new_user_message>\n", 1)[1].split(
                "\n</new_user_message>", 1
            )[0]
            folded = message.casefold()
            if "моя цель" in folded:
                current["goal"] = "Настроить локальный RAG для русскоязычных PDF"
                current["constraints"] = [
                    "Только CPU",
                    "Без внешних API",
                    "Использовать structural index",
                    "Final Top-K не больше 3",
                ]
            elif folded.startswith("цель — повысить"):
                current["goal"] = "Повысить Source Hit@3"
                current["constraints"] = [
                    "Не менять embedding-модель",
                    "Использовать heuristic reranker",
                    "Сохранить разумное время ответа",
                    "Сравнивать Baseline и Enhanced RAG",
                ]
            if "основной индекс" in folded:
                current["terms"]["основной индекс"] = "structural index"
            if "термин «качество»" in folded:
                current["terms"]["качество"] = "Source Hit@3 retrieval"
            if "не больше 4 вместо 3" in folded:
                current["constraints"] = [
                    value
                    for value in current["constraints"]
                    if "final top-k" not in value.casefold()
                ] + ["Final Top-K не больше 4"]
            if "внешних api» явно отменяю" in folded:
                current["constraints"] = [
                    value
                    for value in current["constraints"]
                    if "внешних api" not in value.casefold()
                ]
            if "уменьши final top-k" in folded:
                current["constraints"] = [
                    value
                    for value in current["constraints"]
                    if "final top-k" not in value.casefold()
                ] + ["Final Top-K 3"]
            if "не более 5 секунд" in folded:
                current["constraints"] = [
                    value
                    for value in current["constraints"]
                    if "разумное время ответа" not in value.casefold()
                ] + ["Retrieval не более 5 секунд"]
            if "все мои pdf содержат" in folded:
                current["confirmed_facts"] = ["Все PDF содержат текстовый слой"]
            if "предпочитаю краткие" in folded:
                current["user_preferences"] = ["Краткие ответы со списком команд"]
            if "решение: начнём" in folded:
                current["decisions"] = ["Сравнить threshold 0.30, 0.35 и 0.40"]
            current["changes"] = {"added": {}, "removed": {}, "changed": {}}
            return json.dumps(current, ensure_ascii=False)
        if "самостоятельный поисковый запрос" in system:
            message = user.split("<current_message>\n", 1)[1].split(
                "\n</current_message>", 1
            )[0]
            if "Как его построить" in message:
                return "Как построить structural index для загруженных PDF?"
            if "как он влияет" in message:
                return "Как similarity threshold влияет на пустые результаты?"
            if "Используй его" in message:
                return "Как Query Rewrite помогает найти, где сохраняется диагностика?"
            return message
        if "Преобразуй вопрос" in system:
            question = user.split("Исходный вопрос: ", 1)[1].split("\n", 1)[0]
            return json.dumps(
                {"search_query": question, "alternative_queries": []},
                ensure_ascii=False,
            )
        if "ассистент по локальной базе" in system:
            return (
                "PDF structural index CPU OCR команды; Query Rewrite и SQLite; "
                "threshold 0.35; Source Hit@3; Baseline и Enhanced; Final Top-K 3 и 4 [1]"
            )
        raise AssertionError(system)


class ScenarioSearcher:
    tokenizer = RegexTokenizer()

    def search(self, query: str, strategy: str, top_k: int):
        folded = query.casefold()
        no_source = any(
            marker in folded
            for marker in (
                "kubernetes",
                "подтверждённый факт",
                "проверь, что отменённое",
                "исправление:",
            )
        )
        if no_source:
            return [
                {
                    "chunk_id": "low",
                    "source": "irrelevant.md",
                    "title": None,
                    "section": None,
                    "page_number": None,
                    "content": "нерелевантно",
                    "score": 0.1,
                    "token_count": 1,
                }
            ]
        source = (
            "day23_reranking_filtering_omp.md"
            if any(
                marker in folded
                for marker in ("threshold", "hit@3", "rerank", "query rewrite", "top-k", "диагност")
            )
            else "README.md"
        )
        return [
            {
                "chunk_id": f"chunk-{source}",
                "source": source,
                "title": "Документация",
                "section": "RAG",
                "page_number": None,
                "content": "PDF structural index CPU Query Rewrite threshold 0.35 Hit@3 SQLite OCR.",
                "score": 0.9,
                "token_count": 12,
            }
        ][:top_k]


def test_two_scenarios_have_required_length_and_execute_offline(tmp_path: Path) -> None:
    scenarios = load_chat_scenarios()
    assert [len(item["turns"]) for item in scenarios] == [12, 12]

    storage = ChatStorage(tmp_path / "chat.sqlite")
    searcher = ScenarioSearcher()
    service = ChatService(storage, candidate_searcher=searcher, tokenizer=searcher.tokenizer)
    config = RetrievalConfig(
        candidate_top_k=5,
        final_top_k=3,
        similarity_threshold=0.35,
    )
    results = run_chat_scenarios(
        service,
        ScenarioLLM(),
        config,
        model_name="deterministic-offline-fake",
    )
    report = write_chat_evaluation_report(
        results,
        model_name="deterministic-offline-fake",
        retrieval_config=config,
        path=tmp_path / "chat_memory_evaluation.md",
        execution_mode="фиктивная локальная интеграция без Ollama и сети",
    )
    metrics = aggregate_chat_metrics(results)

    assert metrics["turn_count"] == 24
    assert metrics["goal_preserved_share"] == 1.0
    assert metrics["cancelled_constraint_returns"] == 0
    assert metrics["fabricated_sources_count"] == 0
    assert len(storage.list_conversations()) == 2
    assert all(len(item["turns"]) == 12 for item in results)
    assert all(
        storage.get_summary(item["conversation_id"]) is not None
        for item in results
    )
    assert "scenario" not in report.name
    assert "Итоговые метрики" in report.read_text(encoding="utf-8")
