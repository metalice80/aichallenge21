"""Streamlit-интерфейс локального индексатора документов."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    CHAT_DATABASE,
    DOCUMENTS_DIR,
    INDEX_ROOT,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    RAG_QUESTIONS_PATH,
    RAG_REPORT_PATH,
    RAG_RESULTS_PATH,
    RERANKING_REPORT_PATH,
    RESULTS_DATABASE,
    STRATEGIES,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    configure_logging,
    ensure_directories,
)
from src.chat_storage import ChatStorage
from src.chat_ui import render_rag_chat
from src.embeddings import EmbeddingService
from src.data_management import delete_indexes
from src.evaluation import (
    evaluate_strategies,
    index_statistics,
    load_evaluation_queries,
    write_comparison_report,
)
from src.loaders import DocumentLoadError, load_document
from src.llm import LLMError, OllamaProvider
from src.models import AnswerResult, RetrievalConfig
from src.pipeline import build_indexes
from src.rag import (
    SemanticRetriever,
    answer_question,
    compare_retrieval_modes,
)
from src.rag_evaluation import load_questions
from src.retrieval import SemanticCandidateSearcher
from src.retrieval_evaluation import (
    aggregate_reranking_metrics,
    run_reranking_benchmark,
    run_threshold_experiment,
    write_reranking_report,
)
from src.result_storage import (
    ResultStorage,
    clear_json_results,
    clear_sqlite_results,
)
from src.search import semantic_search
from src.storage import IndexConsistencyError, IndexNotFoundError, IndexStorage
from src.text_utils import count_words, estimate_pages

configure_logging()
ensure_directories()
logger = logging.getLogger(__name__)
result_storage = ResultStorage(RESULTS_DATABASE)
chat_storage = ChatStorage(CHAT_DATABASE)

st.set_page_config(page_title="Локальный индексатор документов", layout="wide")
st.title("Локальный индексатор документов")
st.caption("Семантический поиск и сравнение fixed/structural chunking — полностью локально")


@st.cache_resource(show_spinner="Загрузка локальной модели эмбеддингов…")
def get_embedding_service(batch_size: int = 32) -> EmbeddingService:
    """Создать один сервис эмбеддингов на процесс Streamlit."""
    return EmbeddingService(batch_size=batch_size)


def available_files() -> list[Path]:
    """Вернуть пользовательские и демонстрационные документы без дублей путей."""
    files: list[Path] = []
    for directory in (UPLOAD_DIR, DOCUMENTS_DIR):
        files.extend(
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return files


def uploaded_files() -> list[Path]:
    """Вернуть только загруженные пользователем документы."""
    return [
        path
        for path in sorted(UPLOAD_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def save_uploads(files: list[Any]) -> tuple[int, list[str]]:
    """Сохранить новые файлы и отфильтровать дубликаты по SHA-256."""
    known_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in uploaded_files()
    }
    saved = 0
    messages: list[str] = []
    for uploaded in files:
        safe_name = Path(uploaded.name).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            messages.append(f"{safe_name}: формат не поддерживается")
            continue
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if digest in known_hashes:
            messages.append(f"{safe_name}: дубликат пропущен")
            continue
        target = UPLOAD_DIR / safe_name
        if target.exists():
            target = UPLOAD_DIR / f"{target.stem}-{digest[:8]}{target.suffix}"
        target.write_bytes(data)
        known_hashes.add(digest)
        saved += 1
    return saved, messages


def render_result(result: dict[str, Any]) -> None:
    """Показать одну карточку результата поиска."""
    page = f" · стр. {result['page_number']}" if result.get("page_number") else ""
    section = result.get("section") or "Без секции"
    st.markdown(
        f"**#{result['rank']} · {result['source']}** — similarity `{result['score']:.4f}`  \n"
        f"Секция: `{section}`{page} · {result['token_count']} токенов"
    )
    st.code(result["content"], language=None, wrap_lines=True)


def clear_current_rag_result() -> None:
    """Очистить только текущие несохранённые значения интерфейса."""
    st.session_state["rag-question"] = ""
    st.session_state.pop("rag-results", None)
    st.session_state.pop("rag-run-id", None)


def render_mode_answer(result: AnswerResult, label: str) -> None:
    """Показать ответ и фактически переданный LLM контекст одного режима."""
    st.markdown(f"### {label}")
    st.markdown(result.answer)
    st.caption(
        f"Режим: {result.retrieval_mode} · модель: {result.model_name} · "
        f"retrieval: {result.retrieval_time:.3f} с · "
        f"генерация: {result.generation_time:.3f} с · всего: {result.total_time:.3f} с"
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Чанков в контексте", len(result.retrieved_chunks))
    metric_b.metric("Источников", len(result.sources))
    metric_c.metric("Контекст, токенов", result.context_token_count)
    if result.sources:
        st.markdown(
            "**Источники:** "
            + "; ".join(
                f"[{source['number']}] {source['source']}"
                + (f" — {source['section']}" if source.get("section") else "")
                for source in result.sources
            )
        )
    else:
        st.warning("Релевантные локальные источники в LLM-контекст не передавались.")


def render_retrieval_diagnostics(result: AnswerResult) -> None:
    """Показать Query Rewrite, воронку и статусы всех Enhanced-кандидатов."""
    diagnostics = result.retrieval_diagnostics
    st.markdown("### Query Rewrite")
    query_a, query_b = st.columns(2)
    query_a.markdown(f"**Исходный вопрос**\n\n{result.original_query}")
    query_b.markdown(
        f"**Переписанный запрос**\n\n{result.rewritten_query or 'Не создан'}"
    )
    if result.alternative_queries:
        st.markdown(
            "**Альтернативные запросы:**\n"
            + "\n".join(
                f"{position}. {query}"
                for position, query in enumerate(result.alternative_queries, start=1)
            )
        )
    if diagnostics.get("fallback_reason"):
        st.warning(f"Использован fallback: {diagnostics['fallback_reason']}")
    else:
        st.success(
            "Query Rewrite использован."
            if diagnostics.get("rewrite_used")
            else "Query Rewrite выключен или не изменил запрос."
        )
    st.caption(f"Время rewrite: {result.rewrite_time:.4f} с")

    st.markdown("### Retrieval funnel")
    funnel_a, funnel_b, funnel_c, funnel_d = st.columns(4)
    funnel_a.metric("FAISS-кандидаты", diagnostics.get("candidate_count", 0))
    funnel_b.metric("После similarity", diagnostics.get("accepted_count", 0))
    funnel_c.metric("После reranking", diagnostics.get("reranked_count", 0))
    funnel_d.metric("Final Top-K", diagnostics.get("final_count", 0))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Этап": "Query Rewrite",
                    "Время, с": round(float(diagnostics.get("rewrite_time", 0.0)), 4),
                },
                {
                    "Этап": "FAISS-поиск",
                    "Время, с": round(float(diagnostics.get("search_time", 0.0)), 4),
                },
                {
                    "Этап": "Similarity-фильтр",
                    "Время, с": round(float(diagnostics.get("filter_time", 0.0)), 4),
                },
                {
                    "Этап": "Reranking",
                    "Время, с": round(float(diagnostics.get("rerank_time", 0.0)), 4),
                },
                {
                    "Этап": "Полный retrieval",
                    "Время, с": round(result.retrieval_time, 4),
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    candidates = diagnostics.get("initial_candidates", [])
    st.markdown("### Кандидаты до и после фильтрации")
    if not candidates:
        st.warning("FAISS не вернул кандидатов.")
        return
    statuses = sorted({str(item.get("status", "candidate")) for item in candidates})
    selected_statuses = st.multiselect(
        "Фильтр статусов",
        statuses,
        default=statuses,
        key="rag-candidate-statuses",
    )
    visible = [item for item in candidates if item.get("status") in selected_statuses]
    rows = [
        {
            "Rank до": item.get("original_rank"),
            "Rank после": item.get("final_rank"),
            "Источник": item.get("source"),
            "Section": item.get("section") or "—",
            "Similarity": item.get("similarity"),
            "Fusion": item.get("fusion_score"),
            "Semantic": item.get("semantic_score"),
            "Lexical": item.get("lexical_score"),
            "Metadata": item.get("metadata_score"),
            "Rerank": item.get("rerank_score"),
            "Final": item.get("final_score"),
            "Статус": item.get("status"),
            "Причина": item.get("rejection_reason") or "—",
            "Chunk ID": item.get("chunk_id"),
        }
        for item in visible
    ]

    def color_candidate_status(row: pd.Series) -> list[str]:
        status = str(row["Статус"])
        color = (
            "#d8f3dc"
            if status == "selected"
            else "#fff3bf"
            if status == "accepted_not_selected"
            else "#e9ecef"
            if status == "duplicate"
            else "#ffe3e3"
        )
        return [f"background-color: {color}"] * len(row)

    if rows:
        st.dataframe(
            pd.DataFrame(rows).style.apply(color_candidate_status, axis=1),
            width="stretch",
            hide_index=True,
        )
    for item in visible:
        with st.expander(
            f"{item.get('status')} · {item.get('source')} · "
            f"{item.get('chunk_id')}"
        ):
            st.write(item.get("rejection_reason") or "Кандидат не отклонён.")
            st.caption(
                "Найден по запросам: "
                + ", ".join(item.get("matched_queries", []))
            )
            st.code(item.get("content", ""), language=None, wrap_lines=True)


with st.sidebar:
    st.header("Состояние")
    st.caption(f"Модель: `{MODEL_NAME}`")
    st.caption("Хранение: FAISS CPU + SQLite")
    for strategy in STRATEGIES:
        storage = IndexStorage(strategy)
        if storage.exists:
            try:
                manifest = storage.load_manifest()
                st.success(f"{strategy}: {manifest['chunk_count']} чанков")
                st.caption(f"Создан: {manifest.get('created_at', '—')}")
            except Exception as exc:
                st.error(f"{strategy}: повреждён ({exc})")
        else:
            st.warning(f"{strategy}: индекс отсутствует")
    st.info("Документы и запросы не отправляются во внешние сервисы.")

(tab_documents, tab_indexing, tab_chunks, tab_search, tab_compare, tab_analytics,
 tab_rag, tab_chat, tab_rag_evaluation, tab_settings) = st.tabs(
    [
        "Документы",
        "Индексация",
        "Чанки",
        "Поиск",
        "Сравнение",
        "Аналитика",
        "Сравнение RAG",
        "RAG-чат",
        "Оценка RAG",
        "Настройки",
    ]
)

with tab_documents:
    st.subheader("Загрузка документов")
    uploads = st.file_uploader(
        "Выберите PDF, Markdown, TXT или Python-файлы",
        type=["pdf", "md", "txt", "py"],
        accept_multiple_files=True,
    )
    if st.button("Сохранить выбранные файлы", disabled=not uploads, type="primary"):
        try:
            saved_count, upload_messages = save_uploads(uploads or [])
            if saved_count:
                st.success(f"Сохранено файлов: {saved_count}")
            for message in upload_messages:
                st.warning(message)
        except OSError as exc:
            logger.exception("Ошибка сохранения загрузки")
            st.error(f"Не удалось сохранить файл: {exc}")

    files = uploaded_files()
    if not files:
        st.info("Загруженных пользователем документов пока нет.")
    for path in files:
        with st.expander(f"{path.name} · {path.suffix[1:].upper()} · {path.stat().st_size / 1024:.1f} КБ"):
            try:
                documents = load_document(path)
                combined = "\n\n".join(document.text for document in documents)
                pages = len(documents) if path.suffix.lower() == ".pdf" else estimate_pages(combined)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Символов", f"{len(combined):,}")
                col_b.metric("Слов", f"{count_words(combined):,}")
                col_c.metric("Страниц", pages)
                st.text_area("Предпросмотр", combined[:5000], height=240, disabled=True, key=f"preview-{path.name}")
            except DocumentLoadError as exc:
                st.error(str(exc))
            confirmed = st.checkbox("Подтверждаю удаление", key=f"confirm-{path.name}")
            if st.button("Удалить файл", key=f"delete-{path.name}", disabled=not confirmed):
                try:
                    path.unlink()
                    st.success(f"Файл {path.name} удалён")
                    st.rerun()
                except OSError as exc:
                    logger.exception("Ошибка удаления %s", path)
                    st.error(f"Не удалось удалить файл: {exc}")

with tab_indexing:
    st.subheader("Построение индексов")
    all_files = available_files()
    st.caption(f"Документов для индексации: {len(all_files)} (загруженные и демонстрационные)")
    strategy_choice = st.radio("Стратегия", ["Обе", "fixed", "structural"], horizontal=True)
    settings_a, settings_b, settings_c, settings_d = st.columns(4)
    fixed_size = settings_a.number_input("Fixed chunk", min_value=50, max_value=2000, value=500, step=50)
    overlap = settings_b.number_input("Overlap", min_value=0, max_value=500, value=80, step=10)
    structural_size = settings_c.number_input(
        "Structural max", min_value=50, max_value=2500, value=700, step=50
    )
    batch_size = settings_d.number_input("Batch size", min_value=1, max_value=256, value=32)
    if overlap >= min(fixed_size, structural_size):
        st.error("Overlap должен быть меньше обоих размеров чанка.")
    can_index = bool(all_files) and overlap < min(fixed_size, structural_size)
    if st.button("Начать индексацию", type="primary", disabled=not can_index):
        selected = STRATEGIES if strategy_choice == "Обе" else (strategy_choice,)
        progress = st.progress(0.0, text="Подготовка")
        stage_box = st.empty()

        def update_progress(stage: str, fraction: float, message: str) -> None:
            progress.progress(float(max(0.0, min(1.0, fraction))), text=f"{stage}: {message}")
            stage_box.caption("Загрузка → Извлечение текста → Chunking → Embeddings → Сохранение → Готово")

        try:
            service = get_embedding_service(int(batch_size))
            results = build_indexes(
                all_files,
                selected,
                fixed_size=int(fixed_size),
                overlap=int(overlap),
                structural_size=int(structural_size),
                batch_size=int(batch_size),
                embedding_service=service,
                progress_callback=update_progress,
            )
            st.success("Индексация завершена")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Стратегия": item.strategy,
                            "Документы": item.document_count,
                            "Чанки": item.chunk_count,
                            "Средний размер": round(item.average_chunk_size, 1),
                            "Время, с": round(item.elapsed_seconds, 2),
                            "Путь": item.index_path,
                        }
                        for item in results.values()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            logger.exception("Ошибка индексации")
            st.error(f"Индексация не выполнена: {exc}")

with tab_chunks:
    st.subheader("Просмотр чанков")
    chunk_strategy = st.selectbox("Стратегия", STRATEGIES, key="chunk-strategy")
    chunk_storage = IndexStorage(chunk_strategy)
    if not chunk_storage.exists:
        st.info(f"Сначала постройте индекс {chunk_strategy}.")
    else:
        try:
            all_chunks = chunk_storage.list_chunks()
            source_options = ["Все"] + sorted({str(row["source"]) for row in all_chunks})
            section_options = ["Все"] + sorted(
                {str(row["section"]) for row in all_chunks if row["section"]}
            )
            filters_a, filters_b = st.columns(2)
            selected_source = filters_a.selectbox("Источник", source_options, key="chunk-source")
            selected_section = filters_b.selectbox("Секция", section_options, key="chunk-section")
            max_available = max(int(row["token_count"]) for row in all_chunks)
            token_range = st.slider("Количество токенов", 0, max_available, (0, max_available))
            rows = chunk_storage.list_chunks(
                source=None if selected_source == "Все" else selected_source,
                section=None if selected_section == "Все" else selected_section,
                min_tokens=token_range[0],
                max_tokens=token_range[1],
            )
            table_columns = ["vector_id", "source", "section", "page_number", "token_count", "chunk_id"]
            st.dataframe(pd.DataFrame(rows)[table_columns], width="stretch", hide_index=True)
            if rows:
                labels = {
                    f"#{row['vector_id']} · {row['source']} · {row['section'] or 'Без секции'}": row
                    for row in rows
                }
                selected_label = st.selectbox("Открыть чанк", list(labels))
                selected_chunk = labels[selected_label]
                st.code(selected_chunk["content"], language=None, wrap_lines=True)
                st.json({key: value for key, value in selected_chunk.items() if key != "content"})
                same_document = [
                    row for row in all_chunks if row["document_id"] == selected_chunk["document_id"]
                ]
                position = next(
                    index
                    for index, row in enumerate(same_document)
                    if row["vector_id"] == selected_chunk["vector_id"]
                )
                neighbors = st.columns(2)
                if position > 0:
                    previous = same_document[position - 1]
                    neighbors[0].caption(f"Предыдущий: #{previous['vector_id']} · {previous['section'] or '—'}")
                if position + 1 < len(same_document):
                    following = same_document[position + 1]
                    neighbors[1].caption(f"Следующий: #{following['vector_id']} · {following['section'] or '—'}")
        except Exception as exc:
            logger.exception("Ошибка просмотра чанков")
            st.error(f"Не удалось прочитать чанки: {exc}")

with tab_search:
    st.subheader("Семантический поиск")
    search_query = st.text_input("Запрос", key="search-query")
    search_a, search_b, search_c = st.columns(3)
    search_strategy = search_a.selectbox("Индекс", STRATEGIES, key="search-strategy")
    top_k = search_b.slider("Количество результатов", 1, 20, 5)
    search_sources = IndexStorage(search_strategy).sources()
    search_source = search_c.selectbox("Источник", ["Все"] + search_sources, key="search-source")
    if st.button("Найти", type="primary", disabled=not search_query.strip()):
        try:
            found = semantic_search(
                search_query,
                search_strategy,
                top_k,
                None if search_source == "Все" else search_source,
                embedding_service=get_embedding_service(),
            )
            if not found:
                st.warning("Подходящие фрагменты не найдены.")
            for result in found:
                render_result(result)
        except (IndexNotFoundError, IndexConsistencyError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Ошибка поиска")
            st.error(f"Поиск не выполнен: {exc}")

with tab_compare:
    st.subheader("Сравнение стратегий")
    compare_query = st.text_input("Один запрос для двух индексов", key="compare-query")
    compare_k = st.slider("Результатов в каждой колонке", 1, 10, 3)
    if st.button("Сравнить", type="primary", disabled=not compare_query.strip()):
        try:
            service = get_embedding_service()
            fixed_results = semantic_search(
                compare_query, "fixed", compare_k, embedding_service=service
            )
            structural_results = semantic_search(
                compare_query, "structural", compare_k, embedding_service=service
            )
            if fixed_results and structural_results:
                fixed_first, structural_first = fixed_results[0], structural_results[0]
                metric_a, metric_b, metric_c, metric_d = st.columns(4)
                metric_a.metric(
                    "Одинаковый источник",
                    "Да" if fixed_first["source"] == structural_first["source"] else "Нет",
                )
                metric_b.metric(
                    "Разница score",
                    f"{fixed_first['score'] - structural_first['score']:+.4f}",
                )
                metric_c.metric(
                    "Размер fixed / structural",
                    f"{fixed_first['token_count']} / {structural_first['token_count']}",
                )
                metric_d.metric(
                    "Секции",
                    f"{fixed_first['section'] or '—'} / {structural_first['section'] or '—'}",
                )
            left, right = st.columns(2)
            with left:
                st.markdown("### Fixed")
                for result in fixed_results:
                    render_result(result)
            with right:
                st.markdown("### Structural")
                for result in structural_results:
                    render_result(result)
        except (IndexNotFoundError, IndexConsistencyError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Ошибка сравнения")
            st.error(f"Сравнение не выполнено: {exc}")

with tab_analytics:
    st.subheader("Аналитика индексов")
    existing = [strategy for strategy in STRATEGIES if IndexStorage(strategy).exists]
    if not existing:
        st.info("Для аналитики необходимо построить хотя бы один индекс.")
    else:
        stats = [index_statistics(strategy) for strategy in existing]
        st.dataframe(pd.DataFrame(stats), width="stretch", hide_index=True)
        all_rows: list[dict[str, Any]] = []
        for strategy in existing:
            all_rows.extend(IndexStorage(strategy).list_chunks())
        frame = pd.DataFrame(all_rows)
        chart_a, chart_b = st.columns(2)
        counts = frame.groupby("strategy", as_index=False).size().rename(columns={"size": "chunks"})
        chart_a.plotly_chart(
            px.bar(counts, x="strategy", y="chunks", title="Количество чанков"),
            width="stretch",
        )
        chart_b.plotly_chart(
            px.histogram(
                frame,
                x="token_count",
                color="strategy",
                barmode="overlay",
                title="Распределение размеров чанков",
            ),
            width="stretch",
        )
        by_document = frame.groupby(["strategy", "source"], as_index=False).size()
        st.plotly_chart(
            px.bar(
                by_document,
                x="source",
                y="size",
                color="strategy",
                barmode="group",
                title="Чанки по документам",
                labels={"size": "Чанки"},
            ),
            width="stretch",
        )
        query_path = DOCUMENTS_DIR / "evaluation_queries.json"
        if len(existing) == 2 and query_path.is_file():
            if st.button("Запустить контрольную оценку"):
                try:
                    evaluation = evaluate_strategies(
                        load_evaluation_queries(query_path),
                        embedding_service=get_embedding_service(),
                    )
                    report_path = write_comparison_report(evaluation)
                    st.session_state["evaluation"] = evaluation
                    st.success(f"Отчёт сохранён: {report_path.relative_to(Path.cwd())}")
                except Exception as exc:
                    logger.exception("Ошибка контрольной оценки")
                    st.error(f"Оценка не выполнена: {exc}")
            if "evaluation" in st.session_state:
                evaluation_frame = pd.DataFrame(st.session_state["evaluation"]).T.reset_index()
                st.dataframe(
                    evaluation_frame[
                        ["index", "hit_at_1", "hit_at_3", "mean_search_seconds"]
                    ],
                    width="stretch",
                    hide_index=True,
                )


with tab_rag:
    st.subheader("Сравнение Baseline RAG и Enhanced RAG")
    st.caption(
        "Одинаковые документы, embedding-модель, LLM, temperature и бюджет контекста. "
        "Различаются только retrieval-настройки."
    )
    rag_question = st.text_area(
        "Вопрос",
        height=100,
        key="rag-question",
        placeholder="Например: Какая функция загружает PDF-файлы?",
    )
    rag_model = st.text_input("Модель Ollama", value=OLLAMA_MODEL, key="rag-model")
    st.caption(f"Локальный сервер: `{OLLAMA_BASE_URL}`")
    rag_a, rag_b = st.columns(2)
    rag_strategy = rag_a.selectbox("Индекс RAG", STRATEGIES, index=1, key="rag-strategy")
    baseline_top_k = rag_b.slider("Baseline Top-K", 1, 20, 5, key="baseline-top-k")

    with st.expander("Расширенные настройки Enhanced RAG", expanded=True):
        setting_a, setting_b, setting_c = st.columns(3)
        use_rewrite = setting_a.checkbox("Query Rewrite", value=True, key="use-query-rewrite")
        max_alternatives = setting_b.slider(
            "Альтернативных запросов", 0, 3, 2, key="max-alternative-queries"
        )
        candidate_top_k = setting_c.slider(
            "Candidate Top-K", 1, 20, 20, key="candidate-top-k"
        )
        setting_d, setting_e, setting_f = st.columns(3)
        similarity_threshold = setting_d.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            key="similarity-threshold",
        )
        reranker_label = setting_e.selectbox(
            "Reranker",
            ["Heuristic", "NoOp"],
            key="reranker-type",
        )
        if st.session_state.get("final-top-k", 1) > int(candidate_top_k):
            st.session_state["final-top-k"] = int(candidate_top_k)
        final_top_k = setting_f.slider(
            "Final Top-K",
            1,
            int(candidate_top_k),
            min(5, int(candidate_top_k)),
            key="final-top-k",
        )
        setting_g, setting_h, setting_i = st.columns(3)
        max_per_source = setting_g.number_input(
            "Максимум чанков источника",
            min_value=1,
            max_value=20,
            value=3,
            key="max-chunks-per-source",
        )
        rag_max_tokens = setting_h.number_input(
            "Контекст, токенов",
            min_value=100,
            max_value=16_000,
            value=4000,
            step=100,
            key="rag-max-context",
        )
        rag_temperature = setting_i.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.1,
            key="rag-temperature",
        )
        use_rerank_threshold = st.checkbox(
            "Применить отдельный rerank threshold",
            value=False,
            key="use-rerank-threshold",
        )
        rerank_threshold = (
            st.slider(
                "Rerank threshold",
                0.0,
                1.0,
                0.35,
                0.01,
                key="rerank-threshold",
            )
            if use_rerank_threshold
            else None
        )
        include_no_rag = st.checkbox(
            "Добавить режим без RAG",
            value=False,
            key="include-no-rag",
        )
        include_filtered_rag = st.checkbox(
            "Добавить Filtered RAG",
            value=False,
            key="include-filtered-rag",
        )

    connection_column, compare_column, clear_column = st.columns(3)
    if connection_column.button("Проверить Ollama", key="check-ollama"):
        try:
            connected, message = OllamaProvider(
                rag_model, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
            ).check_connection()
            (st.success if connected else st.error)(message)
        except ValueError as exc:
            st.error(str(exc))
    compare_pressed = compare_column.button(
        "Сравнить режимы",
        type="primary",
        disabled=not rag_question.strip(),
        key="compare-rag-modes",
    )
    clear_column.button(
        "Очистить текущий результат",
        on_click=clear_current_rag_result,
        key="clear-current-rag",
    )

    if compare_pressed:
        status = st.status("Выполнение RAG-сравнения", expanded=True)
        stage_labels = {
            "no_rag_generation": "✓ Ответ без RAG",
            "query_rewrite": "✓ Query Rewrite",
            "query_embedding": "✓ Генерация embedding исходного вопроса",
            "retrieval": "✓ Поиск, фильтрация и reranking",
            "context": "✓ Формирование контекста",
            "rag_generation": "✓ Генерация ответа",
        }
        try:
            provider = OllamaProvider(rag_model, OLLAMA_BASE_URL, OLLAMA_TIMEOUT)
            connected, message = provider.check_connection()
            if not connected:
                raise LLMError(message)
            status.write("✓ Проверка подключения к LLM")
            embedding_service = get_embedding_service()
            retriever = SemanticRetriever(embedding_service)
            candidate_searcher = SemanticCandidateSearcher(embedding_service)
            retrieval_config = RetrievalConfig(
                candidate_top_k=int(candidate_top_k),
                final_top_k=int(final_top_k),
                similarity_threshold=float(similarity_threshold),
                rerank_threshold=(
                    float(rerank_threshold) if rerank_threshold is not None else None
                ),
                use_query_rewrite=bool(use_rewrite),
                reranker_type=("heuristic" if reranker_label == "Heuristic" else "none"),
                max_alternative_queries=int(max_alternatives),
                max_chunks_per_source=int(max_per_source),
            )
            baseline_result, enhanced_result = compare_retrieval_modes(
                rag_question,
                provider,
                strategy=rag_strategy,
                baseline_top_k=int(baseline_top_k),
                retrieval_config=retrieval_config,
                max_context_tokens=int(rag_max_tokens),
                temperature=float(rag_temperature),
                retriever=retriever,
                candidate_searcher=candidate_searcher,
                tokenizer=embedding_service.tokenizer,
                progress_callback=lambda stage: status.write(stage_labels[stage]),
            )
            results = [baseline_result, enhanced_result]
            if include_filtered_rag:
                results.insert(
                    1,
                    answer_question(
                        rag_question,
                        "filtered_rag",
                        provider,
                        strategy=rag_strategy,
                        max_context_tokens=int(rag_max_tokens),
                        temperature=float(rag_temperature),
                        retrieval_config=retrieval_config,
                        candidate_searcher=candidate_searcher,
                        tokenizer=embedding_service.tokenizer,
                        progress_callback=lambda stage: status.write(
                            stage_labels[stage]
                        ),
                    ),
                )
            if include_no_rag:
                results.insert(
                    0,
                    answer_question(
                        rag_question,
                        "no_rag",
                        provider,
                        max_context_tokens=int(rag_max_tokens),
                        temperature=float(rag_temperature),
                        progress_callback=lambda stage: status.write(stage_labels[stage]),
                    ),
                )
            run_id = result_storage.save_mode_comparison(
                results,
                strategy=rag_strategy,
                baseline_top_k=int(baseline_top_k),
                max_context_tokens=int(rag_max_tokens),
                temperature=float(rag_temperature),
                retrieval_config=retrieval_config,
            )
            st.session_state["rag-results"] = results
            st.session_state["rag-run-id"] = run_id
            status.update(label="Сравнение завершено и сохранено", state="complete")
        except Exception as exc:
            logger.exception("Ошибка RAG-сравнения без записи контекста в лог")
            status.update(label="RAG-сравнение остановлено", state="error")
            st.error(str(exc))

    rag_results = st.session_state.get("rag-results")
    if rag_results:
        labels = {
            "no_rag": "Без RAG",
            "baseline_rag": "Baseline RAG",
            "filtered_rag": "Filtered RAG",
            "enhanced_rag": "Enhanced RAG",
        }
        result_tabs = st.tabs([labels.get(result.mode, result.mode) for result in rag_results])
        for result_tab, result in zip(result_tabs, rag_results, strict=True):
            with result_tab:
                render_mode_answer(result, labels.get(result.mode, result.mode))
        enhanced_result = next(
            (result for result in rag_results if result.mode == "enhanced_rag"),
            None,
        )
        if enhanced_result:
            render_retrieval_diagnostics(enhanced_result)

        current_run_id = st.session_state.get("rag-run-id")
        if current_run_id:
            st.markdown("### Ручная оценка")
            preference_labels = {
                "Baseline RAG": "baseline_rag",
                "Enhanced RAG": "enhanced_rag",
                "Filtered RAG": "filtered_rag",
                "Без RAG": "no_rag",
                "Одинаково": "equal",
                "Все неверны": "all_incorrect",
            }
            available_modes = {result.mode for result in rag_results}
            preference_labels = {
                label: value
                for label, value in preference_labels.items()
                if value in available_modes or value in {"equal", "all_incorrect"}
            }
            with st.form(f"manual-evaluation-{current_run_id}"):
                evaluation_a, evaluation_b, evaluation_c = st.columns(3)
                preferred_label = evaluation_a.selectbox(
                    "Какой режим лучше", list(preference_labels)
                )
                score_baseline = evaluation_b.select_slider(
                    "Оценка Baseline RAG", options=[0, 1, 2], value=1
                )
                score_enhanced = evaluation_c.select_slider(
                    "Оценка Enhanced RAG", options=[0, 1, 2], value=1
                )
                correct_sources = st.checkbox("Использованы правильные источники")
                irrelevant_sources = st.checkbox("Есть нерелевантные источники")
                unsupported = st.checkbox("Есть неподтверждённые утверждения")
                rewrite_improved = st.checkbox("Query Rewrite улучшил поиск")
                evaluation_comment = st.text_area("Комментарий")
                save_evaluation = st.form_submit_button("Сохранить оценку")
            if save_evaluation:
                try:
                    result_storage.save_manual_evaluation(
                        int(current_run_id),
                        preferred_mode=preference_labels[preferred_label],
                        score_baseline_rag=int(score_baseline),
                        score_enhanced_rag=int(score_enhanced),
                        correct_sources=correct_sources,
                        has_irrelevant_sources=irrelevant_sources,
                        has_unsupported_claims=unsupported,
                        rewrite_improved=rewrite_improved,
                        comment=evaluation_comment,
                    )
                    st.success("Ручная оценка сохранена без создания дубликата.")
                except Exception as exc:
                    logger.exception("Ошибка сохранения ручной оценки")
                    st.error(f"Оценка не сохранена: {exc}")

    st.markdown("### История сравнений")
    try:
        history = result_storage.list_history(limit=100)
        if not history:
            st.info("Сохранённых сравнений пока нет.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": row["id"],
                            "Дата": row["created_at"],
                            "Вопрос": row["question"][:100],
                            "Модель": row["model_name"],
                            "Режимы": row["retrieval_mode"] or "Day 22",
                            "Rewritten query": row["rewritten_query"] or "—",
                            "Fallback": row["rewrite_fallback_reason"] or "—",
                            "Baseline": row["answer_baseline_rag"][:120],
                            "Enhanced": row["answer_enhanced_rag"][:120],
                            "Оценка": row["preferred_mode"] or "—",
                        }
                        for row in history
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            history_ids = [int(row["id"]) for row in history]
            selected_history_id = st.selectbox(
                "ID сравнения для просмотра", history_ids, key="rag-history-id"
            )
            if st.button("Показать подробности", key="show-rag-history"):
                details = result_storage.get_run(int(selected_history_id))
                if details:
                    st.markdown("#### Восстановленная retrieval-диагностика")
                    st.write(f"Исходный запрос: {details['original_query']}")
                    st.write(
                        "Rewritten query: "
                        f"{details['rewritten_query'] or 'не использован'}"
                    )
                    if details["rewrite_fallback_reason"]:
                        st.warning(
                            f"Query Rewrite fallback: "
                            f"{details['rewrite_fallback_reason']}"
                        )
                    restored_funnel = pd.DataFrame(
                        [
                            {
                                "Этап": "Candidates",
                                "Количество": details["initial_candidate_count"],
                            },
                            {
                                "Этап": "Accepted",
                                "Количество": details["accepted_candidate_count"],
                            },
                            {
                                "Этап": "Final",
                                "Количество": details["final_candidate_count"],
                            },
                        ]
                    )
                    st.plotly_chart(
                        px.bar(
                            restored_funnel,
                            x="Этап",
                            y="Количество",
                            title="Восстановленный retrieval funnel",
                        ),
                        width="stretch",
                    )
                    if details["retrieval_candidates"]:
                        st.dataframe(
                            pd.DataFrame(details["retrieval_candidates"]),
                            width="stretch",
                            hide_index=True,
                        )
                    st.json(details["answers"])
    except Exception as exc:
        logger.exception("Ошибка чтения истории RAG")
        st.error(f"История недоступна: {exc}")


with tab_chat:
    render_rag_chat(chat_storage, get_embedding_service)


with tab_rag_evaluation:
    st.subheader("Оценка Baseline и Enhanced RAG")
    try:
        if "reranking-benchmark-results" not in st.session_state:
            restored_benchmark = result_storage.latest_reranking_results()
            if restored_benchmark:
                stored_config = restored_benchmark[0]["result"].get(
                    "parameters", {}
                ).get("retrieval_config")
                if stored_config:
                    st.session_state["reranking-benchmark-config"] = (
                        RetrievalConfig(**stored_config)
                    )
                st.session_state["reranking-benchmark-results"] = [
                    row["result"] for row in restored_benchmark
                ]
                st.session_state["rag-benchmark-batch"] = restored_benchmark[0][
                    "batch_id"
                ]
        if "threshold-results" not in st.session_state:
            restored_thresholds = result_storage.latest_threshold_results()
            if restored_thresholds:
                st.session_state["threshold-results"] = restored_thresholds
        benchmark_questions = load_questions(RAG_QUESTIONS_PATH)
        st.caption(f"Контрольных вопросов: {len(benchmark_questions)}")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": item["id"],
                        "Вопрос": item["question"],
                        "Ожидание": item["expectation"],
                        "Источники": ", ".join(item["expected_sources"]) or "нет",
                    }
                    for item in benchmark_questions
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        benchmark_a, benchmark_b, benchmark_c = st.columns(3)
        benchmark_model = benchmark_a.text_input(
            "Модель оценки", value=OLLAMA_MODEL, key="benchmark-model"
        )
        benchmark_strategy = benchmark_b.selectbox(
            "Индекс оценки", STRATEGIES, index=1, key="benchmark-strategy"
        )
        benchmark_candidate_top_k = benchmark_c.slider(
            "Candidate Top-K оценки", 1, 20, 20, key="benchmark-candidate-top-k"
        )
        benchmark_d, benchmark_e, benchmark_f = st.columns(3)
        benchmark_threshold = benchmark_d.slider(
            "Similarity threshold оценки",
            0.0,
            1.0,
            0.35,
            0.01,
            key="benchmark-threshold",
        )
        if st.session_state.get("benchmark-final-top-k", 1) > int(
            benchmark_candidate_top_k
        ):
            st.session_state["benchmark-final-top-k"] = int(
                benchmark_candidate_top_k
            )
        benchmark_final_top_k = benchmark_e.slider(
            "Final Top-K оценки",
            1,
            int(benchmark_candidate_top_k),
            min(5, int(benchmark_candidate_top_k)),
            key="benchmark-final-top-k",
        )
        benchmark_rewrite = benchmark_f.checkbox(
            "Query Rewrite в Enhanced",
            value=True,
            key="benchmark-rewrite",
        )
        benchmark_reranker_label = st.selectbox(
            "Reranker оценки",
            ("Heuristic", "NoOp"),
            key="benchmark-reranker",
        )
        threshold_values_text = st.text_input(
            "Threshold experiment — значения через запятую",
            value="0.20, 0.30, 0.35, 0.40, 0.50",
            key="threshold-values",
        )
        st.caption(
            "Threshold experiment изолирует влияние порога: Query Rewrite в нём выключен."
        )
        run_benchmark_button, run_threshold_button = st.columns(2)
        if run_benchmark_button.button(
            "Запустить сравнение 10 вопросов",
            type="primary",
            key="run-reranking-benchmark",
        ):
            try:
                provider = OllamaProvider(
                    benchmark_model, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
                )
                connected, message = provider.check_connection()
                if not connected:
                    raise LLMError(message)
                progress = st.progress(0.0, text="Подготовка контрольного прогона")
                embedding_service = get_embedding_service()
                retriever = SemanticRetriever(embedding_service)
                candidate_searcher = SemanticCandidateSearcher(embedding_service)
                benchmark_config = RetrievalConfig(
                    candidate_top_k=int(benchmark_candidate_top_k),
                    final_top_k=int(benchmark_final_top_k),
                    similarity_threshold=float(benchmark_threshold),
                    use_query_rewrite=bool(benchmark_rewrite),
                    reranker_type=(
                        "heuristic"
                        if benchmark_reranker_label == "Heuristic"
                        else "none"
                    ),
                    max_alternative_queries=2,
                    max_chunks_per_source=3,
                )

                def update_reranking_benchmark(
                    position: int,
                    total: int,
                    question_id: str,
                    stage: str,
                ) -> None:
                    progress.progress(
                        position / total,
                        text=f"{question_id}: {stage} · {position} из {total}",
                    )

                batch_id, benchmark_results = run_reranking_benchmark(
                    benchmark_questions,
                    provider,
                    result_storage,
                    strategy=benchmark_strategy,
                    baseline_top_k=5,
                    config=benchmark_config,
                    retriever=retriever,
                    candidate_searcher=candidate_searcher,
                    tokenizer=embedding_service.tokenizer,
                    progress_callback=update_reranking_benchmark,
                )
                report_path = write_reranking_report(
                    benchmark_results,
                    model_name=benchmark_model,
                    strategy=benchmark_strategy,
                    config=benchmark_config,
                    threshold_results=st.session_state.get("threshold-results"),
                    path=RERANKING_REPORT_PATH,
                )
                st.session_state["reranking-benchmark-results"] = benchmark_results
                st.session_state["reranking-benchmark-config"] = benchmark_config
                st.session_state["rag-benchmark-batch"] = batch_id
                st.success(f"Оценка завершена. Отчёт: {report_path}")
            except Exception as exc:
                logger.exception("Ошибка автоматической оценки reranking")
                st.error(f"Оценка не выполнена: {exc}")

        if run_threshold_button.button(
            "Запустить threshold experiment",
            key="run-threshold-experiment",
        ):
            try:
                threshold_values = [
                    float(value.strip())
                    for value in threshold_values_text.split(",")
                    if value.strip()
                ]
                embedding_service = get_embedding_service()
                benchmark_config = RetrievalConfig(
                    candidate_top_k=int(benchmark_candidate_top_k),
                    final_top_k=int(benchmark_final_top_k),
                    similarity_threshold=float(benchmark_threshold),
                    use_query_rewrite=False,
                    reranker_type=(
                        "heuristic"
                        if benchmark_reranker_label == "Heuristic"
                        else "none"
                    ),
                )
                threshold_results = run_threshold_experiment(
                    benchmark_questions,
                    threshold_values,
                    strategy=benchmark_strategy,
                    config=benchmark_config,
                    rewrite_llm=None,
                    searcher=SemanticCandidateSearcher(embedding_service),
                )
                threshold_batch_id = f"threshold-{uuid4().hex}"
                for item in threshold_results:
                    result_storage.save_benchmark_result(
                        threshold_batch_id,
                        f"threshold-{item['threshold']}",
                        {
                            "type": "threshold_experiment",
                            "strategy": benchmark_strategy,
                            "model_name": benchmark_model,
                            "config": {
                                "candidate_top_k": benchmark_config.candidate_top_k,
                                "final_top_k": benchmark_config.final_top_k,
                                "reranker_type": benchmark_config.reranker_type,
                            },
                            "metrics": item,
                        },
                    )
                st.session_state["threshold-results"] = threshold_results
                current_benchmark = st.session_state.get(
                    "reranking-benchmark-results"
                )
                if current_benchmark:
                    write_reranking_report(
                        current_benchmark,
                        model_name=benchmark_model,
                        strategy=benchmark_strategy,
                        config=st.session_state.get(
                            "reranking-benchmark-config", benchmark_config
                        ),
                        threshold_results=threshold_results,
                        path=RERANKING_REPORT_PATH,
                    )
                st.success("Эксперимент с порогами завершён.")
            except Exception as exc:
                logger.exception("Ошибка threshold experiment")
                st.error(f"Эксперимент не выполнен: {exc}")

        current_benchmark = st.session_state.get("reranking-benchmark-results")
        if current_benchmark:
            st.markdown("### Итоговые retrieval-метрики")
            aggregate = aggregate_reranking_metrics(current_benchmark)
            st.json(aggregate)
            chart_rows = [
                {
                    "Метрика": metric,
                    "Baseline": aggregate[f"baseline_{metric}"],
                    "Enhanced": aggregate[f"enhanced_{metric}"],
                }
                for metric in ("hit_at_1", "hit_at_3", "hit_at_k", "mrr", "precision_at_k")
            ]
            st.plotly_chart(
                px.bar(
                    pd.DataFrame(chart_rows),
                    x="Метрика",
                    y=["Baseline", "Enhanced"],
                    barmode="group",
                    title="Retrieval-качество Baseline и Enhanced",
                ),
                width="stretch",
            )
            detail_rows = [
                {
                    "Метрика": "Первая релевантная позиция",
                    "Baseline": aggregate["baseline_mean_first_relevant_rank"],
                    "Enhanced": aggregate["enhanced_mean_first_relevant_rank"],
                },
                {
                    "Метрика": "Keyword coverage ответа",
                    "Baseline": aggregate["baseline_keyword_coverage"],
                    "Enhanced": aggregate["enhanced_keyword_coverage"],
                },
            ]
            st.plotly_chart(
                px.bar(
                    pd.DataFrame(detail_rows),
                    x="Метрика",
                    y=["Baseline", "Enhanced"],
                    barmode="group",
                    title="Позиция релевантного источника и качество ответа",
                ),
                width="stretch",
            )
            timing_frame = pd.DataFrame(
                [
                    {"Этап": "Rewrite", "Секунды": aggregate["mean_rewrite_time"]},
                    {"Этап": "Filter", "Секунды": aggregate["mean_filter_time"]},
                    {"Этап": "Rerank", "Секунды": aggregate["mean_rerank_time"]},
                    {
                        "Этап": "Retrieval total",
                        "Секунды": aggregate["mean_total_retrieval_time"],
                    },
                ]
            )
            st.plotly_chart(
                px.bar(
                    timing_frame,
                    x="Этап",
                    y="Секунды",
                    title="Среднее время этапов Enhanced Retrieval",
                ),
                width="stretch",
            )
            st.metric(
                "Вопросов, где Enhanced Retrieval лучше",
                aggregate["enhanced_better_question_count"],
            )
            benchmark_labels = {
                f"{result['question']['id']}: {result['question']['question']}": result
                for result in current_benchmark
            }
            selected_question = st.selectbox(
                "Просмотреть контрольный вопрос",
                list(benchmark_labels),
                key="benchmark-question-details",
            )
            selected_result = benchmark_labels[selected_question]
            if selected_result.get("error"):
                st.error(selected_result["error"])
            else:
                benchmark_left, benchmark_right = st.columns(2)
                benchmark_left.markdown("**Baseline RAG**")
                benchmark_left.write(selected_result["baseline"]["answer"])
                benchmark_right.markdown("**Enhanced RAG**")
                benchmark_right.write(selected_result["enhanced"]["answer"])
                st.json(selected_result["metrics"])
                diagnostics = selected_result["enhanced"].get(
                    "retrieval_diagnostics", {}
                )
                funnel = pd.DataFrame(
                    [
                        {
                            "Этап": "Candidates",
                            "Количество": diagnostics.get("candidate_count", 0),
                        },
                        {
                            "Этап": "Accepted",
                            "Количество": diagnostics.get("accepted_count", 0),
                        },
                        {
                            "Этап": "Reranked",
                            "Количество": diagnostics.get("reranked_count", 0),
                        },
                        {
                            "Этап": "Final",
                            "Количество": diagnostics.get("final_count", 0),
                        },
                    ]
                )
                st.plotly_chart(
                    px.bar(
                        funnel,
                        x="Этап",
                        y="Количество",
                        title="Retrieval funnel выбранного вопроса",
                    ),
                    width="stretch",
                )
                candidate_rows = diagnostics.get("initial_candidates", [])
                if candidate_rows:
                    candidate_frame = pd.DataFrame(candidate_rows)
                    st.plotly_chart(
                        px.scatter(
                            candidate_frame,
                            x="original_rank",
                            y="similarity",
                            color="status",
                            hover_data=["chunk_id", "source"],
                            title="Similarity принятых и отброшенных кандидатов",
                        ),
                        width="stretch",
                    )

        threshold_results = st.session_state.get("threshold-results")
        if threshold_results:
            st.markdown("### Зависимость качества от similarity threshold")
            threshold_frame = pd.DataFrame(threshold_results)
            st.dataframe(threshold_frame, width="stretch", hide_index=True)
            st.plotly_chart(
                px.line(
                    threshold_frame,
                    x="threshold",
                    y=["source_hit_at_k", "precision_at_k", "empty_result_share"],
                    markers=True,
                    title="Качество и пустые результаты по threshold",
                ),
                width="stretch",
            )
    except Exception as exc:
        logger.exception("Ошибка загрузки контрольных вопросов")
        st.error(f"Контрольные вопросы недоступны: {exc}")


with tab_settings:
    st.subheader("Управление сохранёнными данными")
    counts = result_storage.counts()
    count_a, count_b, count_c, count_d, count_e = st.columns(5)
    count_a.metric("Сравнения", counts["comparisons"])
    count_b.metric("Ответы", counts["answers"])
    count_c.metric("Retrieval-кандидаты", result_storage.retrieval_candidate_count())
    count_d.metric("Ручные оценки", counts["manual_evaluations"])
    count_e.metric("Контрольные результаты", counts["benchmark_results"])

    if st.button("Экспортировать результаты в JSON", key="export-rag-results"):
        try:
            exported_path = result_storage.export_json(RAG_RESULTS_PATH)
            st.success(f"Экспорт создан: {exported_path}")
        except Exception as exc:
            logger.exception("Ошибка JSON-экспорта")
            st.error(f"Экспорт не выполнен: {exc}")
    if RAG_RESULTS_PATH.is_file():
        st.download_button(
            "Скачать JSON",
            data=RAG_RESULTS_PATH.read_bytes(),
            file_name="rag_results.json",
            mime="application/json",
            key="download-rag-json",
        )

    st.markdown("### Безопасный сброс результатов")
    st.warning(
        "Эти действия не удаляют документы, контрольные вопросы, FAISS или SQLite "
        "метаданных индекса."
    )
    confirm_history = st.checkbox(
        "Я понимаю, что история сравнений будет удалена",
        key="confirm-clear-rag-history",
    )
    if st.button(
        "Очистить историю сравнений",
        disabled=not confirm_history,
        key="clear-rag-history",
    ):
        try:
            deleted = clear_sqlite_results(RESULTS_DATABASE, "history")
            st.session_state.pop("rag-results", None)
            st.session_state.pop("rag-run-id", None)
            st.success(f"Удалено записей: {deleted}")
            st.rerun()
        except Exception as exc:
            logger.exception("Откат очистки истории RAG")
            st.error(f"Очистка отменена: {exc}")

    confirm_benchmark = st.checkbox(
        "Я понимаю, что результаты контрольных вопросов будут удалены",
        key="confirm-clear-rag-benchmark",
    )
    if st.button(
        "Очистить результаты контрольных вопросов",
        disabled=not confirm_benchmark,
        key="clear-rag-benchmark",
    ):
        try:
            deleted = clear_sqlite_results(RESULTS_DATABASE, "benchmark")
            st.session_state.pop("reranking-benchmark-results", None)
            st.session_state.pop("threshold-results", None)
            st.session_state.pop("rag-benchmark-batch", None)
            st.success(f"Удалено записей: {deleted}")
            st.rerun()
        except Exception as exc:
            logger.exception("Откат очистки benchmark")
            st.error(f"Очистка отменена: {exc}")

    confirm_all = st.checkbox(
        "Я понимаю, что все сохранённые результаты RAG будут удалены",
        key="confirm-clear-all-rag",
    )
    if st.button(
        "Сбросить все результаты RAG",
        disabled=not confirm_all,
        key="clear-all-rag-results",
    ):
        try:
            deleted = clear_sqlite_results(RESULTS_DATABASE, "all_results")
            clear_json_results(RAG_RESULTS_PATH)
            for key in (
                "rag-results",
                "rag-run-id",
                "reranking-benchmark-results",
                "reranking-benchmark-config",
                "threshold-results",
                "rag-benchmark-batch",
            ):
                st.session_state.pop(key, None)
            st.success(f"Удалено SQLite-записей: {deleted}; JSON очищен.")
            st.rerun()
        except Exception as exc:
            logger.exception("Откат полного сброса RAG")
            st.error(f"Сброс отменён: {exc}")

    st.markdown("### Опасная зона: удаление индексов")
    st.error(
        "Удаление индексов не относится к сбросу результатов. После него потребуется "
        "повторная индексация документов."
    )
    index_selection_label = st.radio(
        "Удаляемые индексы",
        ["fixed", "structural", "both"],
        horizontal=True,
        key="delete-index-selection",
    )
    reset_word = st.text_input(
        "Для подтверждения введите СБРОС",
        key="delete-index-confirmation",
    )
    if st.button(
        "Удалить выбранные индексы",
        type="primary",
        disabled=reset_word != "СБРОС",
        key="delete-indexes",
    ):
        try:
            deleted_files = delete_indexes(index_selection_label, INDEX_ROOT)
            st.success(
                f"Удалено файлов: {len(deleted_files)}. Выполните повторную индексацию."
            )
            st.rerun()
        except Exception as exc:
            logger.exception("Ошибка безопасного удаления индексов")
            st.error(f"Индексы не удалены: {exc}")
