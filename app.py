"""Streamlit-интерфейс локального индексатора документов."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    DOCUMENTS_DIR,
    INDEX_ROOT,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    RAG_QUESTIONS_PATH,
    RAG_REPORT_PATH,
    RAG_RESULTS_PATH,
    RESULTS_DATABASE,
    STRATEGIES,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    configure_logging,
    ensure_directories,
)
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
from src.models import AnswerResult
from src.pipeline import build_indexes
from src.rag import SemanticRetriever, compare_answers
from src.rag_evaluation import (
    aggregate_metrics,
    load_questions,
    run_benchmark,
    write_rag_report,
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
    st.session_state.pop("rag-pair", None)
    st.session_state.pop("rag-run-id", None)


def render_rag_pair(without_rag: AnswerResult, with_rag: AnswerResult) -> None:
    """Показать два ответа, метрики, источники и найденные чанки."""
    left, right = st.columns(2)
    with left:
        st.markdown("### Без RAG")
        st.markdown(without_rag.answer)
        st.caption(
            f"Модель: {without_rag.model_name} · генерация: "
            f"{without_rag.generation_time:.3f} с"
        )
        st.info("Локальные документы не использовались.")
    with right:
        st.markdown("### С RAG")
        st.markdown(with_rag.answer)
        st.caption(
            f"Модель: {with_rag.model_name} · стратегия: {with_rag.strategy} · "
            f"поиск: {with_rag.retrieval_time:.3f} с · "
            f"генерация: {with_rag.generation_time:.3f} с"
        )
        if with_rag.sources:
            st.markdown(
                "**Источники:** "
                + "; ".join(
                    f"[{source['number']}] {source['source']}"
                    + (f" — {source['section']}" if source.get("section") else "")
                    for source in with_rag.sources
                )
            )
        else:
            st.warning("Релевантные локальные источники отсутствуют.")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Показатель": "Время поиска, с",
                    "Без RAG": 0.0,
                    "С RAG": round(with_rag.retrieval_time, 4),
                },
                {
                    "Показатель": "Время генерации, с",
                    "Без RAG": round(without_rag.generation_time, 4),
                    "С RAG": round(with_rag.generation_time, 4),
                },
                {
                    "Показатель": "Найдено чанков",
                    "Без RAG": 0,
                    "С RAG": len(with_rag.retrieved_chunks),
                },
                {
                    "Показатель": "Источников",
                    "Без RAG": 0,
                    "С RAG": len(with_rag.sources),
                },
                {
                    "Показатель": "Контекст, токенов",
                    "Без RAG": 0,
                    "С RAG": with_rag.context_token_count,
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander(
        f"Найденные чанки и similarity score ({len(with_rag.retrieved_chunks)})"
    ):
        if not with_rag.retrieved_chunks:
            st.info("Чанки не найдены.")
        for position, chunk in enumerate(with_rag.retrieved_chunks, start=1):
            page = chunk.page_number if chunk.page_number is not None else "—"
            st.markdown(
                f"**Источник [{chunk.source_number or position}] · "
                f"similarity `{chunk.score:.4f}`**  \n"
                f"Файл: `{chunk.source}` · Заголовок: `{chunk.title or '—'}` · "
                f"Раздел: `{chunk.section or '—'}` · Страница: `{page}`  \n"
                f"Chunk ID: `{chunk.chunk_id}`"
            )
            st.code(chunk.content, language=None, wrap_lines=True)


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
 tab_rag, tab_rag_evaluation, tab_settings) = st.tabs(
    [
        "Документы",
        "Индексация",
        "Чанки",
        "Поиск",
        "Сравнение",
        "Аналитика",
        "Сравнение RAG",
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
    st.subheader("Сравнение ответов с RAG и без RAG")
    st.caption(
        "Одна локальная Ollama-модель получает один вопрос, одинаковую temperature "
        "и одинаковый стиль ответа. Отличается только наличие контекста."
    )
    rag_question = st.text_area(
        "Вопрос",
        height=100,
        key="rag-question",
        placeholder="Например: Как обеспечивается соответствие FAISS и SQLite?",
    )
    rag_model = st.text_input("Модель Ollama", value=OLLAMA_MODEL, key="rag-model")
    st.caption(f"Локальный сервер: `{OLLAMA_BASE_URL}`")
    rag_a, rag_b, rag_c, rag_d = st.columns(4)
    rag_strategy = rag_a.selectbox("Индекс RAG", STRATEGIES, index=1, key="rag-strategy")
    rag_top_k = rag_b.slider("top_k", 1, 20, 5, key="rag-top-k")
    rag_max_tokens = rag_c.number_input(
        "Контекст, токенов",
        min_value=100,
        max_value=16_000,
        value=4000,
        step=100,
        key="rag-max-context",
    )
    rag_temperature = rag_d.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        key="rag-temperature",
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
        "Сравнить ответы",
        type="primary",
        disabled=not rag_question.strip(),
        key="compare-rag-answers",
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
            "query_embedding": "✓ Генерация embedding вопроса",
            "retrieval": "✓ Поиск релевантных чанков",
            "context": "✓ Формирование контекста",
            "rag_generation": "✓ Ответ с RAG",
        }
        try:
            provider = OllamaProvider(rag_model, OLLAMA_BASE_URL, OLLAMA_TIMEOUT)
            connected, message = provider.check_connection()
            if not connected:
                raise LLMError(message)
            status.write("✓ Проверка подключения к LLM")
            retriever = SemanticRetriever(get_embedding_service())
            without_rag, with_rag = compare_answers(
                rag_question,
                provider,
                strategy=rag_strategy,
                top_k=int(rag_top_k),
                max_context_tokens=int(rag_max_tokens),
                temperature=float(rag_temperature),
                retriever=retriever,
                tokenizer=retriever.tokenizer,
                progress_callback=lambda stage: status.write(stage_labels[stage]),
            )
            run_id = result_storage.save_comparison(
                without_rag,
                with_rag,
                strategy=rag_strategy,
                top_k=int(rag_top_k),
                max_context_tokens=int(rag_max_tokens),
                temperature=float(rag_temperature),
            )
            st.session_state["rag-pair"] = (without_rag, with_rag)
            st.session_state["rag-run-id"] = run_id
            status.update(label="RAG-сравнение завершено и сохранено", state="complete")
        except Exception as exc:
            logger.exception("Ошибка RAG-сравнения без записи контекста")
            status.update(label="RAG-сравнение остановлено", state="error")
            st.error(str(exc))

    rag_pair = st.session_state.get("rag-pair")
    if rag_pair:
        without_rag_result, with_rag_result = rag_pair
        render_rag_pair(without_rag_result, with_rag_result)
        current_run_id = st.session_state.get("rag-run-id")
        if current_run_id:
            st.markdown("### Ручная оценка")
            preference_labels = {
                "Без RAG": "no_rag",
                "С RAG": "rag",
                "Одинаково": "equal",
                "Оба неверны": "both_wrong",
            }
            with st.form(f"manual-evaluation-{current_run_id}"):
                evaluation_a, evaluation_b, evaluation_c = st.columns(3)
                preferred_label = evaluation_a.selectbox(
                    "Какой ответ лучше", list(preference_labels)
                )
                score_without = evaluation_b.select_slider(
                    "Оценка без RAG", options=[0, 1, 2], value=1
                )
                score_with = evaluation_c.select_slider(
                    "Оценка с RAG", options=[0, 1, 2], value=1
                )
                unsupported = st.checkbox("Есть неподтверждённые утверждения")
                evaluation_comment = st.text_area("Комментарий")
                save_evaluation = st.form_submit_button("Сохранить оценку")
            if save_evaluation:
                try:
                    result_storage.save_manual_evaluation(
                        int(current_run_id),
                        preferred_mode=preference_labels[preferred_label],
                        score_without_rag=int(score_without),
                        score_with_rag=int(score_with),
                        has_unsupported_claims=unsupported,
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
                            "Стратегия": row["strategy"],
                            "top_k": row["top_k"],
                            "Без RAG": row["answer_without_rag"][:120],
                            "С RAG": row["answer_with_rag"][:120],
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
                    st.json(details)
    except Exception as exc:
        logger.exception("Ошибка чтения истории RAG")
        st.error(f"История недоступна: {exc}")


with tab_rag_evaluation:
    st.subheader("Автоматическая оценка RAG")
    try:
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
            "Стратегия оценки", STRATEGIES, index=1, key="benchmark-strategy"
        )
        benchmark_top_k = benchmark_c.slider(
            "top_k оценки", 1, 20, 5, key="benchmark-top-k"
        )
        if st.button("Запустить оценку", type="primary", key="run-rag-benchmark"):
            try:
                provider = OllamaProvider(
                    benchmark_model, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
                )
                connected, message = provider.check_connection()
                if not connected:
                    raise LLMError(message)
                progress = st.progress(0.0, text="Подготовка контрольного прогона")
                retriever = SemanticRetriever(get_embedding_service())

                def update_benchmark(position: int, total: int, question_id: str) -> None:
                    progress.progress(
                        position / total,
                        text=f"{question_id}: {position} из {total}",
                    )

                batch_id, benchmark_results = run_benchmark(
                    benchmark_questions,
                    provider,
                    result_storage,
                    strategy=benchmark_strategy,
                    top_k=int(benchmark_top_k),
                    retriever=retriever,
                    tokenizer=retriever.tokenizer,
                    progress_callback=update_benchmark,
                )
                report_path = write_rag_report(
                    benchmark_results,
                    model_name=benchmark_model,
                    strategy=benchmark_strategy,
                    top_k=int(benchmark_top_k),
                    temperature=0.1,
                    path=RAG_REPORT_PATH,
                )
                st.session_state["rag-benchmark-results"] = benchmark_results
                st.session_state["rag-benchmark-batch"] = batch_id
                st.success(f"Оценка завершена. Отчёт: {report_path}")
            except Exception as exc:
                logger.exception("Ошибка автоматической оценки RAG")
                st.error(f"Оценка не выполнена: {exc}")

        current_benchmark = st.session_state.get("rag-benchmark-results")
        if current_benchmark:
            st.markdown("### Итоговые метрики")
            st.json(aggregate_metrics(current_benchmark))
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
                benchmark_left.markdown("**Без RAG**")
                benchmark_left.write(selected_result["without_rag"]["answer"])
                benchmark_right.markdown("**С RAG**")
                benchmark_right.write(selected_result["with_rag"]["answer"])
                st.json(selected_result["metrics"])
    except Exception as exc:
        logger.exception("Ошибка загрузки контрольных вопросов")
        st.error(f"Контрольные вопросы недоступны: {exc}")


with tab_settings:
    st.subheader("Управление сохранёнными данными")
    counts = result_storage.counts()
    count_a, count_b, count_c, count_d = st.columns(4)
    count_a.metric("Сравнения", counts["comparisons"])
    count_b.metric("Ответы", counts["answers"])
    count_c.metric("Ручные оценки", counts["manual_evaluations"])
    count_d.metric("Контрольные результаты", counts["benchmark_results"])

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
            st.session_state.pop("rag-pair", None)
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
            st.session_state.pop("rag-benchmark-results", None)
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
                "rag-pair",
                "rag-run-id",
                "rag-benchmark-results",
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
