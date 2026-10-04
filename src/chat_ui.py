"""Streamlit-компонент постоянного RAG-чата; бизнес-логика остаётся в сервисах."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from .chat_models import TaskState
from .chat_service import ChatService
from .chat_storage import ChatStorage
from .config import CHAT_EXPORT_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from .llm import OllamaProvider
from .models import RetrievalConfig
from .retrieval import SemanticCandidateSearcher
from .storage import IndexStorage
from .task_memory import diff_task_states, task_state_from_manual_payload

_STAGE_LABELS = {
    "updating_memory": "Обновление памяти",
    "contextualizing": "Формирование standalone query",
    "retrieving": "Enhanced Retrieval",
    "generating": "Генерация ответа",
}


def _lines(values: list[str]) -> str:
    return "\n".join(values)


def _parse_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _default_settings() -> dict[str, Any]:
    return {
        "retrieval_mode": "enhanced_rag",
        "candidate_top_k": 20,
        "final_top_k": 5,
        "similarity_threshold": 0.35,
        "reranker_type": "heuristic",
        "use_query_rewrite": True,
        "max_alternative_queries": 2,
        "max_chunks_per_source": 3,
        "recent_message_count": 8,
        "summary_trigger_messages": 12,
        "temperature": 0.1,
        "automatic_memory": True,
        "automatic_summary": True,
        "context_budget": {
            "task_state_tokens": 800,
            "summary_tokens": 1000,
            "recent_messages_tokens": 1500,
            "rag_context_tokens": 4000,
            "response_reserve_tokens": 1500,
        },
    }


def _retrieval_config(settings: dict[str, Any]) -> RetrievalConfig:
    return RetrievalConfig(
        candidate_top_k=int(settings.get("candidate_top_k", 20)),
        final_top_k=int(settings.get("final_top_k", 5)),
        similarity_threshold=float(settings.get("similarity_threshold", 0.35)),
        use_query_rewrite=bool(settings.get("use_query_rewrite", True)),
        reranker_type=str(settings.get("reranker_type", "heuristic")),
        max_alternative_queries=int(settings.get("max_alternative_queries", 2)),
        max_chunks_per_source=int(settings.get("max_chunks_per_source", 3)),
    )


def _render_sources(storage: ChatStorage, message_id: str, strategy: str) -> None:
    sources = storage.list_sources(message_id)
    st.markdown("**Источники**")
    if not sources:
        st.caption("Релевантные источники в локальной базе не найдены.")
        return
    index_storage = IndexStorage(strategy)
    for source in sources:
        location = source.get("section") or source.get("title") or "без раздела"
        page = f", стр. {source['page_number']}" if source.get("page_number") else ""
        st.markdown(
            f"[{source['number']}] `{source['source']}` — {location}{page}  "
            f"Similarity: {source.get('similarity', 0.0):.3f}"
        )
        with st.expander(f"Чанк {source['chunk_id']}"):
            st.json(
                {
                    key: value
                    for key, value in source.items()
                    if key not in {"number"}
                }
            )
            chunk = index_storage.fetch_by_chunk_id(str(source["chunk_id"]))
            if chunk:
                st.code(chunk["content"], language=None, wrap_lines=True)
            else:
                st.info(
                    "Полный текст чанка недоступен в текущем индексе; "
                    "показан сохранённый snapshot метаданных."
                )


def _render_message(storage: ChatStorage, message, strategy: str) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if message.status == "failed":
            st.error(message.error_message or "Обработка сообщения завершилась ошибкой.")
        if message.role == "assistant":
            _render_sources(storage, message.id, strategy)
            with st.expander("Диагностика хода"):
                st.write(f"Статус: `{message.status}`")
                st.write(
                    "Standalone query: "
                    f"{message.standalone_query or 'не сохранён'}"
                )
                st.write(
                    "Rewritten query: "
                    f"{message.rewritten_query or 'не использован'}"
                )
                diagnostics = message.retrieval_diagnostics
                if diagnostics:
                    st.write(
                        "Воронка: "
                        f"{diagnostics.get('candidate_count', 0)} → "
                        f"{diagnostics.get('accepted_count', 0)} → "
                        f"{diagnostics.get('reranked_count', 0)} → "
                        f"{diagnostics.get('final_count', 0)}"
                    )
                    st.json(diagnostics)
                if message.token_usage:
                    st.markdown("**Токены блоков**")
                    st.json(message.token_usage)


def _render_memory_editor(storage: ChatStorage, conversation_id: str, state: TaskState) -> None:
    with st.expander("Редактировать память"):
        with st.form(f"memory-edit-{conversation_id}"):
            goal = st.text_area("Цель", value=state.goal or "")
            constraints = st.text_area("Ограничения — по одному на строку", value=_lines(state.constraints))
            terms = st.text_area(
                "Термины — JSON-объект",
                value=json.dumps(state.terms, ensure_ascii=False, indent=2),
            )
            facts = st.text_area("Подтверждённые факты", value=_lines(state.confirmed_facts))
            decisions = st.text_area("Решения", value=_lines(state.decisions))
            questions = st.text_area("Открытые вопросы", value=_lines(state.open_questions))
            preferences = st.text_area("Предпочтения", value=_lines(state.user_preferences))
            save = st.form_submit_button("Сохранить ручную версию", type="primary")
        if save:
            try:
                candidate = task_state_from_manual_payload(
                    {
                        "goal": goal.strip() or None,
                        "constraints": _parse_lines(constraints),
                        "terms": json.loads(terms or "{}"),
                        "confirmed_facts": _parse_lines(facts),
                        "decisions": _parse_lines(decisions),
                        "open_questions": _parse_lines(questions),
                        "user_preferences": _parse_lines(preferences),
                    }
                )
                change = diff_task_states(state, candidate, reason="Ручное редактирование.")
                if not change.has_changes:
                    st.info("Содержательных изменений нет; версия не увеличена.")
                else:
                    storage.save_manual_task_state(conversation_id, candidate, change)
                    st.success("Ручная версия памяти сохранена.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Память не сохранена: {exc}")


def _render_memory_panel(storage: ChatStorage, conversation_id: str) -> None:
    state = storage.latest_task_state(conversation_id)
    st.markdown("### Память задачи")
    st.caption(f"Версия {state.version} · {state.updated_at or 'ещё не обновлялась'}")
    st.markdown(f"**Цель:** {state.goal or '—'}")
    st.markdown("**Ограничения:**")
    if state.constraints:
        for item in state.constraints:
            st.markdown(f"- {item}")
    else:
        st.caption("—")
    st.markdown("**Термины:**")
    if state.terms:
        for key, value in state.terms.items():
            st.markdown(f"- `{key}` = {value}")
    else:
        st.caption("—")
    for label, values in (
        ("Подтверждённые факты", state.confirmed_facts),
        ("Решения", state.decisions),
        ("Открытые вопросы", state.open_questions),
        ("Предпочтения", state.user_preferences),
    ):
        with st.expander(label):
            if values:
                for item in values:
                    st.markdown(f"- {item}")
            else:
                st.caption("—")

    versions = storage.list_task_state_versions(conversation_id)
    if versions:
        latest = versions[0]
        with st.expander("Последние изменения памяти"):
            st.json(latest["change"])
        with st.expander("История версий"):
            st.dataframe(
                [
                    {
                        "Версия": item["version"],
                        "Источник": item["source"],
                        "Создана": item["created_at"],
                        "Причина": item["change"].get("reason") or "—",
                    }
                    for item in versions
                ],
                hide_index=True,
                width="stretch",
            )
            selected = st.selectbox(
                "Версия для восстановления",
                [item["version"] for item in versions],
                key=f"restore-version-{conversation_id}",
            )
            confirm_restore = st.checkbox(
                "Подтверждаю восстановление как новой версии",
                key=f"confirm-restore-{conversation_id}",
            )
            if st.button(
                "Восстановить версию",
                disabled=not confirm_restore,
                key=f"restore-memory-{conversation_id}",
            ):
                storage.restore_task_state(conversation_id, int(selected))
                st.success("Версия восстановлена как новая.")
                st.rerun()

    _render_memory_editor(storage, conversation_id, state)
    confirm_clear = st.checkbox(
        "Подтверждаю очистку только памяти",
        key=f"confirm-clear-memory-{conversation_id}",
    )
    if st.button(
        "Очистить память",
        disabled=not confirm_clear,
        key=f"clear-memory-{conversation_id}",
    ):
        storage.clear_task_state(conversation_id)
        st.success("Память очищена; сообщения сохранены.")
        st.rerun()


def _render_settings(storage: ChatStorage, conversation) -> None:
    settings = {**_default_settings(), **conversation.settings}
    budget = {**_default_settings()["context_budget"], **settings.get("context_budget", {})}
    with st.expander("Настройки текущего диалога"):
        with st.form(f"chat-settings-{conversation.id}"):
            model_name = st.text_input("Модель Ollama", value=conversation.model_name)
            strategy = st.selectbox(
                "Индекс",
                ["fixed", "structural"],
                index=["fixed", "structural"].index(conversation.strategy),
            )
            st.selectbox("Retrieval mode", ["enhanced_rag"], disabled=True)
            candidate_top_k = st.slider(
                "Candidate Top-K", 1, 20, int(settings["candidate_top_k"])
            )
            final_default = min(int(settings["final_top_k"]), candidate_top_k)
            final_top_k = st.slider("Final Top-K", 1, candidate_top_k, final_default)
            threshold = st.slider(
                "Similarity threshold",
                0.0,
                1.0,
                float(settings["similarity_threshold"]),
                0.01,
            )
            reranker = st.selectbox(
                "Reranker",
                ["heuristic", "none"],
                index=["heuristic", "none"].index(settings["reranker_type"]),
            )
            rewrite = st.toggle(
                "Query Rewrite", value=bool(settings["use_query_rewrite"])
            )
            alternatives = st.slider(
                "Альтернативные запросы",
                0,
                3,
                int(settings["max_alternative_queries"]),
            )
            source_limit = st.slider(
                "Максимум чанков источника",
                1,
                10,
                int(settings["max_chunks_per_source"]),
            )
            recent = st.slider(
                "Recent messages", 2, 20, int(settings["recent_message_count"])
            )
            summary_trigger = st.slider(
                "Summary после сообщений",
                4,
                40,
                int(settings["summary_trigger_messages"]),
            )
            rag_budget = st.number_input(
                "RAG-контекст, токенов",
                min_value=100,
                max_value=20000,
                value=int(budget["rag_context_tokens"]),
                step=100,
            )
            temperature = st.slider(
                "Temperature", 0.0, 2.0, float(settings["temperature"]), 0.05
            )
            automatic_memory = st.toggle(
                "Автоматически обновлять память",
                value=bool(settings["automatic_memory"]),
            )
            automatic_summary = st.toggle(
                "Автоматически обновлять summary",
                value=bool(settings["automatic_summary"]),
            )
            save = st.form_submit_button("Сохранить настройки")
        if save:
            settings.update(
                {
                    "retrieval_mode": "enhanced_rag",
                    "candidate_top_k": candidate_top_k,
                    "final_top_k": final_top_k,
                    "similarity_threshold": threshold,
                    "reranker_type": reranker,
                    "use_query_rewrite": rewrite,
                    "max_alternative_queries": alternatives,
                    "max_chunks_per_source": source_limit,
                    "recent_message_count": recent,
                    "summary_trigger_messages": summary_trigger,
                    "temperature": temperature,
                    "automatic_memory": automatic_memory,
                    "automatic_summary": automatic_summary,
                    "context_budget": {**budget, "rag_context_tokens": int(rag_budget)},
                }
            )
            storage.update_conversation(
                conversation.id,
                model_name=model_name,
                strategy=strategy,
                settings=settings,
            )
            st.success("Настройки сохранены для новых сообщений.")
            st.rerun()


def render_rag_chat(
    storage: ChatStorage,
    embedding_service_factory: Callable[[], Any],
) -> None:
    """Отобразить вкладку создания, продолжения и управления RAG-диалогами."""
    st.subheader("RAG-чат")
    st.caption(
        "История, память задачи, summary, recent messages и RAG-контекст "
        "хранятся и обрабатываются отдельно."
    )
    left, center, right = st.columns([1.15, 2.8, 1.65])

    def create_new_conversation() -> None:
        created = storage.create_conversation(
            model_name=OLLAMA_MODEL,
            strategy="structural",
            settings=_default_settings(),
        )
        st.session_state["chat-conversation-id"] = created.id

    with left:
        st.button(
            "Новый диалог",
            type="primary",
            width="stretch",
            key="create-chat-conversation",
            on_click=create_new_conversation,
        )
        conversations = storage.list_conversations()
        if not conversations:
            st.info("Создайте первый диалог.")
            return
        ids = [conversation.id for conversation in conversations]
        if st.session_state.get("chat-conversation-id") not in ids:
            st.session_state["chat-conversation-id"] = ids[0]
        labels = {
            item.id: f"{item.title}\n{item.updated_at[:16]}"
            for item in conversations
        }
        selected_id = st.radio(
            "Диалоги",
            ids,
            format_func=lambda value: labels[value],
            key="chat-conversation-id",
        )
        conversation = storage.require_conversation(selected_id)
        with st.form(f"rename-{selected_id}"):
            new_title = st.text_input("Название", value=conversation.title)
            rename = st.form_submit_button("Переименовать")
        if rename:
            storage.update_conversation(selected_id, title=new_title)
            st.rerun()
        _render_settings(storage, conversation)

        json_path, markdown_path = storage.write_exports(
            selected_id,
            CHAT_EXPORT_DIR,
            include_memory_history=True,
        )
        st.download_button(
            "Скачать JSON",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Скачать Markdown",
            data=markdown_path.read_bytes(),
            file_name=markdown_path.name,
            mime="text/markdown",
            width="stretch",
        )
        confirm_clear = st.checkbox(
            "Подтверждаю очистку сообщений",
            key=f"confirm-clear-messages-{selected_id}",
        )
        if st.button(
            "Очистить сообщения",
            disabled=not confirm_clear,
            key=f"clear-messages-{selected_id}",
            width="stretch",
        ):
            deleted = storage.clear_messages(selected_id)
            st.success(f"Удалено сообщений: {deleted}. Память сохранена.")
            st.rerun()
        confirm_delete = st.checkbox(
            "Подтверждаю удаление диалога",
            key=f"confirm-delete-chat-{selected_id}",
        )

        def delete_current_conversation() -> None:
            storage.delete_conversation(selected_id)
            st.session_state.pop("chat-conversation-id", None)
            st.session_state["chat-notification"] = (
                "Диалог удалён. Документы и индексы не изменены."
            )

        st.button(
            "Удалить диалог",
            disabled=not confirm_delete,
            key=f"delete-chat-{selected_id}",
            width="stretch",
            on_click=delete_current_conversation,
        )
        notification = st.session_state.pop("chat-notification", None)
        if notification:
            st.success(notification)

    conversation = storage.require_conversation(selected_id)
    messages = storage.list_messages(selected_id)
    with center:
        for message in messages:
            _render_message(storage, message, conversation.strategy)
        latest_assistant = next(
            (message for message in reversed(messages) if message.role == "assistant"),
            None,
        )
        if latest_assistant and latest_assistant.turn_id:
            if st.button(
                "Повторить последний ответ",
                key=f"retry-{latest_assistant.turn_id}",
            ):
                embedding_service = embedding_service_factory()
                service = ChatService(
                    storage,
                    candidate_searcher=SemanticCandidateSearcher(embedding_service),
                    tokenizer=embedding_service.tokenizer,
                )
                llm = OllamaProvider(
                    conversation.model_name,
                    OLLAMA_BASE_URL,
                    OLLAMA_TIMEOUT,
                )
                with st.status("Повторная генерация…") as status:
                    result = service.retry_turn(
                        latest_assistant.turn_id,
                        llm,
                        _retrieval_config(conversation.settings),
                        temperature=float(conversation.settings.get("temperature", 0.1)),
                        progress_callback=lambda stage: status.write(_STAGE_LABELS[stage]),
                    )
                    status.update(
                        label=("Готово" if result.assistant_message.status == "completed" else "Ошибка"),
                        state=("complete" if result.assistant_message.status == "completed" else "error"),
                    )
                st.rerun()

        user_text = st.chat_input("Задайте вопрос по локальным документам")
        if user_text:
            embedding_service = embedding_service_factory()
            service = ChatService(
                storage,
                candidate_searcher=SemanticCandidateSearcher(embedding_service),
                tokenizer=embedding_service.tokenizer,
            )
            llm = OllamaProvider(
                conversation.model_name,
                OLLAMA_BASE_URL,
                OLLAMA_TIMEOUT,
            )
            with st.status("Обработка сообщения…") as status:
                result = service.process_chat_message(
                    selected_id,
                    user_text,
                    llm,
                    _retrieval_config(conversation.settings),
                    temperature=float(conversation.settings.get("temperature", 0.1)),
                    progress_callback=lambda stage: status.write(_STAGE_LABELS[stage]),
                )
                status.update(
                    label=("Готово" if result.assistant_message.status == "completed" else "Ошибка"),
                    state=("complete" if result.assistant_message.status == "completed" else "error"),
                )
            st.rerun()

    with right:
        _render_memory_panel(storage, selected_id)
        summary = storage.get_summary(selected_id)
        with st.expander("Conversation summary"):
            if summary:
                st.write(summary.summary)
                st.caption(
                    "Суммировано до сообщения: "
                    f"{summary.summarized_until_message_id}"
                )
            else:
                st.caption("Старая часть диалога ещё не сворачивалась.")
