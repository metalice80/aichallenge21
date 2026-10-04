from __future__ import annotations

import pytest

from src.llm import LLMEmptyResponseError, OllamaProvider


def test_connection_checks_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("qwen:test", "http://localhost:11434")
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda path, **kwargs: {"models": [{"name": "qwen:test"}]},
    )

    available, message = provider.check_connection()

    assert available is True
    assert "установлена" in message


def test_connection_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("missing:model", "http://localhost:11434")
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda path, **kwargs: {"models": [{"name": "other:model"}]},
    )

    available, message = provider.check_connection()

    assert available is False
    assert "ollama pull missing:model" in message


def test_generate_uses_model_messages_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("qwen:test", "http://localhost:11434")
    captured: dict[str, object] = {}

    def request(path: str, **kwargs):
        captured.update({"path": path, **kwargs})
        return {"message": {"content": " Готовый ответ "}}

    monkeypatch.setattr(provider, "_request_json", request)
    answer = provider.generate([{"role": "user", "content": "Вопрос"}], 0.3)

    assert answer == "Готовый ответ"
    assert captured["path"] == "/api/chat"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen:test"
    assert payload["options"]["temperature"] == 0.3


def test_generate_rejects_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("qwen:test", "http://localhost:11434")
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda path, **kwargs: {"message": {"content": " "}},
    )

    with pytest.raises(LLMEmptyResponseError, match="пустой"):
        provider.generate([{"role": "user", "content": "Вопрос"}])


def test_ollama_rejects_non_local_server() -> None:
    with pytest.raises(ValueError, match="локальному"):
        OllamaProvider("qwen:test", "https://paid.example.com")
