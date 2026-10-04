"""Абстракция LLM и локальный провайдер Ollama."""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Минимальный интерфейс генератора ответов."""

    model_name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        """Сгенерировать ответ по сообщениям."""


class LLMError(RuntimeError):
    """Понятная прикладная ошибка локальной LLM."""


class LLMConnectionError(LLMError):
    """Сервер Ollama недоступен."""


class LLMModelNotFoundError(LLMError):
    """Выбранная модель отсутствует в Ollama."""


class LLMTimeoutError(LLMError):
    """Ollama не ответила за заданное время."""


class LLMEmptyResponseError(LLMError):
    """Ollama вернула пустой ответ."""


class OllamaProvider:
    """Локальный LLM-провайдер через HTTP API Ollama."""

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Название модели Ollama не может быть пустым.")
        if timeout <= 0:
            raise ValueError("Тайм-аут Ollama должен быть больше нуля.")
        parsed_url = urllib.parse.urlparse(base_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Ollama должна быть доступна только по локальному HTTP-адресу.")
        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _request_json(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            if exc.code == 404 or "not found" in detail.lower():
                raise LLMModelNotFoundError(
                    f"Модель Ollama '{self.model_name}' не установлена. "
                    f"Выполните: ollama pull {self.model_name}"
                ) from exc
            raise LLMConnectionError(f"Ollama вернула HTTP {exc.code}.") from exc
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError(
                    f"Ollama не ответила за {self.timeout:g} с."
                ) from exc
            raise LLMConnectionError(
                "Ollama недоступна. Запустите локальный сервер командой `ollama serve`."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeoutError(f"Ollama не ответила за {self.timeout:g} с.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMError("Ollama вернула некорректный JSON.") from exc
        if not isinstance(result, dict):
            raise LLMError("Ollama вернула неожиданный формат ответа.")
        return result

    def check_connection(self) -> tuple[bool, str]:
        """Проверить сервер Ollama и наличие выбранной модели."""
        try:
            response = self._request_json("/api/tags")
        except LLMError as exc:
            return False, str(exc)
        models = response.get("models", [])
        names = {
            str(item.get("name") or item.get("model"))
            for item in models
            if isinstance(item, dict) and (item.get("name") or item.get("model"))
        }
        if self.model_name not in names:
            return (
                False,
                f"Модель '{self.model_name}' не установлена. "
                f"Выполните: ollama pull {self.model_name}",
            )
        return True, f"Ollama доступна, модель '{self.model_name}' установлена."

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        """Получить нестриминговый ответ локальной Ollama."""
        if not messages or any(not item.get("content", "").strip() for item in messages):
            raise ValueError("Сообщения для LLM не могут быть пустыми.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature должна находиться между 0 и 2.")
        logger.info("Запрос к Ollama: model=%s messages=%d", self.model_name, len(messages))
        response = self._request_json(
            "/api/chat",
            payload={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        message = response.get("message")
        answer = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(answer, str) or not answer.strip():
            raise LLMEmptyResponseError("Ollama вернула пустой ответ.")
        return answer.strip()
