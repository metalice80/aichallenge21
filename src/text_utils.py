"""Нормализация текста и детерминированные идентификаторы."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Смещение одного токена в исходной строке."""

    start: int
    end: int


class OffsetTokenizer(Protocol):
    """Минимальный интерфейс токенизатора для chunking."""

    def token_spans(self, text: str) -> list[TokenSpan]:
        """Вернуть символьные границы токенов."""


class RegexTokenizer:
    """Детерминированный локальный токенизатор для тестов и fallback."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def token_spans(self, text: str) -> list[TokenSpan]:
        """Считать слова и знаки препинания отдельными токенами."""
        return [TokenSpan(match.start(), match.end()) for match in self._pattern.finditer(text)]


def stable_id(*parts: object) -> str:
    """Построить SHA-256 идентификатор из стабильных частей."""
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_hash(data: bytes) -> str:
    """Вернуть SHA-256 содержимого файла."""
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str, *, preserve_source: bool = False) -> str:
    """Нормализовать переносы и пустые строки, не разрушая абзацы."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_source:
        return text
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{4,}", "\n\n\n", normalized).strip()


def portable_source(path: Path) -> str:
    """Вернуть имя источника без абсолютного пути пользователя."""
    return path.name


def count_words(text: str) -> int:
    """Приблизительно посчитать слова Unicode."""
    return len(re.findall(r"\w+", text, re.UNICODE))


def estimate_pages(text: str, words_per_page: int = 450) -> int:
    """Оценить число страниц по количеству слов."""
    words = count_words(text)
    return max(1, (words + words_per_page - 1) // words_per_page) if text.strip() else 0
