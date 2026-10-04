"""Демонстрационный модуль каталога векторов.

Файл намеренно содержит функции, классы, методы и подробные docstring,
чтобы структурный chunking мог показать работу Python AST.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


def normalize_vector(values: Iterable[float]) -> list[float]:
    """Нормализовать числовой вектор до единичной длины.

    Нулевой вектор возвращается без изменения. Для ненулевого вектора
    вычисляется евклидова норма, после чего каждый компонент делится на неё.
    Нормализация позволяет использовать внутреннее произведение как
    косинусную близость и одинаково трактовать векторы разного масштаба.
    """
    vector = [float(value) for value in values]
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@dataclass(frozen=True)
class Record:
    """Одна подписанная запись каталога."""
    key: str
    vector: tuple[float, ...]
    text: str


class VectorCatalog:
    """Небольшой пример локального векторного каталога.

    Класс не используется приложением и служит демонстрационным документом.
    Он показывает проверку размерности, пакетное добавление и поиск по
    косинусной близости без внешних сервисов.
    """

    def __init__(self, dimension: int) -> None:
        """Создать пустой каталог заданной положительной размерности."""
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._records: list[Record] = []

    def add(self, key: str, vector: Iterable[float], text: str) -> None:
        """Добавить запись после нормализации и проверки размерности."""
        normalized = normalize_vector(vector)
        if len(normalized) != self.dimension:
            raise ValueError("vector dimension mismatch")
        self._records.append(Record(key, tuple(normalized), text))

    def add_many(self, rows: Iterable[tuple[str, Iterable[float], str]]) -> int:
        """Пакетно добавить записи и вернуть новое количество элементов.

        Метод последовательно вызывает add, поэтому для каждой строки
        применяются те же проверки. В производственном хранилище пакет
        следует сначала полностью проверить, а затем фиксировать транзакцией.
        """
        for key, vector, text in rows:
            self.add(key, vector, text)
        return len(self._records)

    def search(self, query: Iterable[float], limit: int = 5) -> list[tuple[float, Record]]:
        """Найти записи по косинусной близости.

        Нормализованный запрос умножается на каждый нормализованный вектор.
        Результаты сортируются по убыванию оценки. Метод демонстрирует ту же
        математику, которую FAISS IndexFlatIP выполняет эффективнее.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        normalized = normalize_vector(query)
        if len(normalized) != self.dimension:
            raise ValueError("query dimension mismatch")
        scored = [
            (sum(left * right for left, right in zip(normalized, record.vector)), record)
            for record in self._records
        ]
        return sorted(scored, key=lambda item: item[0], reverse=True)[:limit]

    def remove(self, key: str) -> bool:
        """Удалить запись по ключу и сообщить, была ли она найдена."""
        for index, record in enumerate(self._records):
            if record.key == key:
                del self._records[index]
                return True
        return False

    def __len__(self) -> int:
        """Вернуть количество записей."""
        return len(self._records)


def transform_batch_1(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 1 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_2(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 2 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_3(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 3 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_4(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 4 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_5(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 5 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_6(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 6 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_7(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 7 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_8(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 8 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_9(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 9 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_10(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 10 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_11(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 11 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_12(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 12 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_13(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 13 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_14(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 14 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_15(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 15 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_16(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 16 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_17(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 17 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_18(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 18 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_19(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 19 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_20(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 20 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_21(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 21 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_22(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 22 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_23(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 23 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_24(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 24 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_25(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 25 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_26(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 26 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_27(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 27 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_28(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 28 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_29(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 29 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_30(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 30 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_31(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 31 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_32(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 32 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_33(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 33 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_34(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 34 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_35(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 35 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_36(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 36 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_37(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 37 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_38(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 38 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_39(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 39 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]


def transform_batch_40(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    """Нормализовать пакет номер 40 для демонстрации структурного разбиения.

    Функция материализует каждый входной iterable ровно один раз, применяет
    normalize_vector и сохраняет исходный порядок. Отдельное имя делает
    секцию заметной в таблице чанков и позволяет проверить поиск по коду.
    """
    return [normalize_vector(row) for row in rows]
