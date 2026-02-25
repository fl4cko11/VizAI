from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class _ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"  # Длинный текст, не подходит для осей графиков
    BOOLEAN = "boolean"


class _ColumnProfile(BaseModel):
    """
    Сжатое описание колонки, достаточное для выбора типа графика.
    Не содержит самих данных, только метаданные.
    """

    name: str = Field(..., description="Имя колонки (нормализованное, без пробелов)")
    dtype: str = Field(
        ..., description="Оригинальный тип данных pandas (напр. 'int64', 'object')"
    )
    inferred_type: _ColumnType = Field(
        ..., description="Логический тип данных для визуализации"
    )

    # Статистика для чисел
    min_val: float | None = Field(
        None, description="Минимальное значение (для numeric/datetime)"
    )
    max_val: float | None = Field(
        None, description="Максимальное значение (для numeric/datetime)"
    )
    mean_val: float | None = Field(None, description="Среднее значение (для numeric)")

    # Статистика для категорий
    unique_count: int = Field(..., description="Количество уникальных значений")
    top_categories: list[str] | None = Field(
        None, description="Топ-5 самых частых категорий (для categorical)"
    )

    # Качество данных
    null_count: int = Field(..., description="Количество пропусков (NaN)")
    null_percent: float = Field(..., description="Процент пропусков (0.0 - 100.0)")

    # Примеры данных (для понимания формата LLM)
    sample_values: list[str] = Field(
        ..., description="3 случайных значения из колонки для контекста"
    )

    class Config:
        use_enum_values = True  # Сериализовать enum как строку для LLM


class DatasetProfile(BaseModel):
    """
    Агрегированная информация о файле.
    Это 'глаза' агента.
    """

    file_id: str = Field(..., description="Уникальный ID файла во временном хранилище")
    file_path: str = Field(..., description="Полный путь к файлу на диске для Tools")
    row_count: int = Field(..., description="Общее количество строк")
    col_count: int = Field(..., description="Общее количество колонок")
    memory_size_mb: float = Field(
        ..., description="Приблизительный размер в памяти (MB)"
    )

    columns: list[_ColumnProfile] = Field(
        ..., description="Список профилей всех колонок"
    )

    # Автоматические гипотезы (опционально, можно генерировать заранее или поручить LLM)
    potential_time_series: str | None = Field(
        None, description="Имя колонки, которая выглядит как время/дата"
    )
    potential_target: str | None = Field(
        None, description="Имя колонки, которая может быть целевой переменной (число)"
    )


class GeneratedArtifact(BaseModel):
    """
    Метаданные сгенерированного файла для отправки пользователю.
    """

    file_path: str = Field(..., description="Полный путь к файлу на сервере")
    file_name: str = Field(
        ...,
        description="Имя файла, которое увидит пользователь (напр. dashboard_01.html)",
    )
    mime_type: str = Field(default="text/html", description="MIME тип для Telegram")


class AgentState(BaseModel):
    user_query: str = Field(..., description="Исходный запрос пользователя")

    # Аннотация user_data как списка датасетов
    user_data: list[DatasetProfile] = Field(
        default_factory=list,
        description="Список загруженных пользователем CSV файлов и результатов их парсинга",
    )

    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="История диалога в формате [{'role': 'user', 'content': '...'}, ...]",
    )

    artifacts: list[GeneratedArtifact] = Field(
        default_factory=list,
        description="Список сгенерированных файлов (HTML/PNG), готовых к отправке в Telegram",
    )

    used_methods: list[dict[str, Any]] = Field(
        default=None,
        description="Список использованных методов и параметры вызова в формате [{'name': 'func_name1', 'args': '...'}, ...]",
    )

    answer: str | None = Field(default=None, description="Ответ LLM")

    errors: list[str] = Field(
        default_factory=list, description="Ошибки полученные в текущем цикле"
    )

    iteration: int = Field(
        0, ge=0, description="Текущая итерация цикла retrieval → rewrite"
    )
