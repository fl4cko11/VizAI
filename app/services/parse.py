from pathlib import Path

import chardet
import numpy as np
import pandas as pd

from app.schemas.agent_state import DatasetProfile, _ColumnProfile, _ColumnType


def _detect_encoding_safe(file_path: str) -> str:
    """Быстрое определение кодировки по первым байтам."""
    try:
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read(32000))  # Читаем чуть больше для надежности
        return result["encoding"] or "utf-8"
    except Exception:
        return "utf-8"


def _infer_column_type(
    series: pd.Series, unique_count: int, row_count: int
) -> _ColumnType:
    """
    Логика определения логического типа колонки.
    Приоритет: Boolean -> Datetime -> Numeric -> Categorical -> Text
    """
    # 1. Проверка на Boolean (часто приходит как object со значениями True/False или 0/1)
    if series.dtype == "bool":
        return _ColumnType.BOOLEAN

    # Если объект, проверяем уникальные значения на булевы строки
    if series.dtype == "object":
        unique_vals = set(series.dropna().astype(str).str.lower().unique())
        if (
            unique_vals <= {"true", "false", "0", "1", "yes", "no"}
            and len(unique_vals) <= 2
        ):
            return _ColumnType.BOOLEAN

    # 2. Проверка на Numeric (исключаем bool, который в pandas иногда подкласс int)
    if pd.api.types.is_numeric_dtype(series):
        if series.dtype == "bool":
            return _ColumnType.BOOLEAN
        return _ColumnType.NUMERIC

    # 3. Проверка на Datetime (пытаемся парсить, если это объект)
    if series.dtype == "object":
        # Берем сэмпл для скорости, если серия огромная
        sample = (
            series.dropna().sample(min(100, len(series)))
            if len(series) > 100
            else series.dropna()
        )
        if len(sample) > 0:
            try:
                # infer_datetime_format=True ускоряет процесс
                pd.to_datetime(sample, errors="raise")
                return _ColumnType.DATETIME
            except (ValueError, TypeError):
                pass

    # 4. Разделение на Categorical и Text
    # Эвристика: если уникальных значений мало относительно общего числа строк -> Категория
    cardinality_ratio = unique_count / row_count if row_count > 0 else 1

    if unique_count < 50 or cardinality_ratio < 0.05:
        return _ColumnType.CATEGORICAL

    return _ColumnType.TEXT


def generate_dataset_profile(
    file_path: str, file_id: str | None = None
) -> DatasetProfile:
    """
    Генерирует полный профиль датасета для передачи в LLM.

    Args:
        file_path: Путь к CSV файлу.
        file_id: Уникальный ID файла (если нет, генерируется из пути).

    Returns:
        DatasetProfile: Объект с метаданными и статистикой.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if file_id is None:
        file_id = path_obj.stem  # Используем имя файла без расширения как ID

    # 1. Определение кодировки и разделителя
    encoding = _detect_encoding_safe(str(path_obj))

    # Читаем файл с оптимизациями
    try:
        df = pd.read_csv(
            path_obj,
            encoding=encoding,
            on_bad_lines="warn",  # Пропускать битые строки
            low_memory=False,  # Лучше прочитать сразу, чтобы избежать смешанных типов
            dtype_backend="numpy_nullable",  # Современный бест пракис
        )
    except Exception:
        # Пробуем fallback на latin-1 если utf-8 не сработал корректно
        try:
            df = pd.read_csv(path_obj, encoding="latin-1", on_bad_lines="warn")
        except Exception as e2:
            raise ValueError(f"Не удалось прочитать CSV: {e2}")

    row_count, col_count = df.shape
    mem_size = df.memory_usage(deep=True).sum() / (1024 * 1024)

    columns_profiles: list[_ColumnProfile] = []
    potential_time_series: str | None = None
    potential_target: str | None = None

    # Переменные для эвристик выбора целевой переменной
    max_numeric_range = -np.inf

    for col in df.columns:
        series = df[col]
        unique_count = series.nunique()
        null_count = int(series.isnull().sum())
        null_percent = (
            round((null_count / row_count) * 100, 2) if row_count > 0 else 0.0
        )

        # Получаем 3 случайных непустых значения для примера (преобразуем в строку)
        non_null_sample = series.dropna().sample(min(3, len(series)), random_state=42)
        sample_values = [str(val) for val in non_null_sample.tolist()]

        # Определяем логический тип
        inferred_type = _infer_column_type(series, unique_count, row_count)

        # Инициализация полей статистики
        min_val, max_val, mean_val = None, None, None
        top_categories = None

        # Сбор специфичной статистики
        if inferred_type == _ColumnType.NUMERIC:
            min_val = float(series.min()) if not pd.isna(series.min()) else None
            max_val = float(series.max()) if not pd.isna(series.max()) else None
            mean_val = float(series.mean()) if not pd.isna(series.mean()) else None

            # Эвристика для potential_target: ищем числовую колонку с большим разбросом
            if min_val is not None and max_val is not None:
                range_val = max_val - min_val
                if range_val > max_numeric_range and unique_count > 10:
                    max_numeric_range = range_val
                    potential_target = col

        elif inferred_type == _ColumnType.DATETIME:
            # Для дат можно тоже посчитать мин/макс, преобразовав временно
            try:
                dt_series = pd.to_datetime(series, errors="coerce")
                min_val = (
                    dt_series.min().timestamp()
                    if not pd.isna(dt_series.min())
                    else None
                )
                max_val = (
                    dt_series.max().timestamp()
                    if not pd.isna(dt_series.max())
                    else None
                )
                # Первая найденная дата становится кандидатом на time series
                if potential_time_series is None:
                    potential_time_series = col
            except Exception:
                pass

        elif inferred_type == _ColumnType.CATEGORICAL:
            # Берем топ-5 самых частых категорий
            counts = series.value_counts(dropna=True).head(5)
            top_categories = [str(cat) for cat in counts.index.tolist()]

        # Формируем профиль колонки
        col_profile = _ColumnProfile(
            name=str(col),
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            unique_count=int(unique_count),
            top_categories=top_categories,
            null_count=null_count,
            null_percent=null_percent,
            sample_values=sample_values,
        )
        columns_profiles.append(col_profile)

    return DatasetProfile(
        file_id=file_id,
        file_path=str(path_obj.absolute()),
        row_count=row_count,
        col_count=col_count,
        memory_size_mb=round(mem_size, 2),
        columns=columns_profiles,
        potential_time_series=potential_time_series,
        potential_target=potential_target,
    )
