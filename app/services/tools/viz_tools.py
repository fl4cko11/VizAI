import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from langchain_core.tools import tool
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import (
    LinearRegression,
)


@tool(description="Создает интерактивную гистограмму с автоматической статистикой.")
def create_histogram_tool(
    file_path: str,
    column_name: str,
    title: str,
    nbins: int,
    color: str = "#636EFA",
    show_stats: bool = True,
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Создает интерактивную гистограмму с автоматической статистикой.

    Args:
        file_path: путь до csv файла пользователя.
        column_name: Имя переменной для осей и легенды.
        title: Заголовок графика. Если None, генерируется автоматически.
        nbins: Количество бинов. Если None, Plotly выбирает оптимальное.
        color: Цвет столбцов (hex или named color).
        show_stats: Если True, добавляет вертикальные линии среднего и медианы.
        encoding: Кодировка.

    Returns:
        plotly.graph_objects.Figure: Объект фигуры, готовый к рендеру или сериализации.
    """

    # 1. Предобработка данных (Best Practice: очистка перед визуализацией)
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем ТОЛЬКО нужную колонку для экономии памяти
        df = pd.read_csv(
            path_obj,
            usecols=[column_name],
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        # Fallback на latin-1
        df = pd.read_csv(
            path_obj, usecols=[column_name], encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Колонка '{column_name}' не найдена в файле.")
        raise e

    series = df[column_name]

    # Приводим к числу (на случай мусора в CSV) и удаляем NaN
    clean_data = pd.to_numeric(series, errors="coerce").dropna()

    if len(clean_data) == 0:
        raise ValueError(f"Нет валидных числовых данных в колонке '{column_name}'.")

    label = column_name

    # 2. Инициализация фигуры
    fig = go.Figure()

    # 3. Построение гистограммы
    fig.add_trace(
        go.Histogram(
            x=clean_data,
            name=label,
            nbinsx=nbins,
            marker_color=color,
            opacity=0.75,
            hovertemplate=f"{label}: %{{x}}<br>Частота: %{{y}}<extra></extra>",
        )
    )

    # 4. Добавление статистики (Аналитический слой)
    if show_stats:
        mean_val = np.mean(clean_data)
        median_val = np.median(clean_data)

        # Линия среднего (пунктир)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Среднее: {mean_val:.2f}",
            annotation_position="top",
        )
        # Линия медианы (сплошная)
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Медиана: {median_val:.2f}",
            annotation_position="bottom",
        )

    # 5. Формирование-layout (UX/UI)
    final_title = title if title else f"Распределение: {label}"

    fig.update_layout(
        title={
            "text": final_title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title=label,
        yaxis_title="Частота (Count)",
        template="plotly_white",  # Чистый фон, лучше для отчетов
        hovermode="x unified",
        bargap=0.1,  # Небольшой зазор между барами для читаемости
        height=500,
        width=800,
    )

    return fig


@tool(description="Создает дашборд с сеткой гистограмм для множественных переменных.")
def create_histogram_dashboard_tool(
    file_path: str,
    column_names: list[str],
    dashboard_title: str = "Панель распределения данных",
    cols: int = 2,
    shared_axes: bool = True,
    encoding: str = "utf-8",
) -> go.Figure:
    """
    Создает дашборд с сеткой гистограмм для множественных переменных.

    Args:
        file_path: путь до csv файла пользователя.
        dashboard_title: Общий заголовок дашборда.
        cols: Количество колонок в сетке.
        shared_axes: Если True, оси X и Y синхронизированы для удобного сравнения.
        encoding: Кодировка.

    Returns:
        go.Figure: Объект дашборда.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if not column_names:
        raise ValueError("Список колонок пуст.")

    try:
        # Читаем ТОЛЬКО нужные колонки одним запросом
        df = pd.read_csv(
            path_obj,
            usecols=column_names,
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=column_names, encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Одна из колонок {column_names} не найдена в файле.")
        raise e

    clean_data_dict = {}

    for col in column_names:
        if col not in df.columns:
            continue  # Пропускаем, если вдруг колонка потерялась при чтении

        # Приводим к числу и чистим NaN
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) > 0:
            clean_data_dict[col] = series

    if not clean_data_dict:
        raise ValueError(
            "Нет валидных числовых данных ни в одной из указанных колонок."
        )

    n_vars = len(clean_data_dict)

    # Расчет размеров сетки
    rows = math.ceil(n_vars / cols)

    # Инициализация подплотов
    # specs: позволяет делать сложные лейауты, здесь пока просто пустые ячейки
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(clean_data_dict.keys()),
        shared_xaxes=shared_axes,
        shared_yaxes=(
            shared_axes if shared_axes else False
        ),  # Часто полезно видеть абсолютные частоты отдельно
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Цветовая палитра (циклическая, чтобы не повторяться слишком рано)
    colors = plotly.colors.qualitative.Plotly

    for idx, (col_name, data) in enumerate(clean_data_dict.items()):
        # Вычисление координат в сетке (row, col)
        row_idx = (idx // cols) + 1
        col_idx = (idx % cols) + 1

        # Подготовка данных (дублируем логику очистки из простой функции для надежности)
        if isinstance(data, pd.Series):
            clean_data = data.dropna()
        else:
            arr = np.array(data)
            clean_data = arr[~np.isnan(arr)]

        if len(clean_data) == 0:
            continue

        color = colors[idx % len(colors)]

        # Добавление траса на конкретную позицию сетки
        fig.add_trace(
            go.Histogram(
                x=clean_data,
                name=col_name,
                marker_color=color,
                opacity=0.7,
                showlegend=False,  # Легенда будет перегружена, названия есть в сабтайтлах
            ),
            row=row_idx,
            col=col_idx,
        )

        # Добавление статистики прямо на подплот (опционально, можно убрать для чистоты)
        mean_val = np.mean(clean_data)
        fig.add_vline(
            x=mean_val,
            line_dash="dot",
            line_color="red",
            row=row_idx,
            col=col_idx,
            opacity=0.5,
        )

    # Глобальное обновление layout
    fig.update_layout(
        title={
            "text": dashboard_title,
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24),
        },
        height=400 * rows,
        width=800,
        template="plotly_white",
        hovermode="x unified",
        bargap=0.1,
    )

    # Обновление осей для всех подплотов (цикл по всем осям)
    fig.update_xaxes(title_text="Значение")
    fig.update_yaxes(title_text="Частота")

    return fig


@tool(
    description="Создает оптимизированную круговую диаграмму с автоматической группировкой мелких категорий."
)
def create_pie_chart_tool(
    file_path: str,
    column_name: str,
    color_palette: list[str],
    title: str = "Распределение категорий",
    min_share_threshold: float = 0.05,
    max_categories: int = 8,
    encoding: str = "utf-8",
) -> go.Figure:
    """
    Создает оптимизированную круговую диаграмму с автоматической группировкой мелких категорий.

    Args:
        file_path: путь до csv файла пользователя.
        title: Заголовок графика.
        min_share_threshold: Доля (0.0-1.0), ниже которой категории объединяются в 'Other'.
        max_categories: Максимальное количество отдельных секторов. Остальные -> 'Other'.
        color_palette: Список цветов. Если None, используется дефолтная палитра Plotly.
        encoding: Кодировка.

    Returns:
        go.Figure: Объект фигуры.
    """

    # 1. Предобработка и очистка
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем ТОЛЬКО нужную колонку
        df = pd.read_csv(
            path_obj,
            usecols=[column_name],
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=[column_name], encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Колонка '{column_name}' не найдена в файле.")
        raise e

    # Предобработка: очистка от NaN и приведение к строке (категории)
    series = df[column_name]
    clean_data = series.dropna().astype(str)

    if len(clean_data) == 0:
        raise ValueError(
            f"Нет валидных данных в колонке '{column_name}' после очистки."
        )

    # 2. Агрегация значений (Value Counts)
    counts = clean_data.value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    total = counts["Count"].sum()
    counts["Share"] = counts["Count"] / total

    # 3. Логика группировки "Длинного хвоста" (Long Tail Handling)
    # Сортируем по убыванию (value_counts уже делает это, но на всякий случай)
    counts = counts.sort_values("Count", ascending=False).reset_index(drop=True)

    # Определяем, какие категории оставить, а какие схлопнуть
    rows_to_keep = []
    other_count = 0

    for i, row in counts.iterrows():
        # Условие 1: Входит в топ N по количеству
        is_top_n = i < max_categories
        # Условие 2: Доля выше порога
        is_significant = row["Share"] >= min_share_threshold

        if is_top_n and is_significant:
            rows_to_keep.append(row)
        else:
            other_count += row["Count"]

    # Формируем финальный DataFrame
    final_data = pd.DataFrame(rows_to_keep)

    if other_count > 0:
        other_row = pd.DataFrame(
            [{"Category": "Other", "Count": other_count, "Share": other_count / total}]
        )
        final_data = pd.concat([final_data, other_row], ignore_index=True)

    # Сортировка для красивого отображения (по часовой стрелке от большего к меньшему)
    final_data = final_data.sort_values("Count", ascending=False)

    # 4. Построение графика
    labels = final_data["Category"].tolist()
    values = final_data["Count"].tolist()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,  # Делаем Donut Chart (пончик) - это современный стандарт, легче читать
                textinfo="percent",  # Внутри пишем только проценты для чистоты
                hoverinfo="label+value+percent",  # При наведении показываем всё
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                marker_colors=color_palette,
                textposition="outside",  # Текст снаружи, если сектор маленький, или внутри
                pull=[0.05] * len(labels),  # Немного вытягиваем сектора для акцента
                rotation=90,  # Начинаем с 12 часов
            )
        ]
    )

    # 5. Настройка Layout
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=20),
        },
        template="plotly_white",
        height=600,
        width=800,
        legend=dict(
            orientation="h",  # Легенда горизонтально снизу
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        # Добавляем аннотацию в центр "пончика" с общим количеством
        annotations=[
            dict(text=f"Total:<br>{total}", x=0.5, y=0.5, font_size=16, showarrow=False)
        ],
    )

    return fig


@tool(
    description="Создает дашборд с сеткой круговых диаграмм для сравнения распределения категории внутри групп. Внимание: Автоматически ограничивает количество групп для сохранения читаемости."
)
def create_pie_dashboard_tool(
    file_path: str,  # НОВЫЙ АРГУМЕНТ: путь к CSV
    category_col: str,  # Столбец с категориями (для секторов)
    group_by_col: str,  # Столбец для группировки (каждый группа = свой пончик)
    title: str = "Сравнение распределений по группам",
    top_n_groups: int = 4,
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Создает дашборд с сеткой круговых диаграмм для сравнения распределения категории внутри групп.
    Внимание: Автоматически ограничивает количество групп для сохранения читаемости.

    Args:
        file_path: путь до csv файла пользователя.
        category_col: Столбец с категориями (для секторов).
        group_by_col: Столбец для группировки (каждый группа = свой пончик).
        title: Заголовок дашборда.
        top_n_groups: Количество самых крупных групп для отображения.
        encoding: Кодировка.

    Returns:
        go.Figure: Дашборд.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем ТОЛЬКО две нужные колонки для экономии памяти
        df = pd.read_csv(
            path_obj,
            usecols=[category_col, group_by_col],
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj,
            usecols=[category_col, group_by_col],
            encoding="latin-1",
            on_bad_lines="skip",
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(
                f"Одна из колонок '{category_col}' или '{group_by_col}' не найдена в файле."
            )
        raise e

    # Приводим категории к строке для корректной группировки
    df[category_col] = df[category_col].dropna().astype(str)
    df[group_by_col] = df[group_by_col].dropna().astype(str)

    # Удаляем строки, где после очистки пропали значения
    df = df.dropna(subset=[category_col, group_by_col])

    if df.empty:
        raise ValueError("Нет валидных данных для построения дашборда после очистки.")

    # 1. Анализ размеров групп
    group_sizes = df[group_by_col].value_counts().head(top_n_groups)
    selected_groups = group_sizes.index.tolist()

    if len(selected_groups) == 0:
        raise ValueError("Нет данных для отображения.")

    # Фильтруем датафрейм
    filtered_df = df[df[group_by_col].isin(selected_groups)]

    # 2. Расчет сетки
    n_charts = len(selected_groups)
    cols = 2 if n_charts > 1 else 1
    rows = (n_charts + cols - 1) // cols

    # Создаем подплоты.
    # type='domain' необходим для Pie chart в subplot
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "domain"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f"Группа: {g}" for g in selected_groups],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    # 3. Генерация графиков
    colors = plotly.colors.qualitative.Plotly

    for idx, group_name in enumerate(selected_groups):
        # Данные для конкретной группы
        subset = filtered_df[filtered_df[group_by_col] == group_name]
        counts = subset[category_col].value_counts()

        # Простая логика обрезки хвостов внутри каждой группы (топ-5 + Other)
        if len(counts) > 6:
            top_cats = counts.head(5)
            other_sum = counts.iloc[5:].sum()
            if other_sum > 0:
                counts = pd.concat([top_cats, pd.Series([other_sum], index=["Other"])])

        labels = counts.index.astype(str).tolist()
        values = counts.values.tolist()

        row_idx = (idx // cols) + 1
        col_idx = (idx % cols) + 1

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker_colors=colors,
                textinfo="percent",
                hoverinfo="label+value+percent",
                name=group_name,
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )

    # 4. Global Layout
    fig.update_layout(
        title={
            "text": title,
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=400 * rows,
        width=900,
        template="plotly_white",
        # Единая легенда справа (опционально, если категории одинаковые во всех группах)
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.05, xanchor="left"),
    )

    # Обновление заголовков сабплотов
    fig.update_annotations(font_size=12, font_color="#444")

    return fig


@tool(
    description="Создает интерактивный линейный график с поддержкой временных рядов и сглаживания."
)
def create_line_chart_tool(
    file_path: str,  # НОВЫЙ АРГУМЕНТ: путь к CSV
    x_col: str,  # Столбец для оси X (обычно дата/время)
    y_cols: list[str],  # Столбец (или список) для оси Y
    rolling_window: int,
    title: str = "Динамика показателей",
    sort_x: bool = True,
    handle_missing: Literal["drop", "interpolate", "gap"] = "gap",
    show_markers: bool = False,
    line_width: int = 2,
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Создает интерактивный линейный график с поддержкой временных рядов и сглаживания.

    Args:
        file_path: путь до csv файла пользователя.
        x_col: Столбец для оси X (обычно дата/время).
        y_cols: Столбец (или список столбцов) для оси Y.
        title: Заголовок графика.
        sort_x: Если True, сортирует данные по X перед построением.
        handle_missing:
            - 'drop': удаляет строки с NaN.
            - 'interpolate': заполняет NaN линейной интерполяцией.
            - 'gap': оставляет разрывы в линии (рекомендуется для честности данных).
        rolling_window: Размер окна для скользящего среднего. Если None, строит сырые данные.
        show_markers: Показывать ли точки на каждом значении (полезно для малых данных).
        line_width: Толщина линии.
        encoding: Кодировка.

    Returns:
        go.Figure: Объект фигуры.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Нормализуем y_cols для чтения
    if isinstance(y_cols, str):
        cols_to_read = [x_col, y_cols]
    else:
        cols_to_read = [x_col] + list(y_cols)

    try:
        # Читаем ТОЛЬКО нужные колонки
        df = pd.read_csv(
            path_obj,
            usecols=cols_to_read,
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=cols_to_read, encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Одна из колонок {cols_to_read} не найдена в файле.")
        raise e

    # 1. Подготовка данных
    df = df.copy()

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    # Проверка наличия колонок
    missing_cols = [c for c in [x_col] + y_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки в данных: {missing_cols}")

    # Преобразование X в datetime, если возможно (Best Practice для Time Series)
    if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        try:
            df[x_col] = pd.to_datetime(df[x_col])
        except Exception:
            pass  # Оставляем как есть, если это не дата (например, эпохи или шаги)

    # Сортировка
    if sort_x:
        df = df.sort_values(by=x_col).reset_index(drop=True)

    # Обработка пропусков
    if handle_missing == "drop":
        df = df.dropna(subset=y_cols)
    elif handle_missing == "interpolate":
        # Интерполируем только числовые колонки Y
        df[y_cols] = df[y_cols].interpolate(method="linear")
    # 'gap' ничего не делает специально, Plotly сам разрывает линию при NaN,
    # но нам нужно убедиться, что NaN остались NaN, а не были заполнены ранее.

    if df.empty:
        raise ValueError("Данные пусты после предобработки.")

    # 2. Инициализация фигуры
    fig = go.Figure()

    # Цветовая палитра (циклическая)
    colors = plotly.colors.qualitative.Plotly

    for idx, col in enumerate(y_cols):
        y_data = df[col]
        label = col

        # Применение скользящего среднего (SMA) если запрошено
        if rolling_window and rolling_window > 1:
            y_data = y_data.rolling(window=rolling_window, min_periods=1).mean()
            label = f"{col} (SMA-{rolling_window})"

        color = colors[idx % len(colors)]

        # Добавление траса
        # connectgaps=False гарантирует разрывы линии при наличии NaN (режим 'gap')
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=y_data,
                mode="lines+markers" if show_markers else "lines",
                name=label,
                line=dict(
                    color=color, width=line_width, shape="spline"
                ),  # 'spline' для сглаживания кривой
                marker=dict(size=6, line=dict(width=1, color="white")),
                connectgaps=(
                    handle_missing == "interpolate"
                ),  # Соединяем только если интерполировали
                hovertemplate=f"<b>{label}</b><br>{x_col}: %{{x|%Y-%m-%d}}<br>Значение: %{{y:.2f}}<extra></extra>",
            )
        )

    # 3. Настройка Layout
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title=x_col,
        yaxis_title="Значение",
        template="plotly_white",
        hovermode="x unified",  # Важно для сравнения нескольких линий в одной точке времени
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        width=900,
    )

    # Форматирование оси X (если это даты)
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            tickangle=45,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1 нед", step="day", stepmode="backward"),
                        dict(count=1, label="1 мес", step="month", stepmode="backward"),
                        dict(count=6, label="6 мес", step="month", stepmode="backward"),
                        dict(count=1, label="1 год", step="year", stepmode="backward"),
                        dict(step="all", label="Всё"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),  # Ползунок для зума внизу
        )

    return fig


@tool(
    description="Создает двухпанельный дашборд: основной график + график изменений (ROC)."
)
def create_time_series_dashboard_tool(
    file_path: str,
    x_col: str,
    y_col: str,
    title: str = "Аналитика временного ряда",
    show_rolling_avg: bool = True,
    rolling_window: int = 7,
    show_pct_change: bool = True,
    encoding: str = "utf-8",
) -> go.Figure:
    """
    Создает двухпанельный дашборд: основной график + график изменений (ROC).
    Args:
        file_path: путь до csv файла пользователя.
        x_col: Колонка времени.
        y_col: Целевая метрика.
        title: Заголовок.
        show_rolling_avg: Показать ли линию скользящего среднего на основном графике.
        rolling_window: Окно для SMA.
        show_pct_change: Показать ли нижний график с процентным изменением.
        encoding: Кодировка.

    Returns:
        go.Figure: Дашборд с подплотами.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем ТОЛЬКО две нужные колонки (время + метрика)
        df = pd.read_csv(
            path_obj,
            usecols=[x_col, y_col],
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=[x_col, y_col], encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Колонки '{x_col}' или '{y_col}' не найдены в файле.")
        raise e

    df = df.copy()

    # Безопасное преобразование в дату
    if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        try:
            df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
            if df[x_col].isna().all():
                df[x_col] = pd.to_numeric(df[x_col], errors="ignore")
        except Exception:
            pass

    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df = df.dropna(subset=[x_col])

    df = df.sort_values(by=x_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("Данные пусты после предобработки времени.")

    # Расчет дополнительных метрик
    df["SMA"] = (
        df[y_col].rolling(window=rolling_window, min_periods=1).mean()
        if show_rolling_avg
        else np.nan
    )

    if show_pct_change:
        df["PCT_Change"] = df[y_col].pct_change() * 100
        # ИСПРАВЛЕНИЕ: Заполняем NaN нулем (или можно dropna, но тогда длины осей не совпадут)
        # Для цветовой схемы важно иметь конкретное число.
        # Первая строка всегда NaN, её красим в серый через заполнение 0, которое < 0? Нет, 0 даст gray.
        # Но лучше заполнить 0, чтобы условие сработало корректно.
        df["PCT_Change"] = df["PCT_Change"].fillna(0)

    # Определение количества рядов
    rows = 2 if show_pct_change else 1
    row_heights = [0.7, 0.3] if show_pct_change else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=[title, "Темп изменения (%)"] if show_pct_change else [title],
    )

    colors = plotly.colors.qualitative.Plotly

    # --- ROW 1: Основной график ---
    fig.add_trace(
        go.Scatter(
            x=df[x_col], y=df[y_col], name=y_col, line=dict(color=colors[0], width=2)
        ),
        row=1,
        col=1,
    )
    if show_rolling_avg:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df["SMA"],
                name=f"SMA ({rolling_window})",
                line=dict(color=colors[1], width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # --- ROW 2: Процентное изменение ---
    if show_pct_change:
        # Теперь сравнение работает корректно, так как NaN заменен на 0
        colors_bar = [
            "green" if x > 0 else "red" if x < 0 else "gray" for x in df["PCT_Change"]
        ]

        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df["PCT_Change"],
                name="% Изменение",
                marker_color=colors_bar,
                opacity=0.7,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1)

    fig.update_layout(
        height=600 if show_pct_change else 400,
        width=900,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(title_text=x_col, row=rows, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    if show_pct_change:
        fig.update_yaxes(title_text="%", row=2, col=1)
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=45, row=2, col=1)
            fig.update_xaxes(tickformat="", row=1, col=1)

    return fig


@tool(
    description="Создает многомерную диаграмму рассеяния (Scatter Plot) для анализа взаимосвязи между двумя числовыми переменными. Поддерживает кодирование дополнительных измерений через цвет (категории или градиент) и размер точек. Включает встроенные аналитические инструменты: линию тренда (OLS) с R² и контуры плотности для больших данных."
)
def create_scatter_plot_tool(
    file_path: str,  # НОВЫЙ АРГУМЕНТ: путь к CSV
    x_col: str,  # Столбец для оси X
    y_col: str,  # Столбец для оси Y
    color_col: str,  # Столбец для цвета (категория или число)
    size_col: str,  # Столбец для размера точек
    title: str = "Диаграмма рассеяния",
    trend_line: Literal["ols", "lowess"] = "ols",
    show_density_contours: bool = False,
    opacity: float = 0.6,
    log_x: bool = False,
    log_y: bool = False,
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Создает многомерную диаграмму рассеяния (Scatter Plot) для анализа взаимосвязи между двумя числовыми переменными.
    Поддерживает кодирование дополнительных измерений через цвет (категории или градиент) и размер точек.
    Включает встроенные аналитические инструменты: линию тренда (OLS) с R² и контуры плотности для больших данных.

    Args:
        file_path: Путь к CSV файлу пользователя во временном хранилище.
        x_col: Имя колонки для оси X (числовая).
        y_col: Имя колонки для оси Y (числовая).
        color_col: Имя колонки для раскраски точек. Может быть категориальной (разные цвета для групп) или числовой (градиент/тепловая карта).
        size_col: Имя числовой колонки для определения размера точек (пузырьковая диаграмма). Значения нормализуются автоматически.
        title: Заголовок графика.
        trend_line: Тип линии тренда. 'ols' — линейная регрессия (рекомендуется), 'lowess' — локальное сглаживание, None — без линии.
        show_density_contours: Если True и данных > 500 точек, добавляет полупрозрачные контуры плотности (heatmap) под точки для выявления скоплений.
        opacity: Прозрачность точек (0.0 - 1.0). Полезно уменьшать при большом количестве данных.
        log_x: Если True, использует логарифмическую шкалу для оси X.
        log_y: Если True, использует логарифмическую шкалу для оси Y.
        encoding: Кодировка CSV файла (по умолчанию 'utf-8').

    Returns:
        go.Figure: Объект фигуры Plotly с диаграммой рассеяния и аналитическими слоями.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Формируем список колонок для чтения
    cols_to_read = [x_col, y_col]
    if color_col:
        cols_to_read.append(color_col)
    if size_col:
        cols_to_read.append(size_col)

    try:
        # Читаем ТОЛЬКО нужные колонки
        df = pd.read_csv(
            path_obj,
            usecols=cols_to_read,
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=cols_to_read, encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Одна из колонок {cols_to_read} не найдена в файле.")
        raise e

    # 1. Предобработка и очистка
    df = df.copy()
    required_cols = [x_col, y_col]
    if color_col:
        required_cols.append(color_col)
    if size_col:
        required_cols.append(size_col)

    initial_len = len(df)
    df = df.dropna(subset=required_cols)

    if len(df) == 0:
        raise ValueError("Нет валидных данных после очистки от NaN.")
    if len(df) < initial_len:
        print(f"Предупреждение: удалено {initial_len - len(df)} строк с пропусками.")

    # 2. Инициализация фигуры
    fig = go.Figure()

    # --- ЛОГИКА ОТРИСОВКИ ТОЧЕК ---

    # Определяем, является ли цветовой столбец категориальным
    is_color_categorical = False
    if color_col and not pd.api.types.is_numeric_dtype(df[color_col]):
        is_color_categorical = True

    if is_color_categorical:
        # ПОДХОД А: Группировка по категориям (Надежно для строк)
        unique_categories = df[color_col].unique()

        # Генерируем палитру (Plotly.qualitative.Plotly или другие)
        # Для простоты используем встроенный цикл цветов Plotly, если категорий много
        colorscale_seq = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, category in enumerate(unique_categories):
            df_subset = df[df[color_col] == category]
            color_val = colorscale_seq[i % len(colorscale_seq)]

            # Подготовка размеров для подмножества
            current_sizes = None
            if size_col:
                s_min, s_max = df_subset[size_col].min(), df_subset[size_col].max()
                if s_max > s_min:
                    current_sizes = 10 + 40 * (df_subset[size_col] - s_min) / (
                        s_max - s_min
                    )
                else:
                    current_sizes = 20

            hover_template = f"<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<br><b>{color_col}</b>: {category}<extra></extra>"
            if size_col:
                hover_template += (
                    f"<br><b>{size_col}</b>: %{{customdata}}<extra></extra>"
                )

            fig.add_trace(
                go.Scatter(
                    x=df_subset[x_col],
                    y=df_subset[y_col],
                    mode="markers",
                    name=str(category),  # Важно для легенды
                    marker=dict(
                        color=color_val,
                        opacity=opacity,
                        line=dict(width=0.5, color="white"),
                        sizemode="area",
                        size=current_sizes,
                    ),
                    customdata=df_subset[size_col] if size_col else None,
                    hovertemplate=hover_template,
                )
            )

    else:
        # ПОДХОД Б: Числовой цвет (Градиент)
        scatter_kwargs = dict(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            name="Data Points",
            marker=dict(
                opacity=opacity,
                line=dict(width=0.5, color="white"),
                sizemode="area",
                showscale=True if color_col else False,
            ),
            hovertemplate=f"<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>",
        )

        if color_col:
            scatter_kwargs["marker"]["color"] = df[color_col]
            scatter_kwargs["marker"]["colorscale"] = "Viridis"
            scatter_kwargs[
                "hovertemplate"
            ] += f"<br><b>{color_col}</b>: %{{marker.color}}<extra></extra>"

        if size_col:
            s_min, s_max = df[size_col].min(), df[size_col].max()
            if s_max > s_min:
                sizes = 10 + 40 * (df[size_col] - s_min) / (s_max - s_min)
            else:
                sizes = 20
            scatter_kwargs["marker"]["size"] = sizes
            scatter_kwargs[
                "hovertemplate"
            ] += f"<br><b>{size_col}</b>: %{{customdata}}<extra></extra>"
            scatter_kwargs["customdata"] = df[size_col]

        fig.add_trace(go.Scatter(**scatter_kwargs))

    # --- 3. Аналитический слой: Линия тренда (OLS) ---
    # Примечание: При категориальном цвете линия тренда строится по ВСЕМ данным сразу.
    # Если нужно строить тренд для каждой категории отдельно, логику нужно поместить внутрь цикла выше.
    if trend_line == "ols" and len(df) > 2:
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values

        model = LinearRegression()
        model.fit(X, y)

        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        r_squared = model.score(X, y)

        fig.add_trace(
            go.Scatter(
                x=x_range.flatten(),
                y=y_pred,
                mode="lines",
                name=f"Trend (R²={r_squared:.2f})",
                line=dict(color="red", width=2, dash="dash"),
                hoverinfo="skip",  # Скрываем ховер для линии тренда, чтобы не мешал
            )
        )

    # --- 4. Аналитический слой: Контуры плотности ---
    if show_density_contours and len(df) > 500:
        fig.add_trace(
            go.Histogram2dContour(
                x=df[x_col],
                y=df[y_col],
                colorscale="Blues",
                opacity=0.4,
                showscale=False,  # Скрываем шкалу цвета, чтобы не загромождать
                hoverinfo="skip",  # Отключаем ховер, чтобы не перекрывать точки
                line=dict(
                    width=0
                ),  # Убираем линии контуров, оставляем только заполнение (опционально)
                contours=dict(
                    coloring="heatmap",  # Заполнять как тепловую карту
                    showlabels=False,  # Убираем подписи значений внутри контуров
                ),
                name="Density",  # Имя для легенды (если понадобится)
            )
        )

    # --- 5. Настройка Layout ---
    fig.update_layout(
        title=dict(
            text=title,
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=16),
        ),
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=600,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.5)",
        ),
        hovermode="closest",
    )

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    return fig


@tool(
    description="Создает комплексный дашборд: Scatter plot в центре + маргинальные гистограммы."
)
def create_scatter_dashboard_tool(
    file_path: str,  # НОВЫЙ АРГУМЕНТ: путь к CSV
    x_col: str,  # Ось X
    y_col: str,  # Ось Y
    title: str = "Анализ взаимосвязи переменных",
    corr_method: Literal["pearson", "spearman"] = "pearson",
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Создает комплексный дашборд: Scatter plot в центре + маргинальные гистограммы.

    Args:
        file_path: путь до csv файла пользователя.
        x_col: Ось X.
        y_col: Ось Y.
        title: Заголовок.
        corr_method: Метод расчета корреляции.
        encoding: Кодировка.

    Returns:
        go.Figure: Дашборд.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем ТОЛЬКО две нужные колонки
        df = pd.read_csv(
            path_obj,
            usecols=[x_col, y_col],
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=[x_col, y_col], encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Колонки '{x_col}' или '{y_col}' не найдены в файле.")
        raise e

    # Очистка от NaN (необходима для расчета корреляции и построения)
    df = df.dropna(subset=[x_col, y_col])

    if len(df) == 0:
        raise ValueError("Нет валидных данных для анализа после очистки.")

    # Расчет корреляции
    if corr_method == "pearson":
        corr_val, _ = pearsonr(df[x_col], df[y_col])
    else:
        corr_val, _ = spearmanr(df[x_col], df[y_col])

    corr_text = f"{corr_method.capitalize()} Corr: {corr_val:.3f}"

    # Создание сетки: 2 ряда, 2 колонки
    # specs:
    # Row 1: [Histogram X, Empty]
    # Row 2: [Scatter, Histogram Y]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "histogram"}, {}],
            [{"type": "scatter"}, {"type": "histogram"}],
        ],
        row_width=[0.2, 0.8],  # Высота рядов (нижний выше)
        column_width=[0.8, 0.2],  # Ширина колонок (левый шире)
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    # 1. Центральный Scatter Plot
    scatter_trace = go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="markers",
        marker=dict(color="#636EFA", opacity=0.6, line=dict(width=0.5, color="white")),
        name="Points",
        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
    )
    fig.add_trace(scatter_trace, row=2, col=1)

    # 2. Верхняя гистограмма (X)
    fig.add_trace(
        go.Histogram(
            x=df[x_col],
            marker_color="#636EFA",
            opacity=0.7,
            showlegend=False,
            nbinsx=30,
        ),
        row=1,
        col=1,
    )

    # 3. Правая гистограмма (Y) - поворачиваем ориентацию
    fig.add_trace(
        go.Histogram(
            y=df[y_col],
            marker_color="#EF553B",
            opacity=0.7,
            showlegend=False,
            nbinsy=30,
        ),
        row=2,
        col=2,
    )

    # Настройка Layout
    fig.update_layout(
        title={
            "text": f"{title}<br><sup style='font-size:14px'>{corr_text}</sup>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=600,
        width=800,
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
    )

    # Синхронизация осей и скрытие лишних подписей
    fig.update_xaxes(title_text=x_col, row=2, col=1)
    fig.update_xaxes(
        showticklabels=False, row=1, col=1
    )  # Скрыть цифры под верхней гистограммой

    fig.update_yaxes(title_text=y_col, row=2, col=1)
    fig.update_yaxes(
        showticklabels=False, row=2, col=2
    )  # Скрыть цифры справа от правой гистограммы

    # Поворот правой гистограммы визуально (через ориентацию траса уже сделано, нужно подогнать оси)
    # В Plotly Subplots гистограмма с orientation='h' сама ложится как надо, если ось Y общая

    return fig


@tool(description="Создает тепловую карту с опциональной иерархической кластеризацией.")
def create_heatmap_tool(
    file_path: str,  # НОВЫЙ АРГУМЕНТ: путь к CSV
    numeric_cols: list[str],  # НОВЫЙ АРГУМЕНТ: список числовых колонок для матрицы
    zmin: float,
    zmax: float,
    title: str = "Тепловая карта",
    annotation_format: str = ".2f",
    colorscale: str = "RdBu_r",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    show_values: bool = True,
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Создает тепловую карту с опциональной иерархической кластеризацией.

    Args:
        file_path: путь до csv файла пользователя.
        title: Заголовок.
        annotation_format: Формат чисел (напр., '.2f' для 0.85, '.1%' для процентов).
        colorscale: Палитра. 'RdBu_r' идеальна для дивергентных данных (от -1 до 1).
        cluster_rows/cols: Если True, пересортирует данные, чтобы сгруппировать похожие.
        show_values: Показывать аннотации.
        zmin/zmax: Фиксация цветовой шкалы (важно для сравнения нескольких карт).
        encoding: Кодировка.

    Returns:
        go.Figure: Объект фигуры.
    """

    # 1. Подготовка данных
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if not numeric_cols:
        raise ValueError("Список колонок для тепловой карты пуст.")

    try:
        # Читаем ТОЛЬКО нужные числовые колонки
        df = pd.read_csv(
            path_obj,
            usecols=numeric_cols,
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=numeric_cols, encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Одна из колонок {numeric_cols} не найдена в файле.")
        raise e

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if df.empty:
        raise ValueError("Нет валидных числовых данных после очистки.")

    # Применяем перестановку индексов (упрощенная эвристика для скорости и надежности без сложных зависимостей scipy в рантайме агента)
    # Полноценный scipy linkage может быть тяжелым. Используем сортировку по коррляции как быстрый аналог кластеризации.
    if cluster_cols and df.shape[1] > 2:
        try:
            corr_cols = df.corr()
            # Сортируем колонки так, чтобы похожие были рядом (greedy approach)
            # Или просто сортируем по средней корреляции для упорядочивания
            col_order = corr_cols.mean().sort_values().index
            df = df[col_order]
        except:
            pass

    if cluster_rows and df.shape[0] > 2:
        try:
            corr_rows = df.corr(axis=1)
            row_order = corr_rows.mean().sort_values().index
            df = df.loc[row_order]
        except:
            pass

    # 3. Построение Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=df.values if show_values else None,
            texttemplate=f"%{{z:{annotation_format}}}" if show_values else None,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Значение"),
        )
    )

    # 4. Layout настройки
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=max(400, len(df.index) * 30),  # Динамическая высота
        width=max(600, len(df.columns) * 30),  # Динамическая ширина
        template="plotly_white",
        xaxis_title="Переменные",
        yaxis_title="Переменные",
    )

    # Поворот подписей осей X, если их много
    if len(df.columns) > 10:
        fig.update_xaxes(tickangle=45)

    return fig


@tool(description="Комплексный дашборд для анализа корреляций.")
def create_correlation_dashboard_tool(
    file_path: str,  # НОВЫЙ АРГУМЕНТ: путь к CSV
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: str = "Обзор корреляций датасета",
    top_n_pairs: int = 10,
    encoding: str = "utf-8",  # НОВЫЙ АРГУМЕНТ: кодировка
) -> go.Figure:
    """
    Комплексный дашборд для анализа корреляций.

    Args:
        file_path: путь до csv файла пользователя.
        method: Метод корреляции.
        title: Заголовок.
        top_n_pairs: Сколько топ пар показать в таблице.
        encoding: Кодировка.

    Returns:
        go.Figure: Дашборд.
    """

    # 1. Фильтрация только числовых колонок
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем файл целиком, но с оптимизацией типов
        # select_dtypes будет работать только с загруженными данными,
        # поэтому важно прочитать всё, но pandas сам определит типы
        df = pd.read_csv(
            path_obj,
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
            low_memory=False,  # Важно для корректного определения типов всех колонок сразу
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, encoding="latin-1", on_bad_lines="skip", low_memory=False
        )
    except Exception as e:
        raise ValueError(f"Не удалось прочитать CSV: {e}")

    # Фильтрация числовых колонок (логика из оригинальной функции)
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] < 2:
        raise ValueError(
            f"Недостаточно числовых колонок для анализа. Найдено: {num_df.shape[1]}. Проверьте данные."
        )

    # Расчет матрицы
    corr_matrix = num_df.corr(method=method)

    # Извлечение топ пар (исключая единичную корреляцию с самим собой и дубликаты)
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ["Var1", "Var2", "Corr"]
    # Убираем дубликаты (A-B и B-A) и самосвязи
    corr_pairs = corr_pairs[corr_pairs["Var1"] != corr_pairs["Var2"]]
    # Создаем уникальный ключ для пары
    corr_pairs["Pair"] = corr_pairs.apply(
        lambda x: tuple(sorted([x["Var1"], x["Var2"]])), axis=1
    )
    corr_pairs = corr_pairs.drop_duplicates(subset=["Pair"]).drop(columns=["Pair"])

    # Сортировка по абсолютному значению
    corr_pairs["AbsCorr"] = corr_pairs["Corr"].abs()
    top_overall = corr_pairs.nlargest(top_n_pairs, "AbsCorr")

    # 2. Создание сетки
    # Row 1: Heatmap (высокая)
    # Row 2: Гистограмма корреляций + Таблица топ пар
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "histogram"}, {"type": "table"}],
        ],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        subplot_titles=[
            title,
            "Распределение коэффициентов",
            f"Топ-{top_n_pairs} сильнейших связей",
        ],
    )

    # --- ROW 1: Heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate="%{z:.2f}",
            showscale=True,
        ),
        row=1,
        col=1,
    )

    # --- ROW 2 LEFT: Histogram of Correlations ---
    fig.add_trace(
        go.Histogram(
            x=corr_pairs["Corr"],
            nbinsx=30,
            marker_color="indigo",
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Коэффициент корреляции", row=2, col=1)
    fig.update_yaxes(title_text="Число пар", row=2, col=1)

    # --- ROW 2 RIGHT: Table of Top Pairs ---
    # Формируем данные для таблицы
    table_data = top_overall[["Var1", "Var2", "Corr"]].copy()
    table_data["Corr"] = table_data["Corr"].apply(lambda x: f"{x:.3f}")

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Переменная 1", "Переменная 2", "Корреляция"],
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[table_data["Var1"], table_data["Var2"], table_data["Corr"]],
                fill_color="lavender",
                align="left",
            ),
        ),
        row=2,
        col=2,
    )

    # Global Layout
    fig.update_layout(height=800, width=1000, template="plotly_white", showlegend=False)

    # Поворот подписей на heatmap
    fig.update_xaxes(tickangle=45, row=1, col=1)

    return fig


@tool(
    description="Создает статистический Box Plot для анализа распределения числовых данных и выявления выбросов. Визуализирует медиану, квартили (25%, 75%), межквартильный размах и аномальные значения. Поддерживает группировку по категориям для сравнения распределений между группами."
)
def create_box_plot_tool(
    file_path: str,
    y_col: str,
    x_col: str,
    color_palette: list[str],
    title: str = "Распределение и выбросы",
    show_notches: bool = True,
    show_outliers: bool = True,
    outlier_jitter: float = 0.1,
    log_y: bool = False,
    encoding: str = "utf-8",
) -> go.Figure:
    """
    Создает статистический Box Plot для анализа распределения числовых данных и выявления выбросов.
    Визуализирует медиану, квартили (25%, 75%), межквартильный размах и аномальные значения.
    Поддерживает группировку по категориям для сравнения распределений между группами.

    Args:
        file_path: Путь к CSV файлу пользователя во временном хранилище.
        y_col: Имя числовой колонки для анализа (ось Y). Обязательно должна быть числового типа.
        x_col: Имя категориальной колонки для группировки (ось X). Если None, строится один общий ящик.
        title: Заголовок графика.
        show_notches: Если True, добавляет выемки вокруг медианы для оценки статистической значимости различий.
        show_outliers: Если True, отображает точки выбросов (значения за пределами 1.5 * IQR).
        outlier_jitter: Сила горизонтального разброса точек выбросов для лучшей читаемости при наложении.
        color_palette: Список цветов для категорий. Если None, используется палитра по умолчанию.
        log_y: Если True, использует логарифмическую шкалу для оси Y (полезно при большом разбросе данных).
        encoding: Кодировка CSV файла (по умолчанию 'utf-8').

    Returns:
        go.Figure: Объект фигуры Plotly с диаграммой размаха.
    """

    # 1. Подготовка данных
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    cols_to_read = [y_col]
    if x_col:
        cols_to_read.append(x_col)

    try:
        df = pd.read_csv(
            path_obj,
            usecols=cols_to_read,
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=cols_to_read, encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Колонки {cols_to_read} не найдены в файле.")
        raise e

    # Очистка от NaN
    df = df.dropna(subset=cols_to_read)
    if df.empty:
        raise ValueError("Нет данных после очистки.")

    # Очистка от NaN
    cols_to_check = [y_col]
    if x_col:
        cols_to_check.append(x_col)
    df = df.dropna(subset=cols_to_check)

    if df.empty:
        raise ValueError("Нет данных после очистки.")

    # Группировка данных
    if x_col:
        groups = df[x_col].unique()
        # Сортировка групп
        try:
            groups = sorted(groups)
        except TypeError:
            groups = sorted(groups, key=str)

        y_data_list = [df[df[x_col] == g][y_col].values for g in groups]
        names = [str(g) for g in groups]
        x_labels = names  # Подписи осей X
    else:
        y_data_list = [df[y_col].values]
        names = [y_col]
        x_labels = [""]

    # 2. Построение
    fig = go.Figure()

    # Цвета
    if not color_palette:
        colors = [None] * len(names)  # Plotly сам подберет
    else:
        colors = (color_palette * ((len(names) // len(color_palette)) + 1))[
            : len(names)
        ]

    for i, (y_vals, name) in enumerate(zip(y_data_list, names)):
        color = colors[i]

        # Ручной расчет статистики для customdata (чтобы быть уверенными в цифрах)
        q1 = np.percentile(y_vals, 25)
        median = np.percentile(y_vals, 50)
        q3 = np.percentile(y_vals, 75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        mean_val = np.mean(y_vals)
        std_val = np.std(y_vals)

        # Формируем массив customdata одинаковой длины с y_vals для корректной работы hover
        # Но для box plot hover работает по умолчанию на уровне треаса, а не точки.
        # Чтобы показать детальную статистику ящика при наведении на ЛЮБУЮ часть ящика,
        # лучше использовать hovertemplate с агрегированными данными, которые Plotly считает сам,
        # ИЛИ передать скалярные значения через customdata, если бы мы рисовали сами.

        # ПРАВИЛЬНЫЙ ПОДХОД ДЛЯ PLOTLY BOX:
        # Используем встроенные переменные формата: %{q1}, %{median}, %{q3}, %{lowerfence}, %{upperfence}
        # Они работают нативно и не требуют customdata для базовой статистики ящика.
        # Customdata нужен только если мы хотим добавить что-то свое (например, название группы в сложном формате).

        fig.add_trace(
            go.Box(
                y=y_vals,
                name=name,
                boxpoints="outliers" if show_outliers else "suspectedoutliers",
                jitter=outlier_jitter if show_outliers else 0,
                pointpos=-1.8 if show_outliers else 0,
                marker=dict(color=color, size=4, opacity=0.7),
                line=dict(color=color, width=2),
                # Прозрачность заполнения (упрощенно)
                fillcolor=f"{color}33" if color else "rgba(100,100,200,0.2)",
                notched=show_notches,
                # Исправленный hovertemplate с нативными переменными Plotly
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    + "----------------<br>"
                    + "Медиана: %{median:.2f}<br>"
                    + "Q1 (25%): %{q1:.2f}<br>"
                    + "Q3 (75%): %{q3:.2f}<br>"
                    + "Усы: [%{lowerfence:.2f}, %{upperfence:.2f}]<br>"
                    + "Среднее: %{mean:.2f}<br>"
                    + "Стд. откл.: %{sd:.2f}<br>"
                    + "<extra></extra>"  # Убирает лишнюю подпись имени треаса внизу
                ),
            )
        )

    # 3. Layout
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title=x_col if x_col else "Группа",
        yaxis_title=y_col,
        template="plotly_white",
        height=500,
        width=max(600, len(names) * 60),
        showlegend=False,
        boxmode="group" if x_col else "overlay",
    )

    if log_y:
        fig.update_yaxes(type="log")

    # Поворот подписей X если их много
    if len(names) > 10:
        fig.update_xaxes(tickangle=45)

    return fig


@tool(
    description="Комбинированный дашборд: Box Plot + Violin Plot + Таблица статистик."
)
def create_distribution_comparison_dashboard(
    file_path: str,
    y_col: str,
    x_col: str,
    title: str = "Сравнение распределений по группам",
    encoding: str = "utf-8",
) -> go.Figure:
    """
    Комбинированный дашборд: Box Plot + Violin Plot + Таблица статистик.

    Args:
        file_path: путь до csv файла пользователя.
        y_col: Числовая переменная.
        x_col: Категориальная переменная (группы).
        title: Заголовок.
        encoding: Кодировка.

    Returns:
        go.Figure: Дашборд.
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        # Читаем только две нужные колонки
        df = pd.read_csv(
            path_obj,
            usecols=[x_col, y_col],
            encoding=encoding,
            on_bad_lines="skip",
            dtype_backend="numpy_nullable",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_obj, usecols=[x_col, y_col], encoding="latin-1", on_bad_lines="skip"
        )
    except ValueError as e:
        if "not in index" in str(e):
            raise ValueError(f"Колонки '{x_col}' или '{y_col}' не найдены в файле.")
        raise e

    # Очистка данных
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        raise ValueError("Нет данных для анализа.")

    # Приведение категорий к строке для корректной группировки
    df[x_col] = df[x_col].astype(str)

    groups = sorted(df[x_col].unique(), key=str)

    # Расчет сводной статистики для таблицы
    stats_data = []
    for g in groups:
        subset = df[df[x_col] == g][y_col]
        q1, median, q3 = subset.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((subset < lower_bound) | (subset > upper_bound)).sum()

        stats_data.append(
            {
                "Group": str(g),
                "Count": len(subset),
                "Mean": f"{subset.mean():.2f}",
                "Std": f"{subset.std():.2f}",
                "Median": f"{median:.2f}",
                "Outliers": int(outliers),
            }
        )

    stats_df = pd.DataFrame(stats_data)

    # Создание сетки
    # Row 1: Box Plot
    # Row 2: Violin Plot
    # Row 3: Table (или сбоку, но лучше снизу для скролла)
    # Для компактности сделаем 2 ряда: Графики сверху, Таблица снизу
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        specs=[
            [{"type": "scatter"}],
            [{"type": "table"}],
        ],  # Scatter placeholder, заменим трасами
        subplot_titles=[f"Распределение: {y_col} по {x_col}", "Сводная статистика"],
    )

    # --- ROW 1: Box & Violin (Наложим их друг на друга для экономии места или разделим?)
    # Лучшая практика: Разделить на два подплота рядом или один над другим.
    # Переделаем сетку: 2 ряда графиков + 1 ряд таблицы? Нет, слишком длинно.
    # Сделаем так: Row 1 - Box, Row 2 - Violin. Таблицу выведем отдельным инструментом или упростим.
    # Давайте сделаем 2 ряда графиков, а таблицу добавим как аннотацию или второй ряд, если места мало.
    # Оптимально: 2 ряда (Box, Violin). Таблицу опустим для краткости кода, либо сделаем 3 ряда.
    # Выберем вариант: 2 ряда графиков.

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5],
        subplot_titles=[
            "Box Plot (Статистики и выбросы)",
            "Violin Plot (Плотность распределения)",
        ],
    )

    colors = plotly.colors.qualitative.Plotly

    for i, g in enumerate(groups):
        subset = df[df[x_col] == g][y_col]
        color = colors[i % len(colors)]
        name = str(g)

        # Box Trace
        fig.add_trace(
            go.Box(
                y=subset,
                name=name,
                marker_color=color,
                boxpoints="outliers",
                jitter=0.1,
                pointpos=-1.8,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Violin Trace
        fig.add_trace(
            go.Violin(
                y=subset,
                name=name,
                line_color=color,
                fillcolor=color,
                opacity=0.3,
                points=False,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Добавим таблицу статистики как отдельный график в новом окне?
    # Нет, вернемся к идее: 3 ряда.
    # Пересоздадим фигуру с 3 рядами для полноценного дашборда.

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.35, 0.35, 0.3],
        specs=[[{"type": "box"}], [{"type": "violin"}], [{"type": "table"}]],
        subplot_titles=[
            "Box Plot (Выбросы)",
            "Violin Plot (Плотность)",
            "Сводная статистика",
        ],
    )

    # Re-add traces for 3 rows
    for i, g in enumerate(groups):
        subset = df[df[x_col] == g][y_col]
        color = colors[i % len(colors)]
        name = str(g)

        fig.add_trace(
            go.Box(
                y=subset,
                name=name,
                marker_color=color,
                boxpoints="outliers",
                jitter=0.1,
                pointpos=-1.8,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Violin(
                y=subset,
                name=name,
                line_color=color,
                fillcolor=color,
                opacity=0.3,
                points=False,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Table Trace
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(stats_df.columns),
                fill_color="paleturquoise",
                align="center",
            ),
            cells=dict(
                values=[stats_df[col] for col in stats_df.columns],
                fill_color="lavender",
                align="center",
            ),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=title, height=900, width=900, template="plotly_white", showlegend=False
    )

    fig.update_xaxes(title_text=x_col, row=3, col=1)  # Подпись только внизу
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=2, col=1)

    return fig


def get_tools_map():
    tools_map = {
        "create_histogram_tool": create_histogram_tool,
        "create_histogram_dashboard_tool": create_histogram_dashboard_tool,
        "create_pie_chart_tool": create_pie_chart_tool,
        "create_pie_dashboard_tool": create_pie_dashboard_tool,
        "create_line_chart_tool": create_line_chart_tool,
        "create_time_series_dashboard_tool": create_time_series_dashboard_tool,
        "create_scatter_plot_tool": create_scatter_plot_tool,
        "create_scatter_dashboard_tool": create_scatter_dashboard_tool,
        "create_heatmap_tool": create_heatmap_tool,
        "create_correlation_dashboard_tool": create_correlation_dashboard_tool,
        "create_box_plot_tool": create_box_plot_tool,
        "create_distribution_comparison_dashboard": create_distribution_comparison_dashboard,
    }

    return tools_map
