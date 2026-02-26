from typing import Literal

from pydantic import BaseModel, Field


class HistogramInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    column_name: str = Field(..., description="Имя переменной для осей и легенды.")
    title: str = Field(..., description="Заголовок графика.")
    nbins: int = Field(..., description="Количество бинов.")
    color: str = Field(
        default="#636EFA", description="Цвет столбцов (hex или named color)."
    )
    show_stats: bool = Field(
        default=True,
        description="Если True, добавляет вертикальные линии среднего и медианы.",
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class HistogramDashboardInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    column_names: list[str] = Field(
        ..., description="Список имён колонок для гистограмм."
    )
    dashboard_title: str = Field(
        default="Панель распределения данных", description="Общий заголовок дашборда."
    )
    cols: int = Field(default=2, description="Количество колонок в сетке.")
    shared_axes: bool = Field(
        default=True,
        description="Если True, оси X и Y синхронизированы для удобного сравнения.",
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class PieChartInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    column_name: str = Field(..., description="Имя категориальной колонки.")
    color_palette: list[str] = Field(..., description="Список цветов для секторов.")
    title: str = Field(
        default="Распределение категорий", description="Заголовок диаграммы."
    )
    min_share_threshold: float = Field(
        default=0.05,
        description="Минимальная доля категории для отдельного сектора (остальные объединяются в 'Other').",
    )
    max_categories: int = Field(
        default=8,
        description="Максимальное количество отдельных категорий на графике.",
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class PieDashboardInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    category_col: str = Field(..., description="Колонка с категориями (для секторов).")
    group_by_col: str = Field(
        ..., description="Колонка для группировки (по ней создаётся отдельный пончик)."
    )
    title: str = Field(
        default="Сравнение распределений по группам", description="Заголовок дашборда."
    )
    top_n_groups: int = Field(
        default=4, description="Количество самых крупных групп для отображения."
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class LineChartInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    x_col: str = Field(..., description="Столбец для оси X (обычно дата/время).")
    y_cols: list[str] = Field(..., description="Один или несколько столбцов для оси Y.")
    rolling_window: int = Field(
        ..., description="Размер окна для скользящего среднего."
    )
    title: str = Field(default="Динамика показателей", description="Заголовок графика.")
    sort_x: bool = Field(
        default=True, description="Если True, данные сортируются по оси X."
    )
    handle_missing: Literal["drop", "interpolate", "gap"] = Field(
        default="gap",
        description="Как обрабатывать пропущенные значения: 'drop' — удалить, 'interpolate' — интерполировать, 'gap' — оставить разрыв.",
    )
    show_markers: bool = Field(
        default=False, description="Показывать ли маркеры на точках."
    )
    line_width: int = Field(default=2, description="Толщина линии.")
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class TimeSeriesDashboardInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    x_col: str = Field(..., description="Колонка времени (ось X).")
    y_col: str = Field(..., description="Целевая метрика (ось Y).")
    title: str = Field(
        default="Аналитика временного ряда", description="Заголовок дашборда."
    )
    show_rolling_avg: bool = Field(
        default=True, description="Показать скользящее среднее на основном графике."
    )
    rolling_window: int = Field(default=7, description="Окно для скользящего среднего.")
    show_pct_change: bool = Field(
        default=True, description="Показать нижний график с процентным изменением."
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class ScatterPlotInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    x_col: str = Field(..., description="Имя числовой колонки для оси X.")
    y_col: str = Field(..., description="Имя числовой колонки для оси Y.")
    color_col: str | None = Field(
        None, description="Колонка для раскраски точек (категориальная или числовая)."
    )
    size_col: str | None = Field(
        None, description="Колонка для размера точек (числовая)."
    )
    title: str = Field(default="Диаграмма рассеяния", description="Заголовок графика.")
    trend_line: Literal["ols", "lowess"] = Field(
        default="ols", description="Тип линии тренда: 'ols' — линейная регрессия."
    )
    show_density_contours: bool = Field(
        default=False, description="Показать контуры плотности при большом числе точек."
    )
    opacity: float = Field(default=0.6, description="Прозрачность точек (0.0–1.0).")
    log_x: bool = Field(
        default=False, description="Использовать логарифмическую шкалу по X."
    )
    log_y: bool = Field(
        default=False, description="Использовать логарифмическую шкалу по Y."
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class ScatterDashboardInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    x_col: str = Field(..., description="Имя числовой колонки для оси X.")
    y_col: str = Field(..., description="Имя числовой колонки для оси Y.")
    title: str = Field(
        default="Анализ взаимосвязи переменных", description="Заголовок дашборда."
    )
    corr_method: Literal["pearson", "spearman"] = Field(
        default="pearson", description="Метод расчёта корреляции."
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class HeatmapInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    numeric_cols: list[str] = Field(
        ..., description="Список числовых колонок для матрицы корреляций или значений."
    )
    zmin: float = Field(..., description="Минимальное значение цветовой шкалы.")
    zmax: float = Field(..., description="Максимальное значение цветовой шкалы.")
    title: str = Field(default="Тепловая карта", description="Заголовок графика.")
    annotation_format: str = Field(
        default=".2f", description="Формат отображения чисел на ячейках."
    )
    colorscale: str = Field(
        default="RdBu_r", description="Цветовая палитра (например, RdBu_r, Viridis)."
    )
    cluster_rows: bool = Field(
        default=True, description="Пересортировать строки для группировки похожих."
    )
    cluster_cols: bool = Field(
        default=True, description="Пересортировать столбцы для группировки похожих."
    )
    show_values: bool = Field(
        default=True, description="Показывать значения внутри ячеек."
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class CorrelationDashboardInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson", description="Метод расчёта корреляции."
    )
    title: str = Field(
        default="Обзор корреляций датасета", description="Заголовок дашборда."
    )
    top_n_pairs: int = Field(
        default=10,
        description="Количество топ пар с наибольшей корреляцией для отображения.",
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class BoxPlotInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    y_col: str = Field(..., description="Числовая колонка для анализа (ось Y).")
    x_col: str | None = Field(
        None,
        description="Категориальная колонка для группировки (ось X). Если None — один ящик.",
    )
    color_palette: list[str] = Field(..., description="Список цветов для групп.")
    title: str = Field(
        default="Распределение и выбросы", description="Заголовок графика."
    )
    show_notches: bool = Field(
        default=True, description="Показать выемки вокруг медианы (оценка различий)."
    )
    show_outliers: bool = Field(default=True, description="Показать точки выбросов.")
    outlier_jitter: float = Field(
        default=0.1, description="Горизонтальный разброс точек выбросов."
    )
    log_y: bool = Field(default=False, description="Логарифмическая шкала по оси Y.")
    encoding: str = Field(default="utf-8", description="Кодировка файла.")


class DistributionComparisonDashboardInput(BaseModel):
    file_path: str = Field(..., description="Путь до CSV-файла пользователя.")
    y_col: str = Field(..., description="Числовая переменная для анализа.")
    x_col: str = Field(..., description="Категориальная переменная для группировки.")
    title: str = Field(
        default="Сравнение распределений по группам", description="Заголовок дашборда."
    )
    encoding: str = Field(default="utf-8", description="Кодировка файла.")
