from app.services.tools.viz_tools import (
    create_box_plot_tool,
    create_correlation_dashboard_tool,
    create_distribution_comparison_dashboard,
    create_heatmap_tool,
    create_histogram_dashboard_tool,
    create_histogram_tool,
    create_line_chart_tool,
    create_pie_chart_tool,
    create_pie_dashboard_tool,
    create_scatter_dashboard_tool,
    create_scatter_plot_tool,
    create_time_series_dashboard_tool,
)

SALES_FILE = "app/tests/sales_transactions.csv"
EMP_FILE = "app/tests/employee_performance.csv"
SURVEY_FILE = "app/tests/customer_survey.csv"

print("1. Тест: create_histogram_tool")
fig1 = create_histogram_tool.invoke(
    {
        "file_path": SALES_FILE,
        "column_name": "total_amount",
        "title": "Распределение чеков",
    }
)

print("2. Тест: create_histogram_dashboard_tool")
fig2 = create_histogram_dashboard_tool.invoke(
    {
        "file_path": SALES_FILE,
        "column_names": ["quantity", "unit_price", "total_amount"],
        "dashboard_title": "Панель распределения данных",
    }
)

print("3. Тест: create_pie_chart_tool")
fig3 = create_pie_chart_tool.invoke(
    {"file_path": SALES_FILE, "column_name": "category", "title": "Доли категорий"}
)

print("4. Тест: create_pie_dashboard_tool")
fig4 = create_pie_dashboard_tool.invoke(
    {
        "file_path": SALES_FILE,
        "category_col": "category",
        "group_by_col": "region",
        "top_n_groups": 4,
    }
)

print("5. Тест: create_line_chart_tool")
fig5 = create_line_chart_tool.invoke(
    {
        "file_path": SALES_FILE,
        "x_col": "date",
        "y_cols": ["total_amount"],  # Можно передать и строку, если колонка одна
        "title": "Динамика продаж",
    }
)

print("6. Тест: create_time_series_dashboard_tool")
fig6 = create_time_series_dashboard_tool.invoke(
    {"file_path": SALES_FILE, "x_col": "date", "y_col": "total_amount"}
)

print("7. Тест: create_scatter_plot_tool")
fig7 = create_scatter_plot_tool.invoke(
    {
        "file_path": SALES_FILE,
        "x_col": "quantity",
        "y_col": "total_amount",
        "color_col": "category",
        "size_col": "unit_price",
        "trend_line": "ols",
    }
)

print("8. Тест: create_scatter_dashboard_tool")
fig8 = create_scatter_dashboard_tool.invoke(
    {"file_path": EMP_FILE, "x_col": "YearsExperience", "y_col": "Salary"}
)

print("9. Тест: create_heatmap_tool")
num_cols = [
    "YearsExperience",
    "PerformanceScore",
    "Salary",
    "SatisfactionLevel",
    "ProjectsCompleted",
]
fig9 = create_heatmap_tool.invoke(
    {"file_path": EMP_FILE, "numeric_cols": num_cols, "title": "Корреляции сотрудников"}
)

print("10. Тест: create_correlation_dashboard_tool")
fig10 = create_correlation_dashboard_tool.invoke(
    {"file_path": EMP_FILE, "method": "pearson", "top_n_pairs": 5}
)

print("11. Тест: create_box_plot_tool")
fig11 = create_box_plot_tool.invoke(
    {
        "file_path": SALES_FILE,
        "y_col": "total_amount",
        "x_col": "region",
        "title": "Продажи по регионам",
        "show_outliers": True,
    }
)

print("12. Тест: create_distribution_comparison_dashboard")
fig12 = create_distribution_comparison_dashboard.invoke(
    {"file_path": EMP_FILE, "y_col": "Salary", "x_col": "Department"}
)

print("✅ Все тесты завершены успешно!")

print(create_box_plot_tool.description)
