from app.core.config import Settings
from app.services.postprocess import fig_to_html
from app.services.tools.viz_tools import create_histogram_tool

settings = Settings()

fig1 = create_histogram_tool.invoke(
    {
        "file_path": "app/tests/sales_transactions.csv",
        "column_name": "total_amount",
        "title": "Распределение чеков",
    }
)

result = fig_to_html(fig1, settings)

print(result)
