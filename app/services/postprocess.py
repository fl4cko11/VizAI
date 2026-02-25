import uuid

import plotly.graph_objects as go
from plotly.io import to_html

from app.core.config import Settings
from app.schemas.agent_state import GeneratedArtifact


def fig_to_html(tool_result: go.Figure, settings: Settings):
    """
    Сохраняет Figure в HTML и обновляет состояние.
    """
    # 1. Генерация безопасного имени файла
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"viz_{unique_id}.html"

    # 2. Определение пути сохранения (временная папка)
    # Best Practice: Используйте tempfile или специальную папку для сессий
    base_dir = settings.HTML_FILES_DIR
    base_dir.mkdir(exist_ok=True)
    full_path = base_dir / safe_filename

    # 3. Сериализация в HTML (Ключевой момент)
    # include_plotlyjs='cdn' уменьшает размер файла, но требует интернета у пользователя.
    # include_plotlyjs=True встраивает библиотеку (файл ~3MB), но работает офлайн.
    # Для Telegram лучше 'cdn', так как лимит на файл 50MB, но меньше вес = быстрее скачивание.
    html_content = to_html(
        tool_result,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "responsive": True,
            "displayModeBar": True,
        },  # Удобный интерфейс в браузере
    )

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 4. Обновление состояния
    new_artifact = GeneratedArtifact(
        file_path=str(full_path),
        file_name=safe_filename,
        mime_type="text/html",
    )

    return new_artifact
