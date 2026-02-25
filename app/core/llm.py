from logging import Logger
from typing import Any

from langchain_gigachat import GigaChat

from app.core.config import Settings


def get_gigachat(settings: Settings, tools_map: dict[str, Any], logger: Logger):
    try:
        logger.info("🔄 Начинаем соединение с GigaChat")

        # Преобразуем функции в объекты Tool из LangChain
        tools = []
        for tool in tools_map.values():
            tools.append(tool)

        # Создаём модель LLM с привязанными инструментами
        llm = GigaChat(credentials=settings.GIGACHAT_API_AUTH_KEY)
        llm_with_tools = llm.bind_tools(tools)

        return llm_with_tools

    except Exception as e:
        logger.error(f"❌ Ошибка при подключении к GigaChat: {e}")
        raise
