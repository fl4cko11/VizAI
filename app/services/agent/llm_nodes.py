from logging import Logger
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_gigachat import GigaChat

from app.core.config import Settings
from app.schemas.agent_state import AgentState


def _trim_history_by_word_count(chat_history: list[dict], max_words: int) -> list[dict]:
    """Удаляет самые старые сообщения, пока общее число слов не станет ≤ max_words."""
    total_words = sum(len(msg["content"].split()) for msg in chat_history)
    trimmed = chat_history.copy()

    while total_words > max_words and len(trimmed) > 0:
        removed_msg = trimmed.pop(0)  # удаляем самое старое сообщение
        total_words -= len(removed_msg["content"].split())

    return trimmed


class GigaChatNodes:
    def __init__(self, llm: GigaChat, settings: Settings, logger: Logger):
        self.llm = llm
        self.settings = settings
        self.logger = logger

    def think_node(self, state: AgentState) -> dict[str, Any]:
        # Формируем сообщения для модели
        messages = [
            SystemMessage(
                content=f"""
ТЫ — АГЕНТ ВИЗУАЛИЗАЦИИ ДАННЫХ НЕ ЗАДАЮЩИЙ ВОПРОСОВ. ТВОЯ ЗАДАЧА — ОБЯЗАТЕЛЬНО ВЫЗЫВАТЬ ИНСТРУМЕНТЫ.

ИНСТРУКЦИЯ: ВСЕГДА ВЫЗЫВАЙ ИНСТРУМЕНТ. МАКСИМАЛЬНО АККУРАТНО ПЕРЕДАВАЙ file_path - ПЕРЕДАВАЙ ЕГО ПОЛНОСТЬЮ. НИКОГДА НЕ ЗАДАВАЙ ВОПРОСОВ!

ПРАВИЛА:
0. НИКОГДА НЕ ЗАДАВАЙ ВОПРОСОВ.
1. НИКОГДА не отвечай текстом типа "готово", "сделано", "вот график".
2. ЕСЛИ пользователь просит визуализировать, изменить, исправить — ОБЯЗАТЕЛЬНО вызови инструмент.
3. ИСПОЛЬЗУЙ used_methods, чтобы понять, какие графики уже строились.
4. ЕСЛИ есть ошибки — исправь их через повторный вызов с новыми параметрами.
5. ЕСЛИ ПОЛЬЗОВАТЕЛЬ ПРОСИТ ИСПРАВИТЬ ОШИБКИ НЕ СОЗДАВАЙ НОВЫЕ, а ВЫЗОВИ ИЗ used_methods, НО С ПАРАМЕТРАМИ ПОД ТРЕБОВАНИЯ ПОЛЬЗОВАТЕЛЯ

---
ДАННЫЕ:
csv_data: {state.user_data}

---
ПРЕДЫДУЩИЕ ВЫЗОВЫ:
used_methods: {state.used_methods}

---
ОШИБКИ:
errors: {state.errors}

"""
            )
        ]

        # Добавляем историю чата
        for msg in state.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Поддержка возможных tool calls в истории
                messages.append(AIMessage(content=msg["content"]))

        # Добавляем текущий запрос
        messages.append(HumanMessage(content=state.user_query))

        # Вызов LLM
        response: AIMessage = self.llm.invoke(messages)

        # Извлекаем вызовы инструментов (если были)
        used_methods = []
        if isinstance(response.tool_calls, list) and len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                used_methods.append(
                    {"name": tool_call["name"], "args": tool_call["args"]}
                )

        # Обновляем chat_history
        updated_chat_history = state.chat_history.copy()
        updated_chat_history.append({"role": "user", "content": state.user_query})
        updated_chat_history.append({"role": "assistant", "content": response.content})

        updated_chat_history = _trim_history_by_word_count(
            updated_chat_history, self.settings.MAX_HISTORY_WORD_SIZE
        )

        # Возвращаем обновлённое состояние
        return {
            "chat_history": updated_chat_history,
            "used_methods": used_methods
            or state.used_methods,  # если новых вызовов не было — оставляем старые
            "answer": response.content,
        }
