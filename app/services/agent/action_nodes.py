from typing import Any

from app.core.config import Settings
from app.schemas.agent_state import AgentState
from app.services.postprocess import fig_to_html


class ActionNodes:
    def __init__(self, settings: Settings, tools_map: dict[str, Any]):
        self.settings = settings
        self.tools_map = tools_map

    def action_node(self, state: AgentState):
        if not state.used_methods:
            # Нет инструментов для вызова
            return {
                "artifacts": [],
                "answer": "Анализ завершён, но визуализация не требовалась.",
            }

        artifacts = []
        errors = []

        for method in state.used_methods:
            tool_name = method["name"]
            args = method["args"]

            if tool_name not in self.tools_map:
                errors.append(f"Инструмент '{tool_name}' не найден.")
                continue

            tool_func = self.tools_map[tool_name]

            try:
                # Вызов инструмента
                result = tool_func.invoke(input=args)

                artifact = fig_to_html(result, self.settings)

                artifacts.append(artifact)

            except Exception as e:
                errors.append(f"Ошибка при вызове '{tool_name}': {str(e)}")
                continue

        # Формируем ответ
        if errors:
            answer = "Частичный результат:\n" + "\n".join(errors)
        else:
            answer = f"Успешно создано {len(artifacts)} визуализаций."

        old_answer = state.answer or ""
        updated_answer = f"{old_answer}\n\n{answer}".strip()

        return {
            "artifacts": artifacts,
            "answer": updated_answer,
            "errors": errors,
            "iteration": state.iteration + 1,
        }
