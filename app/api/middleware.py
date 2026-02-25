# app/api/middleware.py
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from app.core.config import Settings
from app.core.llm import get_gigachat
from app.core.logging import get_logger
from app.services.agent.action_nodes import ActionNodes
from app.services.agent.agent import build_agent
from app.services.agent.llm_nodes import GigaChatNodes
from app.services.agent.route_nodes import RouteNodes
from app.services.tools.viz_tools import get_tools_map


class AgentMiddleware(BaseMiddleware):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent = None

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if self.agent is None:
            settings = Settings()
            logger = get_logger(settings)
            tools_map = get_tools_map()

            llm = get_gigachat(settings, tools_map, logger)

            llm_nodes = GigaChatNodes(llm, settings, logger)
            action_nodes = ActionNodes(settings, tools_map)
            route_nodes = RouteNodes(settings)

            self.agent = build_agent(llm_nodes, action_nodes, route_nodes)

        # Передаём агент в data, чтобы он был доступен в хэндлерах
        data["agent"] = self.agent
        return await handler(event, data)
