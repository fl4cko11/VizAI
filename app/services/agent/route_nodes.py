from app.core.config import Settings
from app.schemas.agent_state import AgentState


class RouteNodes:
    def __init__(self, settings: Settings):
        self.settings = settings

    def route_after_action(self, state: AgentState):
        if state.iteration <= self.settings.MAX_AGENT_ITTER:
            if state.errors:
                return "think"
            else:
                return "end"
        else:
            return "end"
