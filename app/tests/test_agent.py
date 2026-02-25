from app.core.config import Settings
from app.core.llm import get_gigachat
from app.core.logging import get_logger
from app.schemas.agent_state import AgentState
from app.services.agent.action_nodes import ActionNodes
from app.services.agent.agent import BuildAgent
from app.services.agent.llm_nodes import GigaChatNodes
from app.services.agent.route_nodes import RouteNodes
from app.services.parse import generate_dataset_profile
from app.services.tools.viz_tools import get_tools_map

settings = Settings()
logger = get_logger(settings)
tools_map = get_tools_map()

llm = get_gigachat(settings, tools_map, logger)

llm_nodes = GigaChatNodes(llm, settings, logger)
action_nodes = ActionNodes(settings, tools_map)
route_nodes = RouteNodes(settings)

agent = BuildAgent(llm_nodes, action_nodes, route_nodes)

dataset = generate_dataset_profile("app/tests/customer_survey.csv", "1")

initial_state = AgentState(
    user_query="Проанализируй данные опроса клиентов и построй дашборд",
    user_data=[
        dataset
    ],  # Добавляем профиль датасета, полученный через generate_dataset_profile
    chat_history=[
        {
            "role": "user",
            "content": "Проанализируй данные опроса клиентов и построй дашборд",
        }
    ],
    artifacts=[],
    used_methods=[],
    answer=None,
    errors=[],
    iteration=0,
)

agent.invoke(initial_state)
