from app.core.config import Settings
from app.schemas.agent_state import AgentState
from app.services.agent.action_nodes import ActionNodes
from app.services.parse import generate_dataset_profile
from app.services.tools.viz_tools import get_tools_map

settings = Settings()

tools_map = get_tools_map()

action_nodes = ActionNodes(settings, tools_map)

dataset = generate_dataset_profile("app/tests/customer_survey.csv", "1")
test_state = AgentState(
    user_query="Проанализируй данные опроса клиентов и построй дашборд",
    user_data=[dataset],
    chat_history=[
        {
            "role": "user",
            "content": "Проанализируй данные опроса клиентов и построй дашборд",
        }
    ],
    artifacts=[],
    used_methods=[
        {
            "name": "create_histogram_tool",
            "args": {
                "file_path": "app/tests/sales_transactions.csv",
                "column_name": "total_amount",
                "title": "Распределение чеков",
            },
        }
    ],
    answer=None,
    errors=[],
    iteration=0,
)

result = action_nodes.action_node(test_state)

print(result)
