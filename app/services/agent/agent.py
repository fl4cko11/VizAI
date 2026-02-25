from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.schemas.agent_state import AgentState
from app.services.agent.action_nodes import ActionNodes
from app.services.agent.llm_nodes import GigaChatNodes
from app.services.agent.route_nodes import RouteNodes


def BuildAgent(
    llm_nodes: GigaChatNodes, action_nodes: ActionNodes, route_nodes: RouteNodes
):

    builder = StateGraph(AgentState)
    builder.add_node("think", RunnableLambda(llm_nodes.think_node))
    builder.add_node("action", RunnableLambda(action_nodes.action_node))

    builder.add_edge(START, "think")
    builder.add_edge("think", "action")

    builder.add_conditional_edges(
        "action", route_nodes.route_after_action, {"think": "think", "end": END}
    )

    graph = builder.compile()

    return graph
