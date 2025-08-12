# graph.py
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from agents import science_agent, maths_agent, physics_agent, general_agent
from router import router_node, route_query

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_graph():
    workflow = StateGraph(AgentState)

    # Add the router node (identity node that returns a dict)
    workflow.add_node("router", router_node)

    # Add agent nodes (these return dict state with messages)
    workflow.add_node("science_agent", science_agent)
    workflow.add_node("maths_agent", maths_agent)
    workflow.add_node("physics_agent", physics_agent)
    workflow.add_node("general_agent", general_agent)

    # Set router as entry point
    workflow.set_entry_point("router")

    # Use route_query (a separate function) to choose next node
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "science": "science_agent",
            "maths": "maths_agent",
            "physics": "physics_agent",
            "general": "general_agent"
        }
    )

    # After each agent end the graph
    workflow.add_edge("science_agent", END)
    workflow.add_edge("maths_agent", END)
    workflow.add_edge("physics_agent", END)
    workflow.add_edge("general_agent", END)

    return workflow.compile()
