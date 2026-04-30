from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.planner import initialize_state_node, planner_node
from app.nodes.researcher import researcher_node
from app.nodes.supervisor import route_from_supervisor, supervisor_node
from app.nodes.writer import writer_node
from app.state import MultiAgentState

GRAPH_VERSION = "0.2.0"


def _make_guard_node(max_total_steps: int):
    def guard_node(state: MultiAgentState) -> dict:
        total_steps = int(state.get("total_steps", 0)) + 1
        if total_steps >= max_total_steps:
            return {
                "total_steps": total_steps,
                "finish_reason": "finish by max total steps",
                "supervisor_state": {
                    **state.get("supervisor_state", {}),
                    "current_assignee": "FINISH",
                    "supervisor_reason": "max total steps reached",
                },
            }
        return {"total_steps": total_steps}

    return guard_node


def _route_from_guard(state: MultiAgentState) -> str:
    if state.get("finish_reason") == "finish by max total steps":
        return END
    return "Supervisor"


def _reviewer_passthrough_node(state: MultiAgentState) -> dict:
    return {}


def build_graph(checkpointer=None, max_total_steps: int = 24, enable_reviewer_node: bool = False):
    guard_node = _make_guard_node(max_total_steps=max_total_steps)

    builder = StateGraph(MultiAgentState)
    builder.add_node("Planner", planner_node)
    builder.add_node("InitializeState", initialize_state_node)
    builder.add_node("Guard", guard_node)
    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("Researcher", researcher_node)
    builder.add_node("Writer", writer_node)

    if enable_reviewer_node:
        builder.add_node("Reviewer", _reviewer_passthrough_node)

    builder.add_edge(START, "Planner")
    builder.add_edge("Planner", "InitializeState")
    builder.add_edge("InitializeState", "Guard")
    builder.add_conditional_edges(
        "Guard",
        _route_from_guard,
        {
            "Supervisor": "Supervisor",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {
            "Researcher": "Researcher",
            "Writer": "Writer",
            END: END,
        },
    )
    builder.add_edge("Researcher", "Guard")
    if enable_reviewer_node:
        builder.add_edge("Writer", "Reviewer")
        builder.add_edge("Reviewer", "Guard")
    else:
        builder.add_edge("Writer", "Guard")

    graph = builder.compile(checkpointer=checkpointer)
    return graph

