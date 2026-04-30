from __future__ import annotations

import logging

try:
    from langgraph.graph import END
except ImportError:  # pragma: no cover - fallback for test environments
    END = "__end__"
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - fallback for test environments
    class _Message:
        def __init__(self, content: str):
            self.content = content

    HumanMessage = _Message
    SystemMessage = _Message

from app.llm import get_llm
from app.schemas import SupervisorDecision
from app.state import MultiAgentState

logger = logging.getLogger(__name__)

MAX_SUPERVISOR_TURNS = 2


def _decide_with_llm(state: MultiAgentState) -> tuple[str, str]:
    llm = get_llm()
    structured_llm = llm.with_structured_output(SupervisorDecision)
    researcher_state = state.get("researcher_state", {})
    writer_state = state.get("writer_state", {})

    has_docs = "yes" if researcher_state.get("docs") else "no"
    report_ready = "yes" if writer_state.get("report_md") else "no"
    writer_needs_research = "yes" if writer_state.get("needs_more_research") else "no"
    review_pass = "yes" if writer_state.get("review_is_pass") else "no"

    response = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a workflow supervisor. Route to Researcher, Writer, or FINISH only."
                )
            ),
            HumanMessage(
                content=(
                    f"user_query={state.get('user_query')}\n"
                    f"task_brief={state.get('task_brief')}\n"
                    f"has_docs={has_docs}\n"
                    f"report_ready={report_ready}\n"
                    f"writer_needs_research={writer_needs_research}\n"
                    f"review_pass={review_pass}\n"
                )
            ),
        ]
    )
    return response.next_agent, response.reason


def supervisor_node(state: MultiAgentState) -> dict:
    supervisor_state = dict(state.get("supervisor_state", {}))
    writer_state = state.get("writer_state", {})
    researcher_state = state.get("researcher_state", {})
    trace = list(state.get("supervisor_trace", []))
    turn_count = int(supervisor_state.get("turn_count", 0)) + 1
    finish_reason = state.get("finish_reason", "")

    if turn_count > MAX_SUPERVISOR_TURNS:
        next_agent = "FINISH"
        reason = "max supervisor turns reached"
        finish_reason = "finish by max turns"
    else:
        try:
            next_agent, reason = _decide_with_llm(state)
        except Exception as exc:
            logger.warning("[supervisor] routing failed: %s", exc)
            next_agent, reason = "FINISH", f"routing failed: {exc}"
            finish_reason = "finish by routing failure"

    if writer_state.get("needs_more_research"):
        next_agent = "Researcher"
        reason = writer_state.get("research_request") or "Writer requested more research."
    elif not researcher_state.get("docs"):
        next_agent = "Researcher"
        reason = "No docs collected yet."
    elif not writer_state.get("report_md") or not writer_state.get("review_is_pass"):
        next_agent = "Writer"
        reason = "Writer has no passing report yet."
    elif writer_state.get("review_is_pass"):
        next_agent = "FINISH"
        reason = "Report passed review."
        finish_reason = "finish by pass"

    trace.append(
        {
            "turn": turn_count,
            "summary": {
                "docs": len(researcher_state.get("docs", [])),
                "notes": len(researcher_state.get("notes", [])),
                "has_report": bool(writer_state.get("report_md")),
                "review_pass": bool(writer_state.get("review_is_pass")),
            },
            "decision": next_agent,
            "reason": reason,
        }
    )

    return {
        "supervisor_state": {
            **supervisor_state,
            "current_assignee": next_agent,
            "supervisor_reason": reason,
            "turn_count": turn_count,
        },
        "supervisor_trace": trace,
        "finish_reason": finish_reason,
    }


def route_from_supervisor(state: MultiAgentState) -> str:
    assignee = state.get("supervisor_state", {}).get("current_assignee")
    if assignee not in {"Researcher", "Writer", "FINISH"}:
        logger.warning("[route_from_supervisor] invalid assignee=%s", assignee)
        return END
    if assignee == "FINISH":
        return END
    return assignee
