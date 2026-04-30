from __future__ import annotations

import logging

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - fallback for test environments
    class _Message:
        def __init__(self, content: str):
            self.content = content

    HumanMessage = _Message
    SystemMessage = _Message

from app.llm import get_llm
from app.memory import load_relevant_memory
from app.schemas import PlannerResult
from app.state import MultiAgentState

logger = logging.getLogger(__name__)


def _as_text(response) -> str:
    content = response.content
    return content if isinstance(content, str) else str(content)


def _fallback_plan(user_query: str) -> PlannerResult:
    return PlannerResult(
        task_brief=f"Research and summarize: {user_query}",
        search_query=user_query,
    )


def build_initial_state(task_brief: str, search_query: str) -> dict:
    return {
        "supervisor_state": {
            "task_brief": task_brief,
            "current_assignee": None,
            "supervisor_reason": "initialized",
            "turn_count": 0,
        },
        "researcher_state": {
            "objective": task_brief,
            "search_query": search_query,
            "candidate_sources": [],
            "docs": [],
            "notes": [],
            "seen_domains": [],
            "seen_urls": [],
            "seen_titles": [],
            "tool_trace": [],
            "completed": False,
            "needs_more_research": False,
        },
        "writer_state": {
            "objective": task_brief,
            "summary_points": "",
            "report_md": "",
            "review_score": 0,
            "review_is_pass": False,
            "review_feedback": "",
            "review_error_tags": [],
            "research_request": "",
            "revision_count": 0,
            "citation_check": {},
            "tool_trace": [],
            "completed": False,
            "needs_more_research": False,
        },
    }


def planner_node(state: MultiAgentState) -> dict:
    user_query = state["user_query"]
    relevant_memory = load_relevant_memory(user_query)
    llm = get_llm()
    structured_llm = llm.with_structured_output(PlannerResult)

    try:
        response = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a planner for a multi-agent research workflow. "
                        "Generate a precise task brief and a search query."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User query:\n{user_query}\n\n"
                        f"Relevant memory:\n{relevant_memory}\n\n"
                        "Return only structured output."
                    )
                ),
            ]
        )
    except Exception as exc:
        logger.warning("[planner] structured output failed: %s", exc)
        response = _fallback_plan(user_query)

    query_rewrite_log = list(state.get("query_rewrite_log", []))
    query_rewrite_log.append(
        {
            "raw_query": user_query,
            "task_brief": response.task_brief,
            "search_query": response.search_query,
        }
    )

    return {
        "task_brief": response.task_brief,
        "search_query": response.search_query,
        "relevant_memory": relevant_memory,
        "query_rewrite_log": query_rewrite_log,
    }


def initialize_state_node(state: MultiAgentState) -> dict:
    task_brief = state.get("task_brief") or f"Research and summarize: {state['user_query']}"
    search_query = state.get("search_query") or state["user_query"]
    return build_initial_state(task_brief=task_brief, search_query=search_query)
