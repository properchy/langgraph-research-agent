from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class SearchResult(TypedDict, total=False):
    title: str
    url: str
    snippet: str
    domain: str
    score: int
    is_blocked: bool
    filter_reason: str


class DocItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    domain: str
    meta: dict


class ResearchNote(TypedDict, total=False):
    claim: str
    source_url: str
    source_title: str
    confidence: Literal["low", "medium", "high"]


class SupervisorState(TypedDict, total=False):
    task_brief: str
    current_assignee: str | None
    supervisor_reason: str
    turn_count: int


class ResearcherState(TypedDict, total=False):
    objective: str
    search_query: str
    candidate_sources: list[SearchResult]
    docs: list[DocItem]
    notes: list[ResearchNote]
    seen_domains: list[str]
    seen_urls: list[str]
    seen_titles: list[str]
    tool_trace: list[str]
    completed: bool
    needs_more_research: bool


class WriterState(TypedDict, total=False):
    objective: str
    summary_points: str
    report_md: str
    review_score: int
    review_is_pass: bool
    review_feedback: str
    review_error_tags: list[str]
    research_request: str
    revision_count: int
    citation_check: dict
    tool_trace: list[str]
    completed: bool
    needs_more_research: bool


class MultiAgentState(TypedDict, total=False):
    user_query: str
    task_brief: str
    search_query: str
    relevant_memory: str
    query_rewrite_log: list[dict]
    supervisor_state: SupervisorState
    supervisor_trace: list[dict]
    researcher_state: ResearcherState
    writer_state: WriterState
    total_steps: int
    finish_reason: str
    final_report: str
    graph_version: str

