from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlannerResult(BaseModel):
    task_brief: str = Field(description="Refined research objective.")
    search_query: str = Field(description="Search-ready query in English.")


class SupervisorDecision(BaseModel):
    next_agent: Literal["Researcher", "Writer", "FINISH"] = Field(
        description="Next assignee in the workflow."
    )
    reason: str = Field(description="Why this routing decision was made.")


class ResearcherAction(BaseModel):
    action: Literal[
        "search_web",
        "search_papers",
        "fetch_page",
        "store_note",
        "request_more_search",
        "finish",
    ]
    query: str | None = Field(default=None, description="Search query override.")
    url: str | None = Field(default=None, description="Target URL to fetch.")
    note: str | None = Field(default=None, description="New evidence note.")
    reason: str = Field(default="")


class WriterAction(BaseModel):
    action: Literal[
        "draft_report",
        "revise_report",
        "request_more_research",
        "finish",
    ]
    research_request: str | None = Field(default=None, description="Request for researcher.")
    focus: str | None = Field(default=None, description="Revision focus.")
    reason: str = Field(default="")

