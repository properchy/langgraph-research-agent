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
from app.schemas import ResearcherAction
from app.state import MultiAgentState, ResearchNote, SearchResult
from app.tools.web_fetch import fetch_document
from app.tools.web_search import search_web

logger = logging.getLogger(__name__)

MAX_RESEARCH_TURNS = 2
MIN_DOCS = 2
MIN_NOTES = 5


def _format_sources(sources: list[SearchResult]) -> str:
    if not sources:
        return "No sources."
    lines = []
    for idx, item in enumerate(sources, start=1):
        lines.append(f"[{idx}] {item['title']}\nURL: {item['url']}\nSnippet: {item.get('snippet', '')}")
    return "\n\n".join(lines)


def _choose_action(
    objective: str,
    search_query: str,
    candidate_sources: list[SearchResult],
    notes: list[ResearchNote],
    doc_count: int,
) -> ResearcherAction:
    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearcherAction)
    return structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are the Researcher. Focus on evidence collection and source diversity. "
                    "Allowed actions: search_web, search_papers, fetch_page, store_note, request_more_search, finish."
                )
            ),
            HumanMessage(
                content=(
                    f"objective={objective}\n"
                    f"search_query={search_query}\n"
                    f"doc_count={doc_count}\n"
                    f"candidate_sources=\n{_format_sources(candidate_sources)}\n\n"
                    f"notes_count={len(notes)}\n"
                    "Return the next action."
                )
            ),
        ]
    )


def _build_note(claim: str, source_url: str, source_title: str, confidence: str = "medium") -> ResearchNote:
    return {
        "claim": claim[:300],
        "source_url": source_url,
        "source_title": source_title[:200],
        "confidence": confidence if confidence in {"low", "medium", "high"} else "medium",
    }


def researcher_node(state: MultiAgentState) -> dict:
    rs = dict(state.get("researcher_state", {}))
    objective = rs.get("objective") or state.get("task_brief") or state["user_query"]
    search_query = rs.get("search_query") or state.get("search_query") or state["user_query"]
    candidate_sources = list(rs.get("candidate_sources", []))
    docs = list(rs.get("docs", []))
    notes = list(rs.get("notes", []))
    tool_trace = list(rs.get("tool_trace", []))
    seen_domains = set(rs.get("seen_domains", []))
    seen_urls = set(rs.get("seen_urls", []))
    seen_titles = set(rs.get("seen_titles", []))

    for turn in range(MAX_RESEARCH_TURNS):
        try:
            action = _choose_action(objective, search_query, candidate_sources, notes, len(docs))
        except Exception as exc:
            logger.warning("[researcher] action selection failed: %s", exc)
            action = ResearcherAction(action="finish", reason=str(exc))

        tool_trace.append(f"turn {turn + 1}: {action.action} | {action.reason}")

        if action.action in {"search_web", "request_more_search"}:
            search_query = action.query or search_query
            candidate_sources = search_web(search_query, mode="web")
            tool_trace.append(f"search_web({search_query}) -> {len(candidate_sources)}")
            continue

        if action.action == "search_papers":
            search_query = action.query or search_query
            candidate_sources = search_web(search_query, mode="paper")
            tool_trace.append(f"search_papers({search_query}) -> {len(candidate_sources)}")
            continue

        if action.action == "fetch_page":
            chosen_url = action.url
            if not chosen_url:
                for item in candidate_sources:
                    if item["url"] not in seen_urls and item.get("domain") not in seen_domains:
                        chosen_url = item["url"]
                        break
            if not chosen_url:
                tool_trace.append("fetch_page skipped: no unseen source")
                continue

            fetch_result = fetch_document(chosen_url)
            if fetch_result["ok"]:
                chosen_title = next(
                    (item["title"] for item in candidate_sources if item["url"] == chosen_url),
                    chosen_url,
                )
                domain = next(
                    (item.get("domain", "") for item in candidate_sources if item["url"] == chosen_url),
                    "",
                )
                if chosen_url in seen_urls or chosen_title in seen_titles:
                    tool_trace.append(f"fetch_page({chosen_url}) skipped duplicated content")
                    continue
                docs.append(
                    {
                        "title": chosen_title,
                        "url": chosen_url,
                        "content": fetch_result["content"],
                        "domain": domain,
                        "meta": {
                            "raw_length": fetch_result.get("raw_length", 0),
                            "cleaned_length": fetch_result.get("cleaned_length", 0),
                            "truncated": fetch_result.get("truncated", False),
                        },
                    }
                )
                seen_urls.add(chosen_url)
                seen_titles.add(chosen_title)
                if domain:
                    seen_domains.add(domain)
                snippet = fetch_result["content"][:180].replace("\n", " ")
                notes.append(_build_note(claim=snippet, source_url=chosen_url, source_title=chosen_title))
                tool_trace.append(f"fetch_page({chosen_url}) -> ok")
            else:
                tool_trace.append(
                    f"fetch_page({chosen_url}) -> {fetch_result.get('error_type')} | {fetch_result.get('reason')}"
                )
            continue

        if action.action == "store_note":
            if action.note:
                notes.append(
                    _build_note(
                        claim=action.note,
                        source_url="manual://note",
                        source_title="Researcher self-note",
                        confidence="low",
                    )
                )
            continue

        if action.action == "finish":
            if len(docs) >= MIN_DOCS and len(notes) >= MIN_NOTES:
                break
            if not candidate_sources:
                candidate_sources = search_web(search_query, mode="mixed")
                tool_trace.append("finish rejected: insufficient evidence, trigger mixed search")
            continue

    valid_docs = [doc for doc in docs if len(doc.get("content", "")) >= 300]
    completed = len(valid_docs) >= MIN_DOCS and len(notes) >= MIN_NOTES

    return {
        "researcher_state": {
            **rs,
            "objective": objective,
            "search_query": search_query,
            "candidate_sources": candidate_sources,
            "docs": docs,
            "notes": notes,
            "seen_domains": sorted(seen_domains),
            "seen_urls": sorted(seen_urls),
            "seen_titles": sorted(seen_titles),
            "tool_trace": tool_trace,
            "completed": completed,
            "needs_more_research": not completed,
        },
        "task_brief": objective,
        "search_query": search_query,
    }
