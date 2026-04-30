from __future__ import annotations

import logging
import re

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - fallback for test environments
    class _Message:
        def __init__(self, content: str):
            self.content = content

    HumanMessage = _Message
    SystemMessage = _Message

from app.llm import get_llm
from app.memory import save_long_term_memory
from app.reviewer import review_report
from app.schemas import WriterAction
from app.state import DocItem, MultiAgentState

logger = logging.getLogger(__name__)

MAX_WRITER_TURNS = 4


def _as_text(response) -> str:
    content = response.content
    return content if isinstance(content, str) else str(content)


def format_docs(docs: list[DocItem]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(
            f"[{i}] 标题: {doc['title']}\n"
            f"URL: {doc['url']}\n"
            f"内容:\n{doc['content']}\n"
        )
    return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(blocks)


def summarize_sources(task_brief: str, docs: list[DocItem]) -> str:
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是研究总结助手。请基于材料提炼6-10条要点，每条尽量标注来源编号[1][2]。"
                )
            ),
            HumanMessage(
                content=(
                    f"研究任务:\n{task_brief}\n\n"
                    f"材料:\n{format_docs(docs)}"
                )
            ),
        ]
    )
    return _as_text(response)


def draft_report(task_brief: str, summary_points: str, docs: list[DocItem], research_memory: str) -> str:
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是研究报告写作助手。严格使用以下结构输出Markdown："
                    "1) 摘要 2) 关键发现 3) 证据来源 4) 风险/不确定性 5) 结论。"
                    "不要编造事实，尽量附上来源编号。"
                )
            ),
            HumanMessage(
                content=(
                    f"研究任务:\n{task_brief}\n\n"
                    f"长期记忆:\n{research_memory}\n\n"
                    f"要点:\n{summary_points}\n\n"
                    f"网页内容:\n{format_docs(docs)}"
                )
            ),
        ]
    )
    return _as_text(response)


def revise_report(task_brief: str, report_md: str, review_feedback: str, docs: list[DocItem]) -> str:
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是报告修订助手。必须保留固定结构：摘要/关键发现/证据来源/风险不确定性/结论。"
                    "根据反馈修正证据、结构和可读性。"
                )
            ),
            HumanMessage(
                content=(
                    f"任务:\n{task_brief}\n\n"
                    f"当前报告:\n{report_md}\n\n"
                    f"评审反馈:\n{review_feedback}\n\n"
                    f"可用证据:\n{format_docs(docs)}"
                )
            ),
        ]
    )
    return _as_text(response)


def citation_check(report_md: str, source_count: int) -> dict:
    cited = re.findall(r"\[(\d+)\]", report_md)
    unique_cited = sorted(set(cited))
    invalid_ids = [c for c in unique_cited if int(c) > source_count]
    lines = [line.strip() for line in report_md.splitlines() if line.strip()]
    unsupported_lines = [
        line for line in lines
        if any(token in line for token in ("表明", "证明", "说明", "预计", "结论"))
        and "[" not in line
    ]
    return {
        "num_citations": len(cited),
        "num_unique_sources": len(unique_cited),
        "invalid_source_ids": invalid_ids,
        "unsupported_claim_count": len(unsupported_lines),
    }


def _choose_writer_action(
    objective: str,
    summary_points: str,
    report_md: str,
    review_score: int,
    review_is_pass: bool,
    review_feedback: str,
    docs: list[DocItem],
) -> WriterAction:
    llm = get_llm()
    structured_llm = llm.with_structured_output(WriterAction)
    return structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是Writer。根据评审结果决定下一步：revise_report、request_more_research、finish。"
                )
            ),
            HumanMessage(
                content=(
                    f"目标:\n{objective}\n\n"
                    f"要点:\n{summary_points}\n\n"
                    f"报告:\n{report_md}\n\n"
                    f"评审: score={review_score}, pass={review_is_pass}\n"
                    f"feedback={review_feedback}\n\n"
                    f"证据:\n{format_docs(docs)}"
                )
            ),
        ]
    )


def writer_node(state: MultiAgentState) -> dict:
    wr = dict(state.get("writer_state", {}))
    rs = state.get("researcher_state", {})
    objective = wr.get("objective") or state.get("task_brief") or state["user_query"]
    docs = list(rs.get("docs", []))
    summary_points = wr.get("summary_points", "")
    report_md = wr.get("report_md", "")
    review_score = int(wr.get("review_score", 0) or 0)
    review_is_pass = bool(wr.get("review_is_pass", False))
    review_feedback = wr.get("review_feedback", "")
    review_error_tags = list(wr.get("review_error_tags", []))
    research_request = wr.get("research_request", "")
    revision_count = int(wr.get("revision_count", 0) or 0)
    tool_trace = list(wr.get("tool_trace", []))

    for turn in range(MAX_WRITER_TURNS):
        if not report_md:
            summary_points = summarize_sources(objective, docs)
            tool_trace.append("summarize_sources -> ok")
            report_md = draft_report(objective, summary_points, docs, state.get("relevant_memory", ""))
            tool_trace.append("draft_report -> ok")

        review = review_report(report_md, state["user_query"])
        review_score = int(review["review_score"])
        review_is_pass = bool(review["review_is_pass"])
        review_feedback = review["review_feedback"]
        review_error_tags = list(review.get("error_tags", []))

        citation_stats = citation_check(report_md, source_count=len(docs))
        if citation_stats["num_unique_sources"] < 2:
            review_is_pass = False
            if "evidence_missing" not in review_error_tags:
                review_error_tags.append("evidence_missing")
            review_feedback = review_feedback + "\nNeed at least 2 distinct cited sources."

        if citation_stats["unsupported_claim_count"] > 0:
            review_is_pass = False
            if "unsupported_claim" not in review_error_tags:
                review_error_tags.append("unsupported_claim")

        if review_is_pass:
            save_long_term_memory(
                user_query=state["user_query"],
                task_brief=state.get("task_brief", objective),
                summary_points=summary_points,
                report_md=report_md,
                review_score=review_score,
                review_is_pass=review_is_pass,
                review_feedback=review_feedback,
            )
            tool_trace.append(f"review passed with score {review_score}")
            break

        try:
            action = _choose_writer_action(
                objective=objective,
                summary_points=summary_points,
                report_md=report_md,
                review_score=review_score,
                review_is_pass=review_is_pass,
                review_feedback=review_feedback,
                docs=docs,
            )
        except Exception as exc:
            logger.warning("[writer] action selection failed: %s", exc)
            action = WriterAction(action="revise_report", reason=str(exc))

        tool_trace.append(f"turn {turn + 1}: {action.action} | {action.reason}")

        if action.action == "request_more_research":
            research_request = action.research_request or review_feedback or "Need more citable sources."
            tool_trace.append(f"request_more_research: {research_request}")
            break

        if action.action in {"draft_report", "revise_report"}:
            report_md = revise_report(objective, report_md, review_feedback, docs)
            revision_count += 1
            tool_trace.append("revise_report -> ok")
            continue

        if action.action == "finish":
            break

    needs_more_research = bool(research_request)
    completed = bool(report_md)

    return {
        "writer_state": {
            **wr,
            "objective": objective,
            "summary_points": summary_points,
            "report_md": report_md,
            "review_score": review_score,
            "review_is_pass": review_is_pass,
            "review_feedback": review_feedback,
            "review_error_tags": sorted(set(review_error_tags)),
            "research_request": research_request,
            "revision_count": revision_count,
            "citation_check": citation_check(report_md, source_count=len(docs)),
            "tool_trace": tool_trace,
            "completed": completed,
            "needs_more_research": needs_more_research,
        },
        "final_report": report_md,
    }
