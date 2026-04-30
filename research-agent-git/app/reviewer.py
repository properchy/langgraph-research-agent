from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - fallback for test environments
    class _Message:
        def __init__(self, content: str):
            self.content = content

    HumanMessage = _Message
    SystemMessage = _Message

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:  # pragma: no cover - fallback for test environments
    ChatDeepSeek = None  # type: ignore[assignment]


class ReviewResult(BaseModel):
    score: int = Field(description="Overall quality score from 1 to 10.")
    is_pass: bool = Field(description="Pass if report quality is sufficient.")
    feedback: str = Field(description="Actionable review feedback.")
    error_tags: list[Literal["evidence_missing", "structure_bad", "low_specificity", "unsupported_claim"]] = Field(
        default_factory=list
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatDeepSeek:
    if ChatDeepSeek is None:
        raise RuntimeError("langchain-deepseek is not installed.")
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    return ChatDeepSeek(model="deepseek-chat", temperature=0)


def _fallback_review(report: str) -> dict:
    score = 6
    tags: list[str] = []

    if len(report) < 500:
        score -= 2
        tags.append("low_specificity")
    if "# 摘要" not in report or "# 关键发现" not in report:
        score -= 1
        tags.append("structure_bad")
    citations = re.findall(r"\[(\d+)\]", report)
    if len(set(citations)) < 2:
        score -= 2
        tags.append("evidence_missing")
    if "根据" in report and not citations:
        tags.append("unsupported_claim")

    score = max(1, min(10, score))
    return {
        "review_score": score,
        "review_is_pass": score >= 7 and len(tags) <= 1,
        "review_feedback": "Fallback review used due to reviewer LLM failure.",
        "error_tags": sorted(set(tags)),
    }


def review_report(report: str, query: str) -> dict:
    try:
        llm = _get_llm()
        structured_llm = llm.with_structured_output(ReviewResult)
        response = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict report reviewer. Evaluate factual grounding, structure, and specificity. "
                        "Return structured output only."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original user query:\n{query}\n\n"
                        f"Report to review:\n{report}\n\n"
                        "Expected structure: 摘要, 关键发现, 证据来源, 风险/不确定性, 结论."
                    )
                ),
            ]
        )
        return {
            "review_score": response.score,
            "review_is_pass": response.is_pass,
            "review_feedback": response.feedback,
            "error_tags": response.error_tags,
        }
    except Exception:
        return _fallback_review(report)
