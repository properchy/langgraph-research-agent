from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:  # pragma: no cover - fallback for test environments
    ChatDeepSeek = None  # type: ignore[assignment]


class _UnavailableLLM:
    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        raise RuntimeError("LLM dependency not available. Install langchain-deepseek.")


@lru_cache(maxsize=1)
def get_llm() -> ChatDeepSeek:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    if ChatDeepSeek is None:
        return _UnavailableLLM()  # type: ignore[return-value]
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )
