from __future__ import annotations

import logging
from urllib.parse import urlparse

from app.memory import log_search
from app.state import SearchResult

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for test environments
    DDGS = None  # type: ignore[assignment]

BLOCKED_DOMAINS = {
    "tripadvisor.com",
    "booking.com",
    "expedia.com",
    "klook.com",
    "mafengwo.cn",
}

PREFERRED_HINTS = (
    "blog",
    "article",
    "post",
    "guide",
    "review",
    "research",
    "news",
    "official",
    "gov",
    "edu",
    "arxiv",
    "paper",
)


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _is_blocked_source(url: str) -> bool:
    domain = _extract_domain(url)
    return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in BLOCKED_DOMAINS)


def _source_score(url: str, title: str, snippet: str) -> int:
    text = f"{url} {title} {snippet}".lower()
    if _is_blocked_source(url):
        return -100

    score = 0
    for hint in PREFERRED_HINTS:
        if hint in text:
            score += 2
    if any(token in url.lower() for token in ("/blog/", "/article/", "/guide/", "/post/", "/news/")):
        score += 3
    if any(token in url.lower() for token in ("?", "&", "utm_", "product", "booking", "reserve")):
        score -= 2
    if _extract_domain(url).endswith(".gov") or _extract_domain(url).endswith(".edu"):
        score += 4
    return score


def _build_queries(query: str, mode: str) -> list[str]:
    if mode == "paper":
        return [f"{query} arxiv paper survey", f"{query} scholarly article"]
    if mode == "mixed":
        return [f"{query} arxiv paper", f"{query} blog guide analysis", query]
    return [f"{query} blog article guide overview", query]


def search_web(query: str, mode: str = "web", max_results: int = 8) -> list[SearchResult]:
    if DDGS is None:
        logger.warning("[search_web] ddgs not installed, returning empty result")
        return []

    queries = _build_queries(query, mode)
    candidates: list[SearchResult] = []
    seen_urls: set[str] = set()
    filtered_reasons: dict[str, int] = {}
    raw_candidate_count = 0

    for one_query in queries:
        try:
            raw_results = list(DDGS().text(one_query, max_results=max_results))
        except Exception as exc:
            logger.warning("[search_web] query failed for %s: %s", one_query, exc)
            filtered_reasons["query_error"] = filtered_reasons.get("query_error", 0) + 1
            continue

        for item in raw_results:
            raw_candidate_count += 1
            url = item.get("href") or item.get("url")
            if not url:
                filtered_reasons["missing_url"] = filtered_reasons.get("missing_url", 0) + 1
                continue
            if url in seen_urls:
                filtered_reasons["duplicate_url"] = filtered_reasons.get("duplicate_url", 0) + 1
                continue

            title = item.get("title", "Untitled")
            snippet = item.get("body", "")
            domain = _extract_domain(url)
            blocked = _is_blocked_source(url)
            score = _source_score(url, title, snippet)
            filter_reason = "blocked_domain" if blocked else ""

            seen_urls.add(url)
            if blocked:
                filtered_reasons["blocked_domain"] = filtered_reasons.get("blocked_domain", 0) + 1
                continue

            candidates.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": domain,
                    "score": score,
                    "is_blocked": blocked,
                    "filter_reason": filter_reason,
                }
            )

        if len(candidates) >= 5:
            break

    candidates.sort(key=lambda item: int(item["score"]), reverse=True)
    kept = candidates[:5]

    try:
        log_search(
            query=query,
            mode=mode,
            candidate_count=raw_candidate_count,
            kept_count=len(kept),
            filtered_reasons=filtered_reasons,
        )
    except Exception as exc:
        logger.warning("[search_web] failed to log search: %s", exc)

    return kept


__all__ = ["search_web", "_extract_domain", "_is_blocked_source", "_source_score"]
