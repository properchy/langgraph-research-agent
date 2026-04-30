from __future__ import annotations

import re
from typing import TypedDict

import requests
from bs4 import BeautifulSoup


class FetchResult(TypedDict, total=False):
    ok: bool
    content: str
    error_type: str
    reason: str
    status_code: int
    raw_length: int
    cleaned_length: int
    truncated: bool
    attempts: int


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    chunks: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) >= 40:
            chunks.append(text)
    cleaned = "\n".join(chunks)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _http_error_type(status: int | None) -> str:
    if status == 403:
        return "http_403"
    if status == 401:
        return "http_401"
    if status == 404:
        return "http_404"
    if status == 429:
        return "http_429"
    if status and status >= 500:
        return "http_5xx"
    return "http_error"


def fetch_page(url: str, max_length: int = 5000, min_length: int = 300) -> FetchResult:
    headers = {
        "User-Agent": "Mozilla/5.0 (ResearchFlow/0.2)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            raw_text = resp.text
            cleaned = clean_html(raw_text)
            if not cleaned.strip():
                return {
                    "ok": False,
                    "content": "",
                    "error_type": "empty_after_clean",
                    "reason": "Content is empty after HTML cleanup.",
                    "status_code": resp.status_code,
                    "raw_length": len(raw_text),
                    "cleaned_length": 0,
                    "truncated": False,
                    "attempts": attempt,
                }
            if len(cleaned) < min_length:
                return {
                    "ok": False,
                    "content": "",
                    "error_type": "too_short",
                    "reason": "Content too short for reliable evidence.",
                    "status_code": resp.status_code,
                    "raw_length": len(raw_text),
                    "cleaned_length": len(cleaned),
                    "truncated": False,
                    "attempts": attempt,
                }
            truncated = len(cleaned) > max_length
            return {
                "ok": True,
                "content": cleaned[:max_length],
                "error_type": "",
                "reason": "ok",
                "status_code": resp.status_code,
                "raw_length": len(raw_text),
                "cleaned_length": len(cleaned),
                "truncated": truncated,
                "attempts": attempt,
            }
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            error_type = _http_error_type(status)
            # Retry only on 429 and 5xx.
            if status in {429} or (status is not None and status >= 500):
                if attempt < 3:
                    continue
            return {
                "ok": False,
                "content": "",
                "error_type": error_type,
                "reason": f"HTTP error: {status}",
                "status_code": int(status or 0),
                "raw_length": 0,
                "cleaned_length": 0,
                "truncated": False,
                "attempts": attempt,
            }
        except requests.RequestException as exc:
            if attempt < 3:
                continue
            return {
                "ok": False,
                "content": "",
                "error_type": "network_error",
                "reason": f"Network request failed: {exc}",
                "status_code": 0,
                "raw_length": 0,
                "cleaned_length": 0,
                "truncated": False,
                "attempts": attempt,
            }
        except Exception as exc:
            return {
                "ok": False,
                "content": "",
                "error_type": "parse_error",
                "reason": f"Parsing failed: {exc}",
                "status_code": 0,
                "raw_length": 0,
                "cleaned_length": 0,
                "truncated": False,
                "attempts": attempt,
            }

    return {
        "ok": False,
        "content": "",
        "error_type": "unknown_error",
        "reason": "Unknown fetch error.",
        "status_code": 0,
        "raw_length": 0,
        "cleaned_length": 0,
        "truncated": False,
        "attempts": 3,
    }


def fetch_pdf(url: str) -> FetchResult:
    return {
        "ok": False,
        "content": "",
        "error_type": "pdf_not_implemented",
        "reason": "PDF fetching is reserved for future implementation.",
        "status_code": 0,
        "raw_length": 0,
        "cleaned_length": 0,
        "truncated": False,
        "attempts": 0,
    }


def fetch_document(url: str) -> FetchResult:
    if url.lower().endswith(".pdf"):
        return fetch_pdf(url)
    return fetch_page(url)

