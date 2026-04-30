from __future__ import annotations

import requests

from app.tools.web_fetch import clean_html, fetch_page
from app.tools.web_search import _is_blocked_source, _source_score


def test_is_blocked_source():
    assert _is_blocked_source("https://www.tripadvisor.com/Hotels")
    assert not _is_blocked_source("https://example.edu/post")


def test_source_score_prefers_content_sites():
    good = _source_score("https://example.edu/blog/agent-guide", "Guide", "official guide")
    bad = _source_score("https://shop.example.com/product?id=1", "Buy", "reserve now")
    assert good > bad


def test_clean_html_removes_noise():
    html = """
    <html><body>
      <script>alert(1)</script>
      <p>This is a long paragraph with enough length to pass the 40 character threshold for extraction.</p>
      <footer>footer text</footer>
    </body></html>
    """
    cleaned = clean_html(html)
    assert "alert" not in cleaned
    assert "threshold" in cleaned


def test_fetch_page_http_403(monkeypatch):
    class MockResponse:
        status_code = 403
        text = ""

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    def _mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("app.tools.web_fetch.requests.get", _mock_get)
    result = fetch_page("https://blocked.test/page")
    assert result["ok"] is False
    assert result["error_type"] == "http_403"

