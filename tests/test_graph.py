from __future__ import annotations

from app.nodes.planner import planner_node
from app.nodes.researcher import researcher_node
from app.nodes.supervisor import supervisor_node
from app.nodes.writer import writer_node


def test_planner_generates_task_brief():
    out = planner_node({"user_query": "比较LangGraph和AutoGen"})
    assert "task_brief" in out
    assert out["task_brief"]
    assert "search_query" in out


def test_supervisor_routes_researcher_when_no_docs():
    out = supervisor_node(
        {
            "user_query": "x",
            "task_brief": "y",
            "supervisor_state": {"turn_count": 0},
            "researcher_state": {"docs": []},
            "writer_state": {"report_md": "", "review_is_pass": False},
        }
    )
    assert out["supervisor_state"]["current_assignee"] == "Researcher"


def test_researcher_not_complete_without_docs(monkeypatch):
    monkeypatch.setattr("app.nodes.researcher.search_web", lambda *a, **k: [])
    out = researcher_node(
        {
            "user_query": "test",
            "task_brief": "test",
            "search_query": "test",
            "researcher_state": {
                "objective": "test",
                "search_query": "test",
                "candidate_sources": [],
                "docs": [],
                "notes": [],
            },
        }
    )
    assert out["researcher_state"]["completed"] is False


def test_writer_failed_review_requests_revision(monkeypatch):
    monkeypatch.setattr("app.nodes.writer.summarize_sources", lambda *a, **k: "- p1 [1]\n- p2 [2]")
    monkeypatch.setattr(
        "app.nodes.writer.draft_report",
        lambda *a, **k: "# 摘要\nbad\n# 关键发现\nbad\n# 证据来源\n[1]\n# 风险/不确定性\nunknown\n# 结论\nbad",
    )
    monkeypatch.setattr(
        "app.nodes.writer.review_report",
        lambda *a, **k: {
            "review_score": 5,
            "review_is_pass": False,
            "review_feedback": "Need stronger evidence and structure.",
            "error_tags": ["evidence_missing", "structure_bad"],
        },
    )
    monkeypatch.setattr(
        "app.nodes.writer._choose_writer_action",
        lambda **k: type("A", (), {"action": "request_more_research", "research_request": "Need 2+ quality sources.", "reason": "insufficient evidence"})(),
    )

    out = writer_node(
        {
            "user_query": "test",
            "task_brief": "test",
            "relevant_memory": "",
            "researcher_state": {
                "docs": [
                    {"title": "s1", "url": "u1", "content": "a" * 600},
                    {"title": "s2", "url": "u2", "content": "b" * 600},
                ]
            },
            "writer_state": {"objective": "test"},
        }
    )
    assert out["writer_state"]["needs_more_research"] is True

