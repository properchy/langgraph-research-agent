from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.graph import GRAPH_VERSION, build_graph
from app.memory import init_memory, log_run


def run_research(query: str, thread_id: str, output: Path, db_path: Path) -> dict:
    checkpointer = init_memory(db_path)
    graph = build_graph(checkpointer=checkpointer, max_total_steps=24, enable_reviewer_node=False)
    config = {"configurable": {"thread_id": thread_id}}

    start = time.perf_counter()
    result = graph.invoke({"user_query": query, "graph_version": GRAPH_VERSION}, config=config)
    latency_ms = int((time.perf_counter() - start) * 1000)

    output.parent.mkdir(parents=True, exist_ok=True)
    report_md = result.get("final_report", "No report generated.")
    output.write_text(report_md, encoding="utf-8")

    writer_state = result.get("writer_state", {})
    final_score = int(writer_state.get("review_score", 0) or 0)
    is_pass = bool(writer_state.get("review_is_pass", False))
    log_run(
        thread_id=thread_id,
        user_query=query,
        final_score=final_score,
        is_pass=is_pass,
        latency_ms=latency_ms,
    )

    summary = {
        "thread_id": thread_id,
        "graph_version": GRAPH_VERSION,
        "output": str(output),
        "latency_ms": latency_ms,
        "review_score": final_score,
        "review_is_pass": is_pass,
        "num_sources": len(result.get("researcher_state", {}).get("docs", [])),
        "finish_reason": result.get("finish_reason", ""),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent research workflow.")
    parser.add_argument("--query", required=True, help="User research query.")
    parser.add_argument("--thread-id", default="thread_demo", help="Thread id for LangGraph state.")
    parser.add_argument("--output", default="reports/demo.md", help="Report output path.")
    parser.add_argument("--db-path", default="data/research_memory.db", help="SQLite path.")
    args = parser.parse_args()

    summary = run_research(
        query=args.query,
        thread_id=args.thread_id,
        output=Path(args.output),
        db_path=Path(args.db_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

