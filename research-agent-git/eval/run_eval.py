from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from app.graph import GRAPH_VERSION, build_graph
from app.memory import init_memory


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _fetch_success_rate(tool_trace: list[str]) -> float:
    fetch_events = [x for x in tool_trace if x.startswith("fetch_page(")]
    if not fetch_events:
        return 0.0
    ok = sum(1 for x in fetch_events if "-> ok" in x)
    return ok / len(fetch_events)


def run_eval(dataset_path: Path, output_csv: Path, db_path: Path, limit: int = 0) -> dict:
    dataset = _load_dataset(dataset_path)
    if limit > 0:
        dataset = dataset[:limit]

    checkpointer = init_memory(db_path)
    graph = build_graph(checkpointer=checkpointer, max_total_steps=24, enable_reviewer_node=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "type",
        "query",
        "latency_ms",
        "review_score",
        "review_is_pass",
        "num_sources",
        "fetch_success_rate",
        "completion",
        "finish_reason",
    ]

    rows = []
    for sample in dataset:
        start = time.perf_counter()
        thread_id = f"eval_{sample['id']}"
        result = graph.invoke(
            {"user_query": sample["query"], "graph_version": GRAPH_VERSION},
            config={"configurable": {"thread_id": thread_id}},
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        writer_state = result.get("writer_state", {})
        researcher_state = result.get("researcher_state", {})
        sources = len(researcher_state.get("docs", []))
        completion = bool(result.get("final_report")) and bool(writer_state.get("review_is_pass", False))
        row = {
            "id": sample["id"],
            "type": sample["type"],
            "query": sample["query"],
            "latency_ms": latency_ms,
            "review_score": int(writer_state.get("review_score", 0) or 0),
            "review_is_pass": bool(writer_state.get("review_is_pass", False)),
            "num_sources": sources,
            "fetch_success_rate": _fetch_success_rate(researcher_state.get("tool_trace", [])),
            "completion": completion,
            "finish_reason": result.get("finish_reason", ""),
        }
        rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    summary = {
        "total": total,
        "completion_rate": (sum(1 for x in rows if x["completion"]) / total) if total else 0.0,
        "avg_review_score": (sum(x["review_score"] for x in rows) / total) if total else 0.0,
        "avg_num_sources": (sum(x["num_sources"] for x in rows) / total) if total else 0.0,
        "fetch_success_rate": (sum(x["fetch_success_rate"] for x in rows) / total) if total else 0.0,
        "avg_latency": (sum(x["latency_ms"] for x in rows) / total) if total else 0.0,
        "output_csv": str(output_csv),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch eval for research-agent.")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--output", default="eval/results.csv")
    parser.add_argument("--db-path", default="data/research_memory_eval.db")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    summary = run_eval(
        dataset_path=Path(args.dataset),
        output_csv=Path(args.output),
        db_path=Path(args.db_path),
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

