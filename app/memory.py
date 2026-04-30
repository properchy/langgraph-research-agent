from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - fallback for test environments
    class SqliteSaver:  # type: ignore[override]
        def __init__(self, conn):
            self.conn = conn

_CONN: sqlite3.Connection | None = None
_CHECKPOINTER: SqliteSaver | None = None


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT NOT NULL,
            source_query TEXT NOT NULL,
            memory_text TEXT NOT NULL,
            review_score INTEGER,
            review_is_pass INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            user_query TEXT,
            final_score INTEGER,
            is_pass INTEGER,
            latency_ms INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            mode TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            kept_count INTEGER NOT NULL,
            filtered_reasons TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def init_memory(db_path: str | Path) -> SqliteSaver:
    global _CONN, _CHECKPOINTER

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    _create_tables(conn)

    _CONN = conn
    _CHECKPOINTER = SqliteSaver(conn)
    return _CHECKPOINTER


def get_connection() -> sqlite3.Connection:
    global _CONN, _CHECKPOINTER
    if _CONN is None:
        try:
            init_memory(Path("data/research_memory.db"))
        except Exception:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.row_factory = sqlite3.Row
            _create_tables(conn)
            _CHECKPOINTER = SqliteSaver(conn)
            _CONN = conn
        if _CONN is None:
            raise RuntimeError("Memory not initialized. Call init_memory() first.")
    return _CONN


def get_checkpointer() -> SqliteSaver:
    if _CHECKPOINTER is None:
        raise RuntimeError("Memory not initialized. Call init_memory() first.")
    return _CHECKPOINTER


def _normalize_terms(query: str) -> list[str]:
    terms: list[str] = []
    raw_terms = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query)

    for term in raw_terms:
        normalized = term.lower()
        if normalized not in terms:
            terms.append(normalized)

        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", term):
            for size in (4, 3, 2):
                if len(term) < size:
                    continue
                for index in range(len(term) - size + 1):
                    chunk = term[index : index + size]
                    if chunk not in terms:
                        terms.append(chunk)
    return terms


def load_relevant_memory(query: str, limit: int = 3) -> str:
    conn = get_connection()
    terms = _normalize_terms(query)
    if not terms:
        return "No relevant long-term memory found."

    rows = conn.execute(
        """
        SELECT id, memory_type, source_query, memory_text, review_score, review_is_pass, created_at
        FROM long_term_memory
        ORDER BY id DESC
        LIMIT 100
        """
    ).fetchall()

    scored_rows: list[tuple[int, sqlite3.Row]] = []
    for row in rows:
        blob = f"{row['source_query']} {row['memory_text']}".lower()
        score = sum(1 for term in terms if term in blob)
        if score > 0:
            scored_rows.append((score, row))

    if not scored_rows:
        return "No directly related long-term memory."

    scored_rows.sort(
        key=lambda item: (item[0], item[1]["created_at"], item[1]["id"]),
        reverse=True,
    )

    snippets = []
    for score, row in scored_rows[:limit]:
        status = "pass" if row["review_is_pass"] else "fail"
        snippets.append(
            f"- score={score} | type={row['memory_type']} | review={row['review_score']} | "
            f"status={status} | query={row['source_query']} | memory={row['memory_text'][:300]}"
        )
    return "\n".join(snippets)


def save_long_term_memory(
    user_query: str,
    task_brief: str,
    summary_points: str,
    report_md: str,
    review_score: int,
    review_is_pass: bool,
    review_feedback: str,
) -> None:
    conn = get_connection()
    memory_text = (
        f"Task: {task_brief}\n"
        f"Summary: {summary_points}\n"
        f"Report digest: {report_md[:1200]}\n"
        f"Review feedback: {review_feedback}"
    )
    conn.execute(
        """
        INSERT INTO long_term_memory (
            memory_type, source_query, memory_text, review_score, review_is_pass
        ) VALUES (?, ?, ?, ?, ?)
        """,
        ("research_result", user_query, memory_text, review_score, int(review_is_pass)),
    )
    conn.commit()


def get_memory_stats() -> dict:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            AVG(review_score) AS avg_score,
            AVG(review_is_pass) AS pass_rate
        FROM long_term_memory
        """
    ).fetchone()
    return {
        "total_memories": int(row["total"] or 0),
        "avg_review_score": float(row["avg_score"] or 0.0),
        "pass_rate": float(row["pass_rate"] or 0.0),
    }


def clear_old_memory(days: int = 30) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        DELETE FROM long_term_memory
        WHERE created_at < datetime('now', ?)
        """,
        (f"-{days} day",),
    )
    conn.commit()
    return int(cursor.rowcount or 0)


def log_run(
    thread_id: str,
    user_query: str,
    final_score: int,
    is_pass: bool,
    latency_ms: int,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO run_log (thread_id, user_query, final_score, is_pass, latency_ms)
        VALUES (?, ?, ?, ?, ?)
        """,
        (thread_id, user_query, final_score, int(is_pass), latency_ms),
    )
    conn.commit()


def log_search(
    query: str,
    mode: str,
    candidate_count: int,
    kept_count: int,
    filtered_reasons: dict[str, int],
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO search_log (query, mode, candidate_count, kept_count, filtered_reasons)
        VALUES (?, ?, ?, ?, ?)
        """,
        (query, mode, candidate_count, kept_count, json.dumps(filtered_reasons, ensure_ascii=False)),
    )
    conn.commit()
