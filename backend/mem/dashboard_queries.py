"""Read-only dashboard projections for the memory API."""

from __future__ import annotations

import sqlite3
from typing import Any


class MemoryDashboardQueries:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get_stats(self) -> dict[str, Any]:
        connection = self._connection
        role_rows = connection.execute(
            """SELECT role, COUNT(*) AS c FROM chunks
            WHERE dedup_status='active' GROUP BY role"""
        ).fetchall()
        dedup_rows = connection.execute(
            "SELECT dedup_status, COUNT(*) AS c FROM chunks GROUP BY dedup_status"
        ).fetchall()
        time_range = connection.execute(
            "SELECT MIN(created_at) AS earliest, MAX(created_at) AS latest FROM chunks"
        ).fetchone()
        return {
            "totalChunks": connection.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE dedup_status='active'"
            ).fetchone()["c"],
            "totalTasks": connection.execute(
                "SELECT COUNT(*) AS c FROM tasks"
            ).fetchone()["c"],
            "completedTasks": connection.execute(
                "SELECT COUNT(*) AS c FROM tasks WHERE status='completed'"
            ).fetchone()["c"],
            "totalSkills": connection.execute(
                "SELECT COUNT(*) AS c FROM skills"
            ).fetchone()["c"],
            "totalSessions": connection.execute(
                "SELECT COUNT(DISTINCT session_key) AS c FROM chunks"
            ).fetchone()["c"],
            "roleBreakdown": {row["role"]: row["c"] for row in role_rows},
            "dedupBreakdown": {
                row["dedup_status"]: row["c"] for row in dedup_rows
            },
            "timeRange": {
                "earliest": time_range["earliest"],
                "latest": time_range["latest"],
            },
        }

    def list_tasks(
        self,
        *,
        status: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where = " WHERE status = ?" if status else ""
        params: list[Any] = [status] if status else []
        total = self._connection.execute(
            f"SELECT COUNT(*) AS c FROM tasks{where}", params
        ).fetchone()["c"]
        rows = self._connection.execute(
            f"SELECT * FROM tasks{where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        items = []
        for row in rows:
            chunk_count = self._connection.execute(
                """SELECT COUNT(*) AS c FROM chunks
                WHERE task_id=? AND dedup_status='active'""",
                (row["id"],),
            ).fetchone()["c"]
            items.append({
                "id": row["id"],
                "sessionKey": row["session_key"],
                "title": row["title"] or "",
                "summary": (row["summary"] or "")[:400],
                "status": row["status"],
                "startedAt": row["started_at"],
                "endedAt": row["ended_at"],
                "chunkCount": chunk_count,
            })
        return items, total

    def list_skills(self, *, status: str = "") -> list[dict[str, Any]]:
        if status:
            rows = self._connection.execute(
                "SELECT * FROM skills WHERE status=? ORDER BY updated_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM skills ORDER BY updated_at DESC"
            ).fetchall()
        return [{
            "id": row["id"],
            "name": row["name"],
            "description": (row["description"] or "")[:300],
            "version": row["version"],
            "status": row["status"],
            "qualityScore": row["quality_score"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        } for row in rows]

    def list_memories(
        self,
        *,
        limit: int = 40,
        offset: int = 0,
        session: str = "",
        role: str = "",
    ) -> tuple[list[dict[str, Any]], int]:
        conditions = ["dedup_status='active'"]
        params: list[Any] = []
        if session:
            conditions.append("session_key = ?")
            params.append(session)
        if role:
            conditions.append("role = ?")
            params.append(role)
        where = " WHERE " + " AND ".join(conditions)
        total = self._connection.execute(
            f"SELECT COUNT(*) AS c FROM chunks{where}", params
        ).fetchone()["c"]
        rows = self._connection.execute(
            f"""SELECT id, session_key, role, summary,
            substr(content,1,300) AS excerpt, task_id, created_at
            FROM chunks{where} ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
        return ([{
            "id": row["id"],
            "sessionKey": row["session_key"],
            "role": row["role"],
            "summary": row["summary"] or "",
            "excerpt": row["excerpt"] or "",
            "taskId": row["task_id"],
            "createdAt": row["created_at"],
        } for row in rows], total)
