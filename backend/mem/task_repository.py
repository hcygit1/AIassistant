"""Persistence boundary for memory tasks and their chunk assignments."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from mem.chunk_repository import row_to_chunk
from mem.models import Chunk, Task, TaskStatus


class TaskRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        sync_fts: Callable[[str], None],
        now_ms: Callable[[], int],
    ) -> None:
        self._connection = connection
        self._sync_fts = sync_fts
        self._now_ms = now_ms

    def insert(self, task: Task) -> None:
        now = self._now_ms()
        if not task.started_at:
            task.started_at = now
        if not task.updated_at:
            task.updated_at = now
        self._connection.execute(
            """INSERT INTO tasks
               (id, session_key, owner, title, summary, boundary_summary,
                boundary_compacted_count, status, started_at, ended_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   session_key=excluded.session_key,
                   owner=excluded.owner,
                   title=excluded.title,
                   summary=excluded.summary,
                   boundary_summary=excluded.boundary_summary,
                   boundary_compacted_count=excluded.boundary_compacted_count,
                   status=excluded.status,
                   started_at=excluded.started_at,
                   ended_at=excluded.ended_at,
                   updated_at=excluded.updated_at""",
            (
                task.id,
                task.session_key,
                task.owner,
                task.title,
                task.summary,
                task.boundary_summary,
                task.boundary_compacted_count,
                task.status,
                task.started_at,
                task.ended_at,
                task.updated_at,
            ),
        )
        self._sync_fts(task.id)
        self._connection.commit()

    def get_active(self, owner: str = "agent:main") -> Task | None:
        row = self._connection.execute(
            """SELECT * FROM tasks
            WHERE owner=? AND status='active'
            ORDER BY started_at DESC LIMIT 1""",
            (owner,),
        ).fetchone()
        return row_to_task(row) if row else None

    def finalize(
        self,
        task_id: str,
        title: str,
        summary: str,
        status: TaskStatus = "completed",
    ) -> None:
        now = self._now_ms()
        self._connection.execute(
            """UPDATE tasks
            SET title=?, summary=?, status=?, ended_at=?, updated_at=?
            WHERE id=?""",
            (title, summary, status, now, now, task_id),
        )
        self._sync_fts(task_id)
        self._connection.commit()

    def get(self, task_id: str) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return row_to_task(row) if row else None

    def assign_chunks(self, chunk_ids: list[str], task_id: str) -> None:
        if not chunk_ids:
            return
        now = self._now_ms()
        for chunk_id in chunk_ids:
            self._connection.execute(
                "UPDATE chunks SET task_id=?, updated_at=? WHERE id=?",
                (task_id, now, chunk_id),
            )
        self._connection.commit()

    def get_all_active(self, owner: str = "agent:main") -> list[Task]:
        rows = self._connection.execute(
            """SELECT * FROM tasks
            WHERE owner=? AND status='active' ORDER BY started_at""",
            (owner,),
        ).fetchall()
        return [row_to_task(row) for row in rows]

    def get_active_by_session(
        self,
        session_key: str,
        owner: str = "agent:main",
    ) -> Task | None:
        row = self._connection.execute(
            """SELECT * FROM tasks
            WHERE session_key=? AND owner=? AND status='active'
            ORDER BY started_at DESC LIMIT 1""",
            (session_key, owner),
        ).fetchone()
        return row_to_task(row) if row else None

    def get_unassigned_chunks(
        self,
        session_key: str,
        owner: str | None = None,
    ) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE session_key=? AND task_id IS NULL "
            "AND dedup_status='active' ORDER BY created_at"
        )
        params: list[Any] = [session_key]
        if owner:
            sql = (
                "SELECT * FROM chunks WHERE session_key=? AND owner=? "
                "AND task_id IS NULL AND dedup_status='active' ORDER BY created_at"
            )
            params.append(owner)
        rows = self._connection.execute(sql, params).fetchall()
        return [row_to_chunk(row) for row in rows]

    def update(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = self._now_ms()
        set_clause = ", ".join(f"{field}=?" for field in fields)
        values = [*fields.values(), task_id]
        self._connection.execute(
            f"UPDATE tasks SET {set_clause} WHERE id=?",
            values,
        )
        if "title" in fields or "summary" in fields:
            self._sync_fts(task_id)
        self._connection.commit()


def row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        session_key=row["session_key"],
        owner=row["owner"] or "agent:main",
        title=row["title"] or "",
        summary=row["summary"] or "",
        boundary_summary=row["boundary_summary"] or "",
        boundary_compacted_count=row["boundary_compacted_count"] or 0,
        status=row["status"] or "active",
        started_at=row["started_at"] or 0,
        ended_at=row["ended_at"],
        updated_at=row["updated_at"] or 0,
    )
