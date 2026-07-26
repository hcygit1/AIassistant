"""Persistence boundary for structured session summaries."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import Any

from mem.models import SessionSummary


class SessionSummaryRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        now: Callable[[], float],
        new_id: Callable[[], str],
    ) -> None:
        self._connection = connection
        self._now = now
        self._new_id = new_id

    def upsert(
        self,
        session_id: str,
        agent_id: str,
        summary: dict[str, Any],
    ) -> SessionSummary:
        now = self._now()
        row = self._connection.execute(
            """SELECT id, version, created_at FROM session_summaries
            WHERE session_id = ? AND agent_id = ?""",
            (session_id, agent_id),
        ).fetchone()

        if row:
            summary_id = row["id"]
            version = row["version"] + 1
            created_at = row["created_at"] or 0.0
            self._connection.execute(
                """UPDATE session_summaries SET
                    version = ?, goal = ?, decisions = ?, progress = ?,
                    open_items = ?, entities = ?, user_preferences = ?,
                    raw_summary = ?, token_count = ?, updated_at = ?
                WHERE id = ?""",
                (
                    version,
                    summary.get("goal", ""),
                    self._serialize_list(summary, "decisions"),
                    summary.get("progress", ""),
                    self._serialize_list(summary, "open_items"),
                    self._serialize_list(summary, "entities"),
                    self._serialize_list(summary, "user_preferences"),
                    summary.get("raw_summary", ""),
                    summary.get("token_count", 0),
                    now,
                    summary_id,
                ),
            )
        else:
            summary_id = self._new_id()
            version = 1
            created_at = now
            self._connection.execute(
                """INSERT INTO session_summaries
                    (id, session_id, agent_id, version, goal, decisions, progress,
                     open_items, entities, user_preferences, raw_summary, token_count,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary_id,
                    session_id,
                    agent_id,
                    version,
                    summary.get("goal", ""),
                    self._serialize_list(summary, "decisions"),
                    summary.get("progress", ""),
                    self._serialize_list(summary, "open_items"),
                    self._serialize_list(summary, "entities"),
                    self._serialize_list(summary, "user_preferences"),
                    summary.get("raw_summary", ""),
                    summary.get("token_count", 0),
                    created_at,
                    now,
                ),
            )

        self._connection.commit()
        return SessionSummary(
            id=summary_id,
            session_id=session_id,
            agent_id=agent_id,
            version=version,
            goal=summary.get("goal", ""),
            decisions=self._serialize_list(summary, "decisions"),
            progress=summary.get("progress", ""),
            open_items=self._serialize_list(summary, "open_items"),
            entities=self._serialize_list(summary, "entities"),
            user_preferences=self._serialize_list(summary, "user_preferences"),
            raw_summary=summary.get("raw_summary", ""),
            token_count=summary.get("token_count", 0),
            created_at=created_at,
            updated_at=now,
        )

    def delete(self, session_id: str, agent_id: str) -> bool:
        deleted = self._connection.execute(
            "DELETE FROM session_summaries WHERE session_id = ? AND agent_id = ?",
            (session_id, agent_id),
        ).rowcount
        self._connection.commit()
        return deleted > 0

    def get(self, session_id: str, agent_id: str) -> SessionSummary | None:
        row = self._connection.execute(
            "SELECT * FROM session_summaries WHERE session_id = ? AND agent_id = ?",
            (session_id, agent_id),
        ).fetchone()
        if not row:
            return None
        return SessionSummary(
            id=row["id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            version=row["version"] or 1,
            goal=row["goal"] or "",
            decisions=row["decisions"] or "[]",
            progress=row["progress"] or "",
            open_items=row["open_items"] or "[]",
            entities=row["entities"] or "[]",
            user_preferences=row["user_preferences"] or "[]",
            raw_summary=row["raw_summary"] or "",
            token_count=row["token_count"] or 0,
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
        )

    @staticmethod
    def _serialize_list(summary: dict[str, Any], field: str) -> str:
        return json.dumps(summary.get(field, []), ensure_ascii=False)
