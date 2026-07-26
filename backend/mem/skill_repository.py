"""Persistence boundary for evolved memory skills."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from mem.models import Skill


class SkillRepository:
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

    def insert(self, skill: Skill) -> None:
        now = self._now_ms()
        if not skill.created_at:
            skill.created_at = now
        if not skill.updated_at:
            skill.updated_at = now
        self._connection.execute(
            """INSERT OR REPLACE INTO skills
               (id, name, description, dir_path, version, status, installed,
                owner, visibility, quality_score, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                skill.id,
                skill.name,
                skill.description,
                skill.dir_path,
                skill.version,
                skill.status,
                skill.installed,
                skill.owner,
                skill.visibility,
                skill.quality_score,
                skill.created_at,
                skill.updated_at,
            ),
        )
        self._sync_fts(skill.id)
        self._connection.commit()

    def get(self, skill_id: str) -> Skill | None:
        row = self._connection.execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return row_to_skill(row) if row else None

    def update(self, skill_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = self._now_ms()
        set_clause = ", ".join(f"{field}=?" for field in fields)
        values = [*fields.values(), skill_id]
        self._connection.execute(
            f"UPDATE skills SET {set_clause} WHERE id=?",
            values,
        )
        if "name" in fields or "description" in fields:
            self._sync_fts(skill_id)
        self._connection.commit()


def row_to_skill(row: sqlite3.Row) -> Skill:
    return Skill(
        id=row["id"],
        name=row["name"],
        description=row["description"] or "",
        dir_path=row["dir_path"] or "",
        version=row["version"] or 1,
        status=row["status"] or "active",
        installed=row["installed"] or 0,
        owner=row["owner"] or "agent:main",
        visibility=row["visibility"] or "private",
        quality_score=row["quality_score"],
        created_at=row["created_at"] or 0,
        updated_at=row["updated_at"] or 0,
    )
