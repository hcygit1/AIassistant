"""FTS synchronization and rebuild operations for memory records."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable


class MemoryFtsIndex:
    def __init__(
        self,
        connection: sqlite3.Connection,
        tokenize: Callable[[str], str],
    ) -> None:
        self._connection = connection
        self._tokenize = tokenize

    def sync_chunk(self, chunk_id: str) -> None:
        row = self._connection.execute(
            "SELECT rowid, summary, content FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            return
        rowid = row[0]
        self._connection.execute(
            "DELETE FROM chunks_fts WHERE rowid = ?",
            (rowid,),
        )
        self._connection.execute(
            "INSERT INTO chunks_fts(rowid, summary, content) VALUES (?, ?, ?)",
            (
                rowid,
                self._tokenize(row[1] or ""),
                self._tokenize(row[2] or ""),
            ),
        )

    def sync_task(self, task_id: str) -> None:
        row = self._connection.execute(
            "SELECT rowid, title, summary FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        rowid = row[0]
        self._connection.execute(
            "DELETE FROM tasks_fts WHERE rowid = ?",
            (rowid,),
        )
        self._connection.execute(
            "INSERT INTO tasks_fts(rowid, title, summary) VALUES (?, ?, ?)",
            (
                rowid,
                self._tokenize(row[1] or ""),
                self._tokenize(row[2] or ""),
            ),
        )

    def sync_skill(self, skill_id: str) -> None:
        row = self._connection.execute(
            "SELECT rowid, name, description FROM skills WHERE id = ?",
            (skill_id,),
        ).fetchone()
        if not row:
            return
        rowid = row[0]
        self._connection.execute(
            """DELETE FROM skills_fts
            WHERE rowid NOT IN (SELECT rowid FROM skills)"""
        )
        self._connection.execute(
            "DELETE FROM skills_fts WHERE rowid = ?",
            (rowid,),
        )
        self._connection.execute(
            "INSERT INTO skills_fts(rowid, name, description) VALUES (?, ?, ?)",
            (
                rowid,
                self._tokenize(row[1] or ""),
                self._tokenize(row[2] or ""),
            ),
        )

    def rebuild(self) -> None:
        self._connection.execute("DELETE FROM chunks_fts")
        for row in self._connection.execute(
            "SELECT rowid, summary, content FROM chunks "
            "WHERE dedup_status IN ('active', 'orphaned')"
        ):
            self._connection.execute(
                "INSERT INTO chunks_fts(rowid, summary, content) VALUES (?, ?, ?)",
                (
                    row[0],
                    self._tokenize(row[1] or ""),
                    self._tokenize(row[2] or ""),
                ),
            )

        self._connection.execute("DELETE FROM tasks_fts")
        for row in self._connection.execute(
            "SELECT rowid, title, summary FROM tasks"
        ):
            self._connection.execute(
                "INSERT INTO tasks_fts(rowid, title, summary) VALUES (?, ?, ?)",
                (
                    row[0],
                    self._tokenize(row[1] or ""),
                    self._tokenize(row[2] or ""),
                ),
            )

        self._connection.execute("DELETE FROM skills_fts")
        for row in self._connection.execute(
            "SELECT rowid, name, description FROM skills"
        ):
            self._connection.execute(
                "INSERT INTO skills_fts(rowid, name, description) VALUES (?, ?, ?)",
                (
                    row[0],
                    self._tokenize(row[1] or ""),
                    self._tokenize(row[2] or ""),
                ),
            )

        self._connection.commit()
