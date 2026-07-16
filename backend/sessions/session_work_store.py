"""Persistent store for system session work records."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionWorkRecord:
    id: str
    kind: str
    agent_id: str
    session_id: str
    content: str
    priority: int
    prompt_mode: str = "minimal"
    persist_role: str = "system"
    run_id: str | None = None
    status: str = "queued"
    recover_on_restart: bool = False
    created_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    last_error: str | None = None


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_work (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    priority INTEGER NOT NULL,
    prompt_mode TEXT NOT NULL DEFAULT 'minimal',
    persist_role TEXT NOT NULL DEFAULT 'system',
    run_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    recover_on_restart INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_work_status ON session_work(status);
CREATE INDEX IF NOT EXISTS idx_session_work_agent_session ON session_work(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_session_work_recoverable ON session_work(recover_on_restart, status);
"""


class SessionWorkStore:
    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            from config import DATA_DIR

            db_path = DATA_DIR / "session_work.db"
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(_CREATE_TABLE_SQL)
                conn.commit()
            finally:
                conn.close()

    def create_record(
        self,
        *,
        kind: str,
        agent_id: str,
        session_id: str,
        content: str,
        priority: int,
        prompt_mode: str = "minimal",
        persist_role: str = "system",
        run_id: str | None = None,
        recover_on_restart: bool = False,
    ) -> SessionWorkRecord:
        return SessionWorkRecord(
            id=f"sw-{uuid.uuid4().hex[:16]}",
            kind=kind,
            agent_id=agent_id,
            session_id=session_id,
            content=content,
            priority=priority,
            prompt_mode=prompt_mode,
            persist_role=persist_role,
            run_id=run_id,
            status="queued",
            recover_on_restart=recover_on_restart,
            created_at_ms=int(time.time() * 1000),
        )

    def insert(self, record: SessionWorkRecord) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO session_work
                    (id, kind, agent_id, session_id, content, priority, prompt_mode,
                     persist_role, run_id, status, recover_on_restart, created_at_ms,
                     started_at_ms, finished_at_ms, last_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.id,
                        record.kind,
                        record.agent_id,
                        record.session_id,
                        record.content,
                        record.priority,
                        record.prompt_mode,
                        record.persist_role,
                        record.run_id,
                        record.status,
                        1 if record.recover_on_restart else 0,
                        record.created_at_ms,
                        record.started_at_ms,
                        record.finished_at_ms,
                        record.last_error,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, work_id: str) -> SessionWorkRecord | None:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM session_work WHERE id = ?",
                    (work_id,),
                ).fetchone()
            finally:
                conn.close()
        return self._record_from_row(row) if row else None

    def mark_running(self, work_id: str) -> bool:
        now_ms = int(time.time() * 1000)
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """UPDATE session_work
                    SET status='running', started_at_ms=?, last_error=NULL
                    WHERE id=? AND status IN ('queued', 'running')""",
                    (now_ms, work_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def cancel_queued(self, work_id: str) -> bool:
        now_ms = int(time.time() * 1000)
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """UPDATE session_work
                    SET status='cancelled', finished_at_ms=?
                    WHERE id=? AND status='queued'""",
                    (now_ms, work_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def requeue_for_recovery(self, work_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """UPDATE session_work
                    SET status='queued', started_at_ms=NULL,
                        finished_at_ms=NULL, last_error=NULL
                    WHERE id=? AND recover_on_restart=1
                        AND status IN ('queued', 'running')""",
                    (work_id,),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def mark_done(self, work_id: str) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE session_work SET status='done', finished_at_ms=?, last_error=NULL WHERE id=?",
                    (now_ms, work_id),
                )
                conn.commit()
            finally:
                conn.close()

    def mark_failed(self, work_id: str, error: str | None = None) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE session_work SET status='failed', finished_at_ms=?, last_error=? WHERE id=?",
                    (now_ms, error, work_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_recoverable_pending(self) -> list[SessionWorkRecord]:
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM session_work
                    WHERE recover_on_restart = 1 AND status IN ('queued', 'running')
                    ORDER BY created_at_ms ASC"""
                ).fetchall()
            finally:
                conn.close()
        return [self._record_from_row(row) for row in rows]

    def query(
        self,
        *,
        kind: str | None = None,
        kinds: list[str] | None = None,
        status: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        run_id_prefix: str | None = None,
        exclude_run_id_prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionWorkRecord]:
        conditions: list[str] = []
        params: list[object] = []
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if run_id_prefix:
            conditions.append("run_id LIKE ?")
            params.append(f"{run_id_prefix}%")
        if exclude_run_id_prefix:
            conditions.append("(run_id IS NULL OR run_id NOT LIKE ?)")
            params.append(f"{exclude_run_id_prefix}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = (
            f"SELECT * FROM session_work WHERE {where} "
            "ORDER BY created_at_ms DESC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()

        return [self._record_from_row(row) for row in rows]

    def count(
        self,
        *,
        kind: str | None = None,
        kinds: list[str] | None = None,
        status: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        run_id_prefix: str | None = None,
        exclude_run_id_prefix: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[object] = []
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if run_id_prefix:
            conditions.append("run_id LIKE ?")
            params.append(f"{run_id_prefix}%")
        if exclude_run_id_prefix:
            conditions.append("(run_id IS NULL OR run_id NOT LIKE ?)")
            params.append(f"{exclude_run_id_prefix}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT COUNT(*) as cnt FROM session_work WHERE {where}"
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(sql, params).fetchone()
            finally:
                conn.close()
        return int(row["cnt"]) if row else 0

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> SessionWorkRecord:
        return SessionWorkRecord(
            id=row["id"],
            kind=row["kind"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            content=row["content"],
            priority=row["priority"],
            prompt_mode=row["prompt_mode"],
            persist_role=row["persist_role"],
            run_id=row["run_id"],
            status=row["status"],
            recover_on_restart=bool(row["recover_on_restart"]),
            created_at_ms=row["created_at_ms"],
            started_at_ms=row["started_at_ms"],
            finished_at_ms=row["finished_at_ms"],
            last_error=row["last_error"],
        )

    def prune_finished_older_than(
        self,
        *,
        older_than_ms: int,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> int:
        conditions = [
            "status IN ('done', 'failed', 'cancelled')",
            "finished_at_ms IS NOT NULL",
            "finished_at_ms < ?",
        ]
        params: list[object] = [older_than_ms]
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kinds)

        where = " AND ".join(conditions)
        with self._lock:
            conn = self._get_conn()
            try:
                if limit is None:
                    cursor = conn.execute(
                        f"DELETE FROM session_work WHERE {where}",
                        params,
                    )
                else:
                    cursor = conn.execute(
                        f"""DELETE FROM session_work
                            WHERE id IN (
                                SELECT id FROM session_work
                                WHERE {where}
                                ORDER BY finished_at_ms ASC
                                LIMIT ?
                            )""",
                        [*params, limit],
                    )
                conn.commit()
                return cursor.rowcount if cursor.rowcount != -1 else 0
            finally:
                conn.close()


session_work_store = SessionWorkStore()
