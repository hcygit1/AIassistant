"""记忆存储层 — 单 SQLite 文件 + FTS5 + sqlite-vec ANN

Schema 与 docs/memory-system-refactor.md §3.1 一致:
  chunks / chunks_fts / vec_chunks
  tasks  / tasks_fts  / vec_tasks
  skills / skills_fts / vec_skills
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import sqlite_vec
from sqlite_vec import serialize_float32

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FTS5 tokenizer helper (jieba if available, else regex)
# ---------------------------------------------------------------------------

_HAS_JIEBA = False
try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except ImportError:
    pass


def _tokenize_for_fts(text: str) -> str:
    text_lower = text.lower()
    if _HAS_JIEBA:
        words = jieba.cut_for_search(text_lower)
        return " ".join(w.strip() for w in words if w.strip() and len(w.strip()) > 1)
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text_lower)
    return " ".join(t for t in tokens if len(t) > 1)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

DedupStatus = Literal["active", "duplicate", "merged", "orphaned"]
SummarySource = Literal["llm", "fallback"]
EmbeddingStatus = Literal["ok", "failed", "skipped"]
TaskStatus = Literal["active", "completed", "skipped"]
SkillStatus = Literal["active", "archived", "draft"]
SkillVisibility = Literal["private", "public"]


@dataclass
class Chunk:
    id: str
    session_key: str
    turn_id: str
    seq: int
    role: str
    content: str
    kind: str = "paragraph"
    summary: str = ""
    task_id: str | None = None
    skill_id: str | None = None
    owner: str = "agent:main"
    content_hash: str = ""
    dedup_status: DedupStatus = "active"
    dedup_target: str | None = None
    dedup_reason: str | None = None
    summary_source: SummarySource = "llm"
    embedding_status: EmbeddingStatus = "ok"
    embedding_error: str | None = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class Task:
    id: str
    session_key: str
    owner: str = "agent:main"
    title: str = ""
    summary: str = ""
    boundary_summary: str = ""
    boundary_compacted_count: int = 0
    status: TaskStatus = "active"
    started_at: int = 0
    ended_at: int | None = None
    updated_at: int = 0


@dataclass
class Skill:
    id: str
    name: str
    description: str = ""
    dir_path: str = ""
    version: int = 1
    status: SkillStatus = "active"
    installed: int = 0
    owner: str = "agent:main"
    visibility: SkillVisibility = "private"
    quality_score: float | None = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    summary: str = ""
    content_excerpt: str = ""
    role: str = ""
    session_key: str = ""
    task_id: str | None = None
    created_at: int = 0


@dataclass
class TaskSearchHit:
    task_id: str
    score: float
    title: str = ""
    summary: str = ""
    status: str = ""
    started_at: int = 0
    ended_at: int | None = None


@dataclass
class SkillSearchHit:
    skill_id: str
    score: float
    name: str = ""
    description: str = ""


@dataclass
class SessionSummary:
    id: str
    session_id: str
    agent_id: str
    version: int = 1
    goal: str = ""
    decisions: str = ""
    progress: str = ""
    open_items: str = ""
    entities: str = ""
    user_preferences: str = ""
    raw_summary: str = ""
    token_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


# ---------------------------------------------------------------------------
# MemStore
# ---------------------------------------------------------------------------


class MemStore:
    """SQLite schema + sqlite-vec + FTS5，所有 CRUD 方法。"""

    def __init__(self, db_path: str, dimensions: int = 1536):
        self.db_path = db_path
        self.dimensions = dimensions
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")

        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._init_schema()
        logger.info("MemStore initialized: %s (dim=%d)", db_path, dimensions)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        c = self._conn
        dim = self.dimensions

        c.executescript(f"""
            -- Chunks
            CREATE TABLE IF NOT EXISTS chunks (
                id           TEXT PRIMARY KEY,
                session_key  TEXT NOT NULL,
                turn_id      TEXT NOT NULL,
                seq          INTEGER NOT NULL,
                role         TEXT NOT NULL,
                content      TEXT NOT NULL,
                kind         TEXT NOT NULL DEFAULT 'paragraph',
                summary      TEXT NOT NULL DEFAULT '',
                task_id      TEXT,
                skill_id     TEXT,
                owner        TEXT NOT NULL DEFAULT 'agent:main',
                content_hash TEXT NOT NULL DEFAULT '',
                dedup_status TEXT NOT NULL DEFAULT 'active',
                dedup_target TEXT,
                dedup_reason TEXT,
                summary_source TEXT NOT NULL DEFAULT 'llm',
                embedding_status TEXT NOT NULL DEFAULT 'ok',
                embedding_error TEXT,
                created_at   INTEGER NOT NULL,
                updated_at   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_session
                ON chunks(session_key);
            CREATE INDEX IF NOT EXISTS idx_chunks_turn
                ON chunks(session_key, turn_id, seq);
            CREATE INDEX IF NOT EXISTS idx_chunks_created
                ON chunks(created_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_dedup
                ON chunks(session_key, role, content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_dedup_status
                ON chunks(dedup_status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_chunks_owner
                ON chunks(owner);
            CREATE INDEX IF NOT EXISTS idx_chunks_task
                ON chunks(task_id);

            -- Tasks
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                owner       TEXT NOT NULL DEFAULT 'agent:main',
                title       TEXT NOT NULL DEFAULT '',
                summary     TEXT NOT NULL DEFAULT '',
                boundary_summary TEXT NOT NULL DEFAULT '',
                boundary_compacted_count INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'active',
                started_at  INTEGER NOT NULL,
                ended_at    INTEGER,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_key);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner);

            -- Skills
            CREATE TABLE IF NOT EXISTS skills (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL UNIQUE,
                description   TEXT NOT NULL DEFAULT '',
                dir_path      TEXT NOT NULL DEFAULT '',
                version       INTEGER NOT NULL DEFAULT 1,
                status        TEXT NOT NULL DEFAULT 'active',
                installed     INTEGER NOT NULL DEFAULT 0,
                owner         TEXT NOT NULL DEFAULT 'agent:main',
                visibility    TEXT NOT NULL DEFAULT 'private',
                quality_score REAL,
                created_at    INTEGER NOT NULL,
                updated_at    INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);
            CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
            CREATE INDEX IF NOT EXISTS idx_skills_owner ON skills(owner);

            -- Session Summaries (结构化压缩摘要, 每个活跃 session 至多 1 行)
            CREATE TABLE IF NOT EXISTS session_summaries (
                id               TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                agent_id         TEXT NOT NULL,
                version          INTEGER DEFAULT 1,
                goal             TEXT,
                decisions        TEXT,
                progress         TEXT,
                open_items       TEXT,
                entities         TEXT,
                user_preferences TEXT,
                raw_summary      TEXT,
                token_count      INTEGER,
                created_at       REAL,
                updated_at       REAL,
                UNIQUE(session_id, agent_id)
            );
        """)

        self._ensure_chunk_columns()
        self._ensure_task_columns()

        # FTS5 virtual tables — 独立存储预分词文本，不绑定外部内容表
        for stmt in [
            """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                summary, content,
                tokenize='unicode61'
            )""",
            """CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
                title, summary,
                tokenize='unicode61'
            )""",
            """CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                name, description,
                tokenize='unicode61'
            )""",
        ]:
            try:
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass

        # sqlite-vec virtual tables
        for stmt in [
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dim}])",
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_tasks USING vec0(task_id TEXT PRIMARY KEY, embedding float[{dim}])",
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_skills USING vec0(skill_id TEXT PRIMARY KEY, embedding float[{dim}])",
        ]:
            c.execute(stmt)

        c.commit()

    # ------------------------------------------------------------------
    # FTS sync helpers — 预分词写入 FTS 索引
    # ------------------------------------------------------------------

    def _sync_chunk_fts(self, chunk_id: str) -> None:
        row = self._conn.execute(
            "SELECT rowid, summary, content FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            return
        rowid = row[0]
        self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
        self._conn.execute(
            "INSERT INTO chunks_fts(rowid, summary, content) VALUES (?, ?, ?)",
            (rowid, _tokenize_for_fts(row[1] or ""), _tokenize_for_fts(row[2] or "")),
        )

    def _sync_task_fts(self, task_id: str) -> None:
        row = self._conn.execute(
            "SELECT rowid, title, summary FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        rowid = row[0]
        self._conn.execute("DELETE FROM tasks_fts WHERE rowid = ?", (rowid,))
        self._conn.execute(
            "INSERT INTO tasks_fts(rowid, title, summary) VALUES (?, ?, ?)",
            (rowid, _tokenize_for_fts(row[1] or ""), _tokenize_for_fts(row[2] or "")),
        )

    def _sync_skill_fts(self, skill_id: str) -> None:
        row = self._conn.execute(
            "SELECT rowid, name, description FROM skills WHERE id = ?",
            (skill_id,),
        ).fetchone()
        if not row:
            return
        rowid = row[0]
        self._conn.execute("DELETE FROM skills_fts WHERE rowid = ?", (rowid,))
        self._conn.execute(
            "INSERT INTO skills_fts(rowid, name, description) VALUES (?, ?, ?)",
            (rowid, _tokenize_for_fts(row[1] or ""), _tokenize_for_fts(row[2] or "")),
        )

    def rebuild_fts_indexes(self) -> None:
        """重建所有 FTS 索引（迁移或修复时使用）。"""
        self._conn.execute("DELETE FROM chunks_fts")
        for row in self._conn.execute(
            "SELECT rowid, summary, content FROM chunks WHERE dedup_status IN ('active', 'orphaned')"
        ):
            self._conn.execute(
                "INSERT INTO chunks_fts(rowid, summary, content) VALUES (?, ?, ?)",
                (row[0], _tokenize_for_fts(row[1] or ""), _tokenize_for_fts(row[2] or "")),
            )

        self._conn.execute("DELETE FROM tasks_fts")
        for row in self._conn.execute("SELECT rowid, title, summary FROM tasks"):
            self._conn.execute(
                "INSERT INTO tasks_fts(rowid, title, summary) VALUES (?, ?, ?)",
                (row[0], _tokenize_for_fts(row[1] or ""), _tokenize_for_fts(row[2] or "")),
            )

        self._conn.execute("DELETE FROM skills_fts")
        for row in self._conn.execute("SELECT rowid, name, description FROM skills"):
            self._conn.execute(
                "INSERT INTO skills_fts(rowid, name, description) VALUES (?, ?, ?)",
                (row[0], _tokenize_for_fts(row[1] or ""), _tokenize_for_fts(row[2] or "")),
            )

        self._conn.commit()

    def _ensure_chunk_columns(self) -> None:
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        additions = {
            "summary_source": "ALTER TABLE chunks ADD COLUMN summary_source TEXT NOT NULL DEFAULT 'llm'",
            "embedding_status": "ALTER TABLE chunks ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'ok'",
            "embedding_error": "ALTER TABLE chunks ADD COLUMN embedding_error TEXT",
        }
        for name, stmt in additions.items():
            if name not in cols:
                self._conn.execute(stmt)
        self._conn.commit()
        logger.info("FTS indexes rebuilt with pre-tokenized content")

    def _ensure_task_columns(self) -> None:
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        additions = {
            "boundary_summary": "ALTER TABLE tasks ADD COLUMN boundary_summary TEXT NOT NULL DEFAULT ''",
            "boundary_compacted_count": "ALTER TABLE tasks ADD COLUMN boundary_compacted_count INTEGER NOT NULL DEFAULT 0",
        }
        for name, stmt in additions.items():
            if name not in cols:
                self._conn.execute(stmt)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Chunks — Write
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> None:
        if not chunk.content_hash:
            chunk.content_hash = _content_hash(chunk.content)
        now = _now_ms()
        if not chunk.created_at:
            chunk.created_at = now
        if not chunk.updated_at:
            chunk.updated_at = now

        self._conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, session_key, turn_id, seq, role, content, kind, summary,
                task_id, skill_id, owner, content_hash,
                dedup_status, dedup_target, dedup_reason,
                summary_source, embedding_status, embedding_error,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                chunk.id,
                chunk.session_key,
                chunk.turn_id,
                chunk.seq,
                chunk.role,
                chunk.content,
                chunk.kind,
                chunk.summary,
                chunk.task_id,
                chunk.skill_id,
                chunk.owner,
                chunk.content_hash,
                chunk.dedup_status,
                chunk.dedup_target,
                chunk.dedup_reason,
                chunk.summary_source,
                chunk.embedding_status,
                chunk.embedding_error,
                chunk.created_at,
                chunk.updated_at,
            ),
        )
        self._sync_chunk_fts(chunk.id)
        self._conn.commit()

    def update_chunk_summary(
        self,
        chunk_id: str,
        summary: str,
        *,
        summary_source: SummarySource | None = None,
    ) -> None:
        if summary_source is None:
            self._conn.execute(
                "UPDATE chunks SET summary = ?, updated_at = ? WHERE id = ?",
                (summary, _now_ms(), chunk_id),
            )
        else:
            self._conn.execute(
                "UPDATE chunks SET summary = ?, summary_source = ?, updated_at = ? WHERE id = ?",
                (summary, summary_source, _now_ms(), chunk_id),
            )
        self._sync_chunk_fts(chunk_id)
        self._conn.commit()

    def update_chunk_embedding_status(
        self,
        chunk_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE chunks SET embedding_status = ?, embedding_error = ?, updated_at = ? WHERE id = ?",
            (status, error, _now_ms(), chunk_id),
        )
        self._conn.commit()

    def mark_dedup_status(
        self,
        chunk_id: str,
        status: DedupStatus,
        target: str | None = None,
        reason: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE chunks SET dedup_status=?, dedup_target=?, dedup_reason=?, updated_at=? WHERE id=?",
            (status, target, reason, _now_ms(), chunk_id),
        )
        self._conn.commit()

    def orphan_chunk(self, chunk_id: str, reason: str | None = None) -> None:
        self._conn.execute(
            "UPDATE chunks SET dedup_status='orphaned', task_id=NULL, dedup_reason=?, updated_at=? WHERE id=?",
            (reason, _now_ms(), chunk_id),
        )
        self._conn.commit()

    def find_active_chunk_by_hash(self, content: str, owner: str) -> str | None:
        h = _content_hash(content)
        row = self._conn.execute(
            "SELECT id FROM chunks WHERE content_hash=? AND dedup_status='active' AND owner=? LIMIT 1",
            (h, owner),
        ).fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------------
    # Chunks — Read
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_task(self, task_id: str, limit: int | None = None) -> list[Chunk]:
        sql = "SELECT * FROM chunks WHERE task_id=? AND dedup_status='active' ORDER BY created_at"
        params: list[Any] = [task_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunks_for_embedding_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE dedup_status='active' AND embedding_status='failed' "
            "ORDER BY updated_at LIMIT ?"
        )
        params: list[Any] = [limit]
        if owner:
            sql = (
                "SELECT * FROM chunks WHERE dedup_status='active' AND embedding_status='failed' "
                "AND owner=? ORDER BY updated_at LIMIT ?"
            )
            params = [owner, limit]
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunks_for_summary_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE dedup_status='active' AND summary_source='fallback' "
            "ORDER BY updated_at LIMIT ?"
        )
        params: list[Any] = [limit]
        if owner:
            sql = (
                "SELECT * FROM chunks WHERE dedup_status='active' AND summary_source='fallback' "
                "AND owner=? ORDER BY updated_at LIMIT ?"
            )
            params = [owner, limit]
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunks_in_range(
        self,
        session_key: str,
        turn_id: str,
        seq: int,
        window: int = 2,
        owner: str | None = None,
    ) -> list[Chunk]:
        """Get neighboring chunks for timeline display."""
        params: list[Any] = [session_key]
        sql = "SELECT * FROM chunks WHERE session_key = ?"
        if owner:
            sql += " AND owner = ?"
            params.append(owner)
        sql += " ORDER BY created_at, seq"
        rows = self._conn.execute(sql, params).fetchall()

        target_idx = -1
        for i, r in enumerate(rows):
            if r["turn_id"] == turn_id and r["seq"] == seq:
                target_idx = i
                break
        if target_idx == -1:
            return []

        radius = window * 3
        start = max(0, target_idx - radius)
        end = min(len(rows), target_idx + radius + 1)
        return [_row_to_chunk(rows[i]) for i in range(start, end)]

    # ------------------------------------------------------------------
    # Embeddings (sqlite-vec)
    # ------------------------------------------------------------------

    def upsert_chunk_embedding(self, chunk_id: str, vec: list[float]) -> None:
        blob = serialize_float32(vec)
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, blob),
        )
        self._conn.commit()

    def delete_chunk_embedding(self, chunk_id: str) -> None:
        self._conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.commit()

    def upsert_task_embedding(self, task_id: str, vec: list[float]) -> None:
        blob = serialize_float32(vec)
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_tasks(task_id, embedding) VALUES (?, ?)",
            (task_id, blob),
        )
        self._conn.commit()

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
        blob = serialize_float32(vec)
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_skills(skill_id, embedding) VALUES (?, ?)",
            (skill_id, blob),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # ANN Search (sqlite-vec)
    # ------------------------------------------------------------------

    def ann_search_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        """ANN search on vec_chunks, join back to chunks for metadata."""
        blob = serialize_float32(query_vec)
        rows = self._conn.execute(
            """SELECT v.chunk_id, v.distance,
                      c.summary, c.content, c.role, c.session_key,
                      c.task_id, c.created_at
               FROM vec_chunks v
               JOIN chunks c ON c.id = v.chunk_id
               WHERE v.embedding MATCH ? AND k = ?
                 AND c.dedup_status = 'active'
               ORDER BY v.distance""",
            (blob, top_k * 2),
        ).fetchall()

        hits: list[SearchHit] = []
        for r in rows:
            if exclude_session and r["session_key"] == exclude_session:
                continue
            score = 1.0 - r["distance"] if r["distance"] is not None else 0.0
            hits.append(
                SearchHit(
                    chunk_id=r["chunk_id"],
                    score=score,
                    summary=r["summary"] or "",
                    content_excerpt=r["content"][:300] if r["content"] else "",
                    role=r["role"] or "",
                    session_key=r["session_key"] or "",
                    task_id=r["task_id"],
                    created_at=r["created_at"] or 0,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def ann_search_tasks(
        self, query_vec: list[float], top_k: int = 5,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        blob = serialize_float32(query_vec)
        sql = """SELECT v.task_id, v.distance,
                        t.title, t.summary, t.status, t.started_at, t.ended_at
                 FROM vec_tasks v
                 JOIN tasks t ON t.id = v.task_id
                 WHERE v.embedding MATCH ? AND k = ?
                   AND t.status = 'completed'"""
        params: list[Any] = [blob, top_k]
        if owner:
            sql += " AND t.owner = ?"
            params.append(owner)
        sql += " ORDER BY v.distance"
        rows = self._conn.execute(sql, params).fetchall()

        return [
            TaskSearchHit(
                task_id=r["task_id"],
                score=1.0 - (r["distance"] or 0.0),
                title=r["title"] or "",
                summary=r["summary"] or "",
                status=r["status"] or "",
                started_at=r["started_at"] or 0,
                ended_at=r["ended_at"],
            )
            for r in rows
        ]

    def ann_search_skills(
        self, query_vec: list[float], top_k: int = 5,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        blob = serialize_float32(query_vec)
        sql = """SELECT v.skill_id, v.distance,
                        s.name, s.description
                 FROM vec_skills v
                 JOIN skills s ON s.id = v.skill_id
                 WHERE v.embedding MATCH ? AND k = ?
                   AND s.status IN ('active', 'draft')"""
        params: list[Any] = [blob, top_k]
        if owner:
            sql += " AND s.owner = ?"
            params.append(owner)
        sql += " ORDER BY v.distance"
        rows = self._conn.execute(sql, params).fetchall()

        return [
            SkillSearchHit(
                skill_id=r["skill_id"],
                score=1.0 - (r["distance"] or 0.0),
                name=r["name"] or "",
                description=r["description"] or "",
            )
            for r in rows
        ]

    def ann_dedup_candidates(
        self,
        query_vec: list[float],
        threshold: float,
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[SearchHit]:
        """Find top-K similar active chunks above threshold for dedup."""
        blob = serialize_float32(query_vec)
        sql = """SELECT v.chunk_id, v.distance,
                        c.summary, c.content, c.role, c.session_key,
                        c.task_id, c.created_at
                 FROM vec_chunks v
                 JOIN chunks c ON c.id = v.chunk_id
                 WHERE v.embedding MATCH ? AND k = ?
                   AND c.dedup_status = 'active'"""
        params: list[Any] = [blob, top_k * 3]
        if owner:
            sql += " AND c.owner = ?"
            params.append(owner)
        sql += " ORDER BY v.distance"

        rows = self._conn.execute(sql, params).fetchall()
        hits: list[SearchHit] = []
        for r in rows:
            score = 1.0 - (r["distance"] or 0.0)
            if score < threshold:
                continue
            hits.append(
                SearchHit(
                    chunk_id=r["chunk_id"],
                    score=score,
                    summary=r["summary"] or "",
                    content_excerpt=r["content"][:300] if r["content"] else "",
                    role=r["role"] or "",
                    session_key=r["session_key"] or "",
                    task_id=r["task_id"],
                    created_at=r["created_at"] or 0,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------------
    # FTS Search
    # ------------------------------------------------------------------

    def fts_search_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []
        try:
            sql = """SELECT c.id as chunk_id, rank,
                            c.summary, c.content, c.role, c.session_key,
                            c.task_id, c.created_at
                     FROM chunks_fts f
                     JOIN chunks c ON c.rowid = f.rowid
                     WHERE chunks_fts MATCH ?
                       AND c.dedup_status = 'active'"""
            params: list[Any] = [sanitized]
            if exclude_session:
                sql += " AND c.session_key != ?"
                params.append(exclude_session)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit * 2)

            rows = self._conn.execute(sql, params).fetchall()
            if not rows:
                return []

            max_abs = max(abs(r["rank"]) for r in rows) or 1.0
            hits: list[SearchHit] = []
            for r in rows:
                hits.append(
                    SearchHit(
                        chunk_id=r["chunk_id"],
                        score=abs(r["rank"]) / max_abs,
                        summary=r["summary"] or "",
                        content_excerpt=r["content"][:300] if r["content"] else "",
                        role=r["role"] or "",
                        session_key=r["session_key"] or "",
                        task_id=r["task_id"],
                        created_at=r["created_at"] or 0,
                    )
                )
            return hits[:limit]
        except sqlite3.OperationalError:
            logger.warning("FTS query failed for: %s", sanitized)
            return []

    def fts_search_tasks(
        self, query: str, limit: int = 10,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []
        try:
            sql = """SELECT t.id as task_id, rank,
                            t.title, t.summary, t.status, t.started_at, t.ended_at
                     FROM tasks_fts f
                     JOIN tasks t ON t.rowid = f.rowid
                     WHERE tasks_fts MATCH ?
                       AND t.status = 'completed'"""
            params: list[Any] = [sanitized]
            if owner:
                sql += " AND t.owner = ?"
                params.append(owner)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()

            if not rows:
                return []
            max_abs = max(abs(r["rank"]) for r in rows) or 1.0
            return [
                TaskSearchHit(
                    task_id=r["task_id"],
                    score=abs(r["rank"]) / max_abs,
                    title=r["title"] or "",
                    summary=r["summary"] or "",
                    status=r["status"] or "",
                    started_at=r["started_at"] or 0,
                    ended_at=r["ended_at"],
                )
                for r in rows
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS task query failed for: %s", sanitized)
            return []

    # ------------------------------------------------------------------
    # Tasks — CRUD
    # ------------------------------------------------------------------

    def insert_task(self, task: Task) -> None:
        now = _now_ms()
        if not task.started_at:
            task.started_at = now
        if not task.updated_at:
            task.updated_at = now
        self._conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, session_key, owner, title, summary, boundary_summary, boundary_compacted_count,
                status, started_at, ended_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
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
        self._sync_task_fts(task.id)
        self._conn.commit()

    def get_active_task(self, owner: str = "agent:main") -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE owner=? AND status='active' ORDER BY started_at DESC LIMIT 1",
            (owner,),
        ).fetchone()
        return _row_to_task(row) if row else None

    def finalize_task(
        self, task_id: str, title: str, summary: str, status: TaskStatus = "completed"
    ) -> None:
        now = _now_ms()
        self._conn.execute(
            "UPDATE tasks SET title=?, summary=?, status=?, ended_at=?, updated_at=? WHERE id=?",
            (title, summary, status, now, now, task_id),
        )
        self._sync_task_fts(task_id)
        self._conn.commit()

    def get_task(self, task_id: str) -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _row_to_task(row) if row else None

    def assign_chunks_to_task(self, chunk_ids: list[str], task_id: str) -> None:
        if not chunk_ids:
            return
        now = _now_ms()
        for cid in chunk_ids:
            self._conn.execute(
                "UPDATE chunks SET task_id=?, updated_at=? WHERE id=?",
                (task_id, now, cid),
            )
        self._conn.commit()

    def get_all_active_tasks(self, owner: str = "agent:main") -> list[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE owner=? AND status='active' ORDER BY started_at",
            (owner,),
        ).fetchall()
        return [_row_to_task(r) for r in rows]

    def get_active_task_by_session(
        self, session_key: str, owner: str = "agent:main"
    ) -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE session_key=? AND owner=? AND status='active' "
            "ORDER BY started_at DESC LIMIT 1",
            (session_key, owner),
        ).fetchone()
        return _row_to_task(row) if row else None

    def get_unassigned_chunks(
        self, session_key: str, owner: str | None = None,
    ) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE session_key=? AND task_id IS NULL "
            "AND dedup_status='active' ORDER BY created_at"
        )
        params: list[Any] = [session_key]
        if owner:
            sql = (
                "SELECT * FROM chunks WHERE session_key=? AND owner=? AND task_id IS NULL "
                "AND dedup_status='active' ORDER BY created_at"
            )
            params.append(owner)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now_ms()
        set_clause = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [task_id]
        self._conn.execute(
            f"UPDATE tasks SET {set_clause} WHERE id=?", vals
        )
        if "title" in fields or "summary" in fields:
            self._sync_task_fts(task_id)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Skills — CRUD
    # ------------------------------------------------------------------

    def insert_skill(self, skill: Skill) -> None:
        now = _now_ms()
        if not skill.created_at:
            skill.created_at = now
        if not skill.updated_at:
            skill.updated_at = now
        self._conn.execute(
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
        self._sync_skill_fts(skill.id)
        self._conn.commit()

    def get_skill(self, skill_id: str) -> Skill | None:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return _row_to_skill(row) if row else None

    def update_skill(self, skill_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now_ms()
        set_clause = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [skill_id]
        self._conn.execute(
            f"UPDATE skills SET {set_clause} WHERE id=?", vals
        )
        if "name" in fields or "description" in fields:
            self._sync_skill_fts(skill_id)
        self._conn.commit()

    def fts_search_skills(
        self, query: str, limit: int = 10,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []
        try:
            sql = """SELECT s.id as skill_id, rank,
                            s.name, s.description
                     FROM skills_fts f
                     JOIN skills s ON s.rowid = f.rowid
                     WHERE skills_fts MATCH ?
                       AND s.status IN ('active', 'draft')"""
            params: list[Any] = [sanitized]
            if owner:
                sql += " AND s.owner = ?"
                params.append(owner)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()

            if not rows:
                return []
            max_abs = max(abs(r["rank"]) for r in rows) or 1.0
            return [
                SkillSearchHit(
                    skill_id=r["skill_id"],
                    score=abs(r["rank"]) / max_abs,
                    name=r["name"] or "",
                    description=r["description"] or "",
                )
                for r in rows
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS skill query failed for: %s", sanitized)
            return []

    # ------------------------------------------------------------------
    # Task-scoped Chunk Search (for recall)
    # ------------------------------------------------------------------

    def ann_search_chunks_in_tasks(
        self,
        query_vec: list[float],
        task_ids: list[str],
        top_k: int = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        if not task_ids:
            return []
        blob = serialize_float32(query_vec)
        sql = """SELECT v.chunk_id, v.distance,
                        c.summary, c.content, c.role, c.session_key,
                        c.task_id, c.created_at
                 FROM vec_chunks v
                 JOIN chunks c ON c.id = v.chunk_id
                 WHERE v.embedding MATCH ? AND k = ?
                   AND c.dedup_status = 'active'"""
        params: list[Any] = [blob, top_k * 3]
        if owner:
            sql += " AND c.owner = ?"
            params.append(owner)
        sql += " ORDER BY v.distance"
        rows = self._conn.execute(sql, params).fetchall()

        task_set = set(task_ids)
        hits: list[SearchHit] = []
        for r in rows:
            if r["task_id"] not in task_set:
                continue
            score = 1.0 - (r["distance"] or 0.0)
            hits.append(SearchHit(
                chunk_id=r["chunk_id"],
                score=score,
                summary=r["summary"] or "",
                content_excerpt=r["content"][:300] if r["content"] else "",
                role=r["role"] or "",
                session_key=r["session_key"] or "",
                task_id=r["task_id"],
                created_at=r["created_at"] or 0,
            ))
            if len(hits) >= top_k:
                break
        return hits

    def fts_search_chunks_in_tasks(
        self,
        query: str,
        task_ids: list[str],
        limit: int = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        if not task_ids:
            return []
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []
        try:
            placeholders = ",".join("?" for _ in task_ids)
            sql = f"""SELECT c.id as chunk_id, rank,
                            c.summary, c.content, c.role, c.session_key,
                            c.task_id, c.created_at
                     FROM chunks_fts f
                     JOIN chunks c ON c.rowid = f.rowid
                     WHERE chunks_fts MATCH ?
                       AND c.dedup_status = 'active'
                       AND c.task_id IN ({placeholders})"""
            params: list[Any] = [sanitized, *task_ids]
            if owner:
                sql += " AND c.owner = ?"
                params.append(owner)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit * 2)
            rows = self._conn.execute(sql, params).fetchall()
            if not rows:
                return []
            max_abs = max(abs(r["rank"]) for r in rows) or 1.0
            return [
                SearchHit(
                    chunk_id=r["chunk_id"],
                    score=abs(r["rank"]) / max_abs,
                    summary=r["summary"] or "",
                    content_excerpt=r["content"][:300] if r["content"] else "",
                    role=r["role"] or "",
                    session_key=r["session_key"] or "",
                    task_id=r["task_id"],
                    created_at=r["created_at"] or 0,
                )
                for r in rows[:limit]
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS chunks_in_tasks query failed for: %s", sanitized)
            return []

    def ann_search_orphan_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        blob = serialize_float32(query_vec)
        sql = """SELECT v.chunk_id, v.distance,
                        c.summary, c.content, c.role, c.session_key,
                        c.task_id, c.created_at
                 FROM vec_chunks v
                 JOIN chunks c ON c.id = v.chunk_id
                 WHERE v.embedding MATCH ? AND k = ?
                   AND c.dedup_status IN ('active', 'orphaned')"""
        params: list[Any] = [blob, top_k * 3]
        if owner:
            sql += " AND c.owner = ?"
            params.append(owner)
        sql += " ORDER BY v.distance"
        rows = self._conn.execute(sql, params).fetchall()

        hits: list[SearchHit] = []
        for r in rows:
            if r["task_id"] is not None:
                continue
            if exclude_session and r["session_key"] == exclude_session:
                continue
            score = 1.0 - (r["distance"] or 0.0)
            hits.append(SearchHit(
                chunk_id=r["chunk_id"],
                score=score,
                summary=r["summary"] or "",
                content_excerpt=r["content"][:300] if r["content"] else "",
                role=r["role"] or "",
                session_key=r["session_key"] or "",
                task_id=None,
                created_at=r["created_at"] or 0,
            ))
            if len(hits) >= top_k:
                break
        return hits

    def fts_search_orphan_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []
        try:
            sql = """SELECT c.id as chunk_id, rank,
                            c.summary, c.content, c.role, c.session_key,
                            c.task_id, c.created_at
                     FROM chunks_fts f
                     JOIN chunks c ON c.rowid = f.rowid
                     WHERE chunks_fts MATCH ?
                       AND c.dedup_status IN ('active', 'orphaned')
                       AND c.task_id IS NULL"""
            params: list[Any] = [sanitized]
            if exclude_session:
                sql += " AND c.session_key != ?"
                params.append(exclude_session)
            if owner:
                sql += " AND c.owner = ?"
                params.append(owner)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit * 2)
            rows = self._conn.execute(sql, params).fetchall()
            if not rows:
                return []
            max_abs = max(abs(r["rank"]) for r in rows) or 1.0
            return [
                SearchHit(
                    chunk_id=r["chunk_id"],
                    score=abs(r["rank"]) / max_abs,
                    summary=r["summary"] or "",
                    content_excerpt=r["content"][:300] if r["content"] else "",
                    role=r["role"] or "",
                    session_key=r["session_key"] or "",
                    task_id=None,
                    created_at=r["created_at"] or 0,
                )
                for r in rows[:limit]
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS orphan_chunks query failed for: %s", sanitized)
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


    # ------------------------------------------------------------------
    # Session Summaries (结构化压缩摘要)
    # ------------------------------------------------------------------

    def upsert_session_summary(
        self,
        session_id: str,
        agent_id: str,
        summary: dict[str, Any],
    ) -> SessionSummary:
        now = time.time()
        row = self._conn.execute(
            "SELECT id, version FROM session_summaries WHERE session_id = ? AND agent_id = ?",
            (session_id, agent_id),
        ).fetchone()

        if row:
            sid, old_version = row["id"], row["version"]
            new_version = old_version + 1
            self._conn.execute("""
                UPDATE session_summaries SET
                    version = ?, goal = ?, decisions = ?, progress = ?,
                    open_items = ?, entities = ?, user_preferences = ?,
                    raw_summary = ?, token_count = ?, updated_at = ?
                WHERE id = ?
            """, (
                new_version,
                summary.get("goal", ""),
                json.dumps(summary.get("decisions", []), ensure_ascii=False),
                summary.get("progress", ""),
                json.dumps(summary.get("open_items", []), ensure_ascii=False),
                json.dumps(summary.get("entities", []), ensure_ascii=False),
                json.dumps(summary.get("user_preferences", []), ensure_ascii=False),
                summary.get("raw_summary", ""),
                summary.get("token_count", 0),
                now,
                sid,
            ))
        else:
            sid = str(uuid.uuid4())
            new_version = 1
            self._conn.execute("""
                INSERT INTO session_summaries
                    (id, session_id, agent_id, version, goal, decisions, progress,
                     open_items, entities, user_preferences, raw_summary, token_count,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sid, session_id, agent_id, new_version,
                summary.get("goal", ""),
                json.dumps(summary.get("decisions", []), ensure_ascii=False),
                summary.get("progress", ""),
                json.dumps(summary.get("open_items", []), ensure_ascii=False),
                json.dumps(summary.get("entities", []), ensure_ascii=False),
                json.dumps(summary.get("user_preferences", []), ensure_ascii=False),
                summary.get("raw_summary", ""),
                summary.get("token_count", 0),
                now, now,
            ))

        self._conn.commit()
        return SessionSummary(
            id=sid, session_id=session_id, agent_id=agent_id,
            version=new_version,
            goal=summary.get("goal", ""),
            decisions=json.dumps(summary.get("decisions", []), ensure_ascii=False),
            progress=summary.get("progress", ""),
            open_items=json.dumps(summary.get("open_items", []), ensure_ascii=False),
            entities=json.dumps(summary.get("entities", []), ensure_ascii=False),
            user_preferences=json.dumps(summary.get("user_preferences", []), ensure_ascii=False),
            raw_summary=summary.get("raw_summary", ""),
            token_count=summary.get("token_count", 0),
            created_at=now, updated_at=now,
        )

    def delete_session_summary(self, session_id: str, agent_id: str) -> bool:
        deleted = self._conn.execute(
            "DELETE FROM session_summaries WHERE session_id = ? AND agent_id = ?",
            (session_id, agent_id),
        ).rowcount
        self._conn.commit()
        return deleted > 0

    def get_session_summary(
        self, session_id: str, agent_id: str,
    ) -> SessionSummary | None:
        row = self._conn.execute(
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_fts(query: str) -> str:
    """Sanitize query string for FTS5 MATCH."""
    tokenized = _tokenize_for_fts(query)
    if not tokenized.strip():
        return ""
    words = tokenized.split()
    safe = [w for w in words if re.match(r"^[\w\u4e00-\u9fff]+$", w)]
    if not safe:
        return ""
    return " OR ".join(safe)


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        session_key=row["session_key"],
        turn_id=row["turn_id"],
        seq=row["seq"],
        role=row["role"],
        content=row["content"],
        kind=row["kind"] or "paragraph",
        summary=row["summary"] or "",
        task_id=row["task_id"],
        skill_id=row["skill_id"],
        owner=row["owner"] or "agent:main",
        content_hash=row["content_hash"] or "",
        dedup_status=row["dedup_status"] or "active",
        dedup_target=row["dedup_target"],
        dedup_reason=row["dedup_reason"],
        summary_source=row["summary_source"] or "llm",
        embedding_status=row["embedding_status"] or "ok",
        embedding_error=row["embedding_error"],
        created_at=row["created_at"] or 0,
        updated_at=row["updated_at"] or 0,
    )


def _row_to_task(row: sqlite3.Row) -> Task:
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


def _row_to_skill(row: sqlite3.Row) -> Skill:
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
