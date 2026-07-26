"""记忆存储层 — 单 SQLite 文件 + FTS5 + sqlite-vec ANN

Schema 与 docs/memory-system-refactor.md §3.1 一致:
  chunks / chunks_fts / vec_chunks
  tasks  / tasks_fts  / vec_tasks
  skills / skills_fts / vec_skills
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import sqlite_vec
from sqlite_vec import serialize_float32

from mem.chunk_repository import ChunkRepository, row_to_chunk as _row_to_chunk
from mem.dashboard_queries import MemoryDashboardQueries
from mem.fts_index import MemoryFtsIndex
from mem.models import (
    Chunk,
    DedupStatus,
    EmbeddingStatus,
    SearchHit,
    SessionSummary,
    Skill,
    SkillSearchHit,
    SkillStatus,
    SkillVisibility,
    SummarySource,
    Task,
    TaskSearchHit,
    TaskStatus,
)
from mem.schema import MemorySchema
from mem.search_queries import MemorySearchQueries
from mem.session_summary_repository import SessionSummaryRepository

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

        self._schema = MemorySchema(self._conn, dimensions)
        self._fts_index = MemoryFtsIndex(
            self._conn,
            lambda text: _tokenize_for_fts(text),
        )
        self._chunks = ChunkRepository(
            self._conn,
            sync_fts=lambda chunk_id: self._sync_chunk_fts(chunk_id),
            now_ms=lambda: _now_ms(),
            content_hash=lambda content: _content_hash(content),
        )
        self._dashboard_queries = MemoryDashboardQueries(self._conn)
        self._search_queries = MemorySearchQueries(
            self._conn,
            lambda query: _sanitize_fts(query),
        )
        self._session_summaries = SessionSummaryRepository(
            self._conn,
            now=lambda: time.time(),
            new_id=lambda: str(uuid.uuid4()),
        )
        self._init_schema()
        logger.info("MemStore initialized: %s (dim=%d)", db_path, dimensions)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._schema.initialize()

    # ------------------------------------------------------------------
    # FTS sync helpers — 预分词写入 FTS 索引
    # ------------------------------------------------------------------

    def _sync_chunk_fts(self, chunk_id: str) -> None:
        self._fts_index.sync_chunk(chunk_id)

    def _sync_task_fts(self, task_id: str) -> None:
        self._fts_index.sync_task(task_id)

    def _sync_skill_fts(self, skill_id: str) -> None:
        self._fts_index.sync_skill(skill_id)

    def rebuild_fts_indexes(self) -> None:
        """重建所有 FTS 索引（迁移或修复时使用）。"""
        self._fts_index.rebuild()

    def _ensure_chunk_columns(self) -> None:
        self._schema.ensure_chunk_columns()

    def _ensure_task_columns(self) -> None:
        self._schema.ensure_task_columns()

    # ------------------------------------------------------------------
    # Chunks — Write
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> None:
        self._chunks.insert(chunk)

    def update_chunk_summary(
        self,
        chunk_id: str,
        summary: str,
        *,
        summary_source: SummarySource | None = None,
    ) -> None:
        self._chunks.update_summary(
            chunk_id,
            summary,
            summary_source=summary_source,
        )

    def update_chunk_embedding_status(
        self,
        chunk_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> None:
        self._chunks.update_embedding_status(chunk_id, status, error)

    def mark_dedup_status(
        self,
        chunk_id: str,
        status: DedupStatus,
        target: str | None = None,
        reason: str | None = None,
    ) -> None:
        self._chunks.mark_dedup_status(
            chunk_id,
            status,
            target,
            reason,
        )

    def orphan_chunk(self, chunk_id: str, reason: str | None = None) -> None:
        self._chunks.orphan(chunk_id, reason)

    def find_active_chunk_by_hash(self, content: str, owner: str) -> str | None:
        return self._chunks.find_active_by_hash(content, owner)

    # ------------------------------------------------------------------
    # Chunks — Read
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_chunks_by_task(self, task_id: str, limit: int | None = None) -> list[Chunk]:
        return self._chunks.get_by_task(task_id, limit)

    def get_chunks_for_embedding_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        return self._chunks.get_for_embedding_retry(owner, limit)

    def get_chunks_for_summary_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        return self._chunks.get_for_summary_retry(owner, limit)

    def get_chunks_in_range(
        self,
        session_key: str,
        turn_id: str,
        seq: int,
        window: int = 2,
        owner: str | None = None,
    ) -> list[Chunk]:
        """Get neighboring chunks for timeline display."""
        return self._chunks.get_in_range(
            session_key,
            turn_id,
            seq,
            window,
            owner,
        )

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
        return self._search_queries.ann_search_chunks(
            query_vec,
            top_k=top_k,
            exclude_session=exclude_session,
        )

    def ann_search_tasks(
        self, query_vec: list[float], top_k: int = 5,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        return self._search_queries.ann_search_tasks(
            query_vec,
            top_k=top_k,
            owner=owner,
        )

    def ann_search_skills(
        self, query_vec: list[float], top_k: int = 5,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        return self._search_queries.ann_search_skills(
            query_vec,
            top_k=top_k,
            owner=owner,
        )

    def ann_dedup_candidates(
        self,
        query_vec: list[float],
        threshold: float,
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[SearchHit]:
        """Find top-K similar active chunks above threshold for dedup."""
        return self._search_queries.ann_dedup_candidates(
            query_vec,
            threshold,
            top_k=top_k,
            owner=owner,
        )

    # ------------------------------------------------------------------
    # FTS Search
    # ------------------------------------------------------------------

    def fts_search_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.fts_search_chunks(
            query,
            limit=limit,
            exclude_session=exclude_session,
        )

    def fts_search_tasks(
        self, query: str, limit: int = 10,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        return self._search_queries.fts_search_tasks(
            query,
            limit=limit,
            owner=owner,
        )

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
        return self._search_queries.fts_search_skills(
            query,
            limit=limit,
            owner=owner,
        )

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
        return self._search_queries.ann_search_chunks_in_tasks(
            query_vec,
            task_ids,
            top_k=top_k,
            owner=owner,
        )

    def fts_search_chunks_in_tasks(
        self,
        query: str,
        task_ids: list[str],
        limit: int = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.fts_search_chunks_in_tasks(
            query,
            task_ids,
            limit=limit,
            owner=owner,
        )

    def ann_search_orphan_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.ann_search_orphan_chunks(
            query_vec,
            top_k=top_k,
            exclude_session=exclude_session,
            owner=owner,
        )

    def fts_search_orphan_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.fts_search_orphan_chunks(
            query,
            limit=limit,
            exclude_session=exclude_session,
            owner=owner,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_dashboard_stats(self) -> dict[str, Any]:
        return self._dashboard_queries.get_stats()

    def list_dashboard_tasks(
        self,
        *,
        status: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        return self._dashboard_queries.list_tasks(
            status=status,
            limit=limit,
            offset=offset,
        )

    def list_dashboard_skills(self, *, status: str = "") -> list[dict[str, Any]]:
        return self._dashboard_queries.list_skills(status=status)

    def list_dashboard_memories(
        self,
        *,
        limit: int = 40,
        offset: int = 0,
        session: str = "",
        role: str = "",
    ) -> tuple[list[dict[str, Any]], int]:
        return self._dashboard_queries.list_memories(
            limit=limit,
            offset=offset,
            session=session,
            role=role,
        )

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
        return self._session_summaries.upsert(
            session_id,
            agent_id,
            summary,
        )

    def delete_session_summary(self, session_id: str, agent_id: str) -> bool:
        return self._session_summaries.delete(session_id, agent_id)

    def get_session_summary(
        self, session_id: str, agent_id: str,
    ) -> SessionSummary | None:
        return self._session_summaries.get(session_id, agent_id)


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
