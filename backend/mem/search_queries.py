"""Read-only FTS and ANN queries for memory retrieval."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from sqlite_vec import serialize_float32

from mem.models import SearchHit, SkillSearchHit, TaskSearchHit


logger = logging.getLogger(__name__)


class MemorySearchQueries:
    def __init__(
        self,
        connection: sqlite3.Connection,
        sanitize_fts: Callable[[str], str],
    ) -> None:
        self._connection = connection
        self._sanitize_fts = sanitize_fts

    def ann_search_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        blob = serialize_float32(query_vec)
        rows = self._connection.execute(
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
        for row in rows:
            if exclude_session and row["session_key"] == exclude_session:
                continue
            score = (
                1.0 - row["distance"]
                if row["distance"] is not None
                else 0.0
            )
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    score=score,
                    summary=row["summary"] or "",
                    content_excerpt=(
                        row["content"][:300] if row["content"] else ""
                    ),
                    role=row["role"] or "",
                    session_key=row["session_key"] or "",
                    task_id=row["task_id"],
                    created_at=row["created_at"] or 0,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def ann_search_tasks(
        self,
        query_vec: list[float],
        top_k: int = 5,
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
        rows = self._connection.execute(sql, params).fetchall()

        return [
            TaskSearchHit(
                task_id=row["task_id"],
                score=1.0 - (row["distance"] or 0.0),
                title=row["title"] or "",
                summary=row["summary"] or "",
                status=row["status"] or "",
                started_at=row["started_at"] or 0,
                ended_at=row["ended_at"],
            )
            for row in rows
        ]

    def ann_search_skills(
        self,
        query_vec: list[float],
        top_k: int = 5,
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
        rows = self._connection.execute(sql, params).fetchall()

        return [
            SkillSearchHit(
                skill_id=row["skill_id"],
                score=1.0 - (row["distance"] or 0.0),
                name=row["name"] or "",
                description=row["description"] or "",
            )
            for row in rows
        ]

    def ann_dedup_candidates(
        self,
        query_vec: list[float],
        threshold: float,
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[SearchHit]:
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

        rows = self._connection.execute(sql, params).fetchall()
        hits: list[SearchHit] = []
        for row in rows:
            score = 1.0 - (row["distance"] or 0.0)
            if score < threshold:
                continue
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    score=score,
                    summary=row["summary"] or "",
                    content_excerpt=(
                        row["content"][:300] if row["content"] else ""
                    ),
                    role=row["role"] or "",
                    session_key=row["session_key"] or "",
                    task_id=row["task_id"],
                    created_at=row["created_at"] or 0,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def fts_search_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        sanitized = self._sanitize_fts(query)
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

            rows = self._connection.execute(sql, params).fetchall()
            if not rows:
                return []

            max_abs = max(abs(row["rank"]) for row in rows) or 1.0
            hits: list[SearchHit] = []
            for row in rows:
                hits.append(
                    SearchHit(
                        chunk_id=row["chunk_id"],
                        score=abs(row["rank"]) / max_abs,
                        summary=row["summary"] or "",
                        content_excerpt=(
                            row["content"][:300] if row["content"] else ""
                        ),
                        role=row["role"] or "",
                        session_key=row["session_key"] or "",
                        task_id=row["task_id"],
                        created_at=row["created_at"] or 0,
                    )
                )
            return hits[:limit]
        except sqlite3.OperationalError:
            logger.warning("FTS query failed for: %s", sanitized)
            return []

    def fts_search_tasks(
        self,
        query: str,
        limit: int = 10,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        sanitized = self._sanitize_fts(query)
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
            rows = self._connection.execute(sql, params).fetchall()

            if not rows:
                return []
            max_abs = max(abs(row["rank"]) for row in rows) or 1.0
            return [
                TaskSearchHit(
                    task_id=row["task_id"],
                    score=abs(row["rank"]) / max_abs,
                    title=row["title"] or "",
                    summary=row["summary"] or "",
                    status=row["status"] or "",
                    started_at=row["started_at"] or 0,
                    ended_at=row["ended_at"],
                )
                for row in rows
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS task query failed for: %s", sanitized)
            return []

    def fts_search_skills(
        self,
        query: str,
        limit: int = 10,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        sanitized = self._sanitize_fts(query)
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
            rows = self._connection.execute(sql, params).fetchall()

            if not rows:
                return []
            max_abs = max(abs(row["rank"]) for row in rows) or 1.0
            return [
                SkillSearchHit(
                    skill_id=row["skill_id"],
                    score=abs(row["rank"]) / max_abs,
                    name=row["name"] or "",
                    description=row["description"] or "",
                )
                for row in rows
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS skill query failed for: %s", sanitized)
            return []

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
        rows = self._connection.execute(sql, params).fetchall()

        task_set = set(task_ids)
        hits: list[SearchHit] = []
        for row in rows:
            if row["task_id"] not in task_set:
                continue
            score = 1.0 - (row["distance"] or 0.0)
            hits.append(SearchHit(
                chunk_id=row["chunk_id"],
                score=score,
                summary=row["summary"] or "",
                content_excerpt=(
                    row["content"][:300] if row["content"] else ""
                ),
                role=row["role"] or "",
                session_key=row["session_key"] or "",
                task_id=row["task_id"],
                created_at=row["created_at"] or 0,
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
        sanitized = self._sanitize_fts(query)
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
            rows = self._connection.execute(sql, params).fetchall()
            if not rows:
                return []
            max_abs = max(abs(row["rank"]) for row in rows) or 1.0
            return [
                SearchHit(
                    chunk_id=row["chunk_id"],
                    score=abs(row["rank"]) / max_abs,
                    summary=row["summary"] or "",
                    content_excerpt=(
                        row["content"][:300] if row["content"] else ""
                    ),
                    role=row["role"] or "",
                    session_key=row["session_key"] or "",
                    task_id=row["task_id"],
                    created_at=row["created_at"] or 0,
                )
                for row in rows[:limit]
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
        rows = self._connection.execute(sql, params).fetchall()

        hits: list[SearchHit] = []
        for row in rows:
            if row["task_id"] is not None:
                continue
            if exclude_session and row["session_key"] == exclude_session:
                continue
            score = 1.0 - (row["distance"] or 0.0)
            hits.append(SearchHit(
                chunk_id=row["chunk_id"],
                score=score,
                summary=row["summary"] or "",
                content_excerpt=(
                    row["content"][:300] if row["content"] else ""
                ),
                role=row["role"] or "",
                session_key=row["session_key"] or "",
                task_id=None,
                created_at=row["created_at"] or 0,
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
        sanitized = self._sanitize_fts(query)
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
            rows = self._connection.execute(sql, params).fetchall()
            if not rows:
                return []
            max_abs = max(abs(row["rank"]) for row in rows) or 1.0
            return [
                SearchHit(
                    chunk_id=row["chunk_id"],
                    score=abs(row["rank"]) / max_abs,
                    summary=row["summary"] or "",
                    content_excerpt=(
                        row["content"][:300] if row["content"] else ""
                    ),
                    role=row["role"] or "",
                    session_key=row["session_key"] or "",
                    task_id=None,
                    created_at=row["created_at"] or 0,
                )
                for row in rows[:limit]
            ]
        except sqlite3.OperationalError:
            logger.warning("FTS orphan_chunks query failed for: %s", sanitized)
            return []
