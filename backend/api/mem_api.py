"""记忆系统 API — 为前端 MemoryModal 提供数据"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(tags=["mem"])
logger = logging.getLogger(__name__)


def _get_store():
    from runtime.agent import agent_manager
    return agent_manager.mem_stores.get("main")


# ------------------------------------------------------------------
# GET /api/mem/stats
# ------------------------------------------------------------------

@router.get("/mem/stats")
async def mem_stats():
    store = _get_store()
    if not store:
        return {"ok": False, "error": "mem system not initialized"}
    try:
        conn = store._conn
        total_chunks = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE dedup_status='active'"
        ).fetchone()["c"]
        total_tasks = conn.execute("SELECT COUNT(*) as c FROM tasks").fetchone()["c"]
        completed_tasks = conn.execute(
            "SELECT COUNT(*) as c FROM tasks WHERE status='completed'"
        ).fetchone()["c"]
        total_skills = conn.execute("SELECT COUNT(*) as c FROM skills").fetchone()["c"]
        total_sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_key) as c FROM chunks"
        ).fetchone()["c"]

        role_rows = conn.execute(
            "SELECT role, COUNT(*) as c FROM chunks WHERE dedup_status='active' GROUP BY role"
        ).fetchall()
        role_breakdown = {r["role"]: r["c"] for r in role_rows}

        dedup_rows = conn.execute(
            "SELECT dedup_status, COUNT(*) as c FROM chunks GROUP BY dedup_status"
        ).fetchall()
        dedup_breakdown = {r["dedup_status"]: r["c"] for r in dedup_rows}

        time_range = conn.execute(
            "SELECT MIN(created_at) as earliest, MAX(created_at) as latest FROM chunks"
        ).fetchone()

        return {
            "ok": True,
            "totalChunks": total_chunks,
            "totalTasks": total_tasks,
            "completedTasks": completed_tasks,
            "totalSkills": total_skills,
            "totalSessions": total_sessions,
            "roleBreakdown": role_breakdown,
            "dedupBreakdown": dedup_breakdown,
            "timeRange": {
                "earliest": time_range["earliest"],
                "latest": time_range["latest"],
            },
        }
    except Exception as e:
        logger.warning("mem_stats error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/tasks
# ------------------------------------------------------------------

@router.get("/mem/tasks")
async def mem_tasks(
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    store = _get_store()
    if not store:
        return {"ok": False, "tasks": [], "total": 0}
    try:
        conn = store._conn
        conditions = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        total = conn.execute(f"SELECT COUNT(*) as c FROM tasks{where}", params).fetchone()["c"]
        rows = conn.execute(
            f"SELECT * FROM tasks{where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

        items = []
        for r in rows:
            chunk_count = conn.execute(
                "SELECT COUNT(*) as c FROM chunks WHERE task_id=? AND dedup_status='active'",
                (r["id"],),
            ).fetchone()["c"]
            items.append({
                "id": r["id"],
                "sessionKey": r["session_key"],
                "title": r["title"] or "",
                "summary": (r["summary"] or "")[:400],
                "status": r["status"],
                "startedAt": r["started_at"],
                "endedAt": r["ended_at"],
                "chunkCount": chunk_count,
            })
        return {"ok": True, "tasks": items, "total": total}
    except Exception as e:
        logger.warning("mem_tasks error: %s", e)
        return {"ok": False, "tasks": [], "total": 0, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/task/{task_id}
# ------------------------------------------------------------------

@router.get("/mem/task/{task_id}")
async def mem_task_detail(task_id: str):
    store = _get_store()
    if not store:
        return {"ok": False, "error": "not initialized"}
    try:
        task = store.get_task(task_id)
        if not task:
            return {"ok": False, "error": "not found"}
        chunks = store.get_chunks_by_task(task_id, limit=100)
        chunk_items = []
        for c in chunks:
            text = c.content
            if len(text) > 500:
                text = text[:497] + "..."
            chunk_items.append({
                "id": c.id,
                "role": c.role,
                "content": text,
                "summary": c.summary,
                "createdAt": c.created_at,
            })
        return {
            "ok": True,
            "id": task.id,
            "sessionKey": task.session_key,
            "title": task.title,
            "summary": task.summary,
            "status": task.status,
            "startedAt": task.started_at,
            "endedAt": task.ended_at,
            "chunks": chunk_items,
        }
    except Exception as e:
        logger.warning("mem_task_detail error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/skills
# ------------------------------------------------------------------

@router.get("/mem/skills")
async def mem_skills(
    status: str = Query("", description="Filter by status"),
):
    store = _get_store()
    if not store:
        return {"ok": False, "skills": []}
    try:
        conn = store._conn
        if status:
            rows = conn.execute(
                "SELECT * FROM skills WHERE status=? ORDER BY updated_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM skills ORDER BY updated_at DESC").fetchall()
        items = []
        for r in rows:
            items.append({
                "id": r["id"],
                "name": r["name"],
                "description": (r["description"] or "")[:300],
                "version": r["version"],
                "status": r["status"],
                "qualityScore": r["quality_score"],
                "createdAt": r["created_at"],
                "updatedAt": r["updated_at"],
            })
        return {"ok": True, "skills": items}
    except Exception as e:
        logger.warning("mem_skills error: %s", e)
        return {"ok": False, "skills": [], "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/skill/{skill_id}
# ------------------------------------------------------------------

@router.get("/mem/skill/{skill_id}")
async def mem_skill_detail(skill_id: str):
    store = _get_store()
    if not store:
        return {"ok": False, "error": "not initialized"}
    try:
        skill = store.get_skill(skill_id)
        if not skill:
            return {"ok": False, "error": "not found"}
        return {
            "ok": True,
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "status": skill.status,
            "qualityScore": skill.quality_score,
            "dirPath": skill.dir_path,
            "createdAt": skill.created_at,
            "updatedAt": skill.updated_at,
        }
    except Exception as e:
        logger.warning("mem_skill_detail error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/memories
# ------------------------------------------------------------------

@router.get("/mem/memories")
async def mem_memories(
    limit: int = Query(40, ge=1, le=200),
    page: int = Query(1, ge=1),
    session: str = Query("", description="Filter by session_key"),
    role: str = Query("", description="Filter by role"),
):
    store = _get_store()
    if not store:
        return {"ok": False, "memories": [], "total": 0}
    try:
        conn = store._conn
        offset = (page - 1) * limit
        conditions = ["dedup_status='active'"]
        params: list[Any] = []
        if session:
            conditions.append("session_key = ?")
            params.append(session)
        if role:
            conditions.append("role = ?")
            params.append(role)
        where = " WHERE " + " AND ".join(conditions)
        total = conn.execute(f"SELECT COUNT(*) as c FROM chunks{where}", params).fetchone()["c"]
        rows = conn.execute(
            f"SELECT id, session_key, role, summary, substr(content,1,300) as excerpt, "
            f"task_id, created_at FROM chunks{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        items = []
        for r in rows:
            items.append({
                "id": r["id"],
                "sessionKey": r["session_key"],
                "role": r["role"],
                "summary": r["summary"] or "",
                "excerpt": r["excerpt"] or "",
                "taskId": r["task_id"],
                "createdAt": r["created_at"],
            })
        return {
            "ok": True,
            "memories": items,
            "total": total,
            "page": page,
            "totalPages": max(1, -(-total // limit)),
        }
    except Exception as e:
        logger.warning("mem_memories error: %s", e)
        return {"ok": False, "memories": [], "total": 0, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/search
# ------------------------------------------------------------------

@router.get("/mem/search")
async def mem_search(
    q: str = Query("", description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    store = _get_store()
    if not store or not q.strip():
        return {"ok": True, "results": [], "query": q}
    try:
        fts_hits = store.fts_search_chunks(q, limit=limit)
        results = []
        for h in fts_hits:
            results.append({
                "id": h.chunk_id,
                "score": round(h.score, 3),
                "role": h.role,
                "summary": h.summary,
                "excerpt": h.content_excerpt[:300],
                "sessionKey": h.session_key,
                "taskId": h.task_id,
                "createdAt": h.created_at,
            })
        return {"ok": True, "results": results, "query": q, "total": len(results)}
    except Exception as e:
        logger.warning("mem_search error: %s", e)
        return {"ok": True, "results": [], "query": q, "error": str(e)}
