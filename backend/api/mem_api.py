"""记忆系统 API — 为前端 MemoryModal 提供数据"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_agent_manager

router = APIRouter(tags=["mem"])
logger = logging.getLogger(__name__)


def _get_store(agent_id: str, agent_manager: Any):
    return agent_manager.mem_stores.get(agent_id)


# ------------------------------------------------------------------
# GET /api/mem/stats
# ------------------------------------------------------------------

@router.get("/mem/stats")
async def mem_stats(
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "error": "mem system not initialized"}
    try:
        return {"ok": True, **store.get_dashboard_stats()}
    except Exception as e:
        logger.warning("mem_stats error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/tasks
# ------------------------------------------------------------------

@router.get("/mem/tasks")
async def mem_tasks(
    agent_id: str = Query("main"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "tasks": [], "total": 0}
    try:
        items, total = store.list_dashboard_tasks(
            status=status, limit=limit, offset=offset
        )
        return {"ok": True, "tasks": items, "total": total}
    except Exception as e:
        logger.warning("mem_tasks error: %s", e)
        return {"ok": False, "tasks": [], "total": 0, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/task/{task_id}
# ------------------------------------------------------------------

@router.get("/mem/task/{task_id}")
async def mem_task_detail(
    task_id: str,
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
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
    agent_id: str = Query("main"),
    status: str = Query("", description="Filter by status"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "skills": []}
    try:
        items = store.list_dashboard_skills(status=status)
        return {"ok": True, "skills": items}
    except Exception as e:
        logger.warning("mem_skills error: %s", e)
        return {"ok": False, "skills": [], "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/skill/{skill_id}
# ------------------------------------------------------------------

@router.get("/mem/skill/{skill_id}")
async def mem_skill_detail(
    skill_id: str,
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
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
    agent_id: str = Query("main"),
    limit: int = Query(40, ge=1, le=200),
    page: int = Query(1, ge=1),
    session: str = Query("", description="Filter by session_key"),
    role: str = Query("", description="Filter by role"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "memories": [], "total": 0}
    try:
        offset = (page - 1) * limit
        items, total = store.list_dashboard_memories(
            limit=limit,
            offset=offset,
            session=session,
            role=role,
        )
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
    agent_id: str = Query("main"),
    q: str = Query("", description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
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
