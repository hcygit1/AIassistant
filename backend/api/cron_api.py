"""Cron API — CRUD、手动触发、任务历史、自然语言提醒"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


def _cron_service():
    from scheduler.cron_service import cron_service

    return cron_service


def _raise_cron_error(exc) -> None:
    if exc.code == "disabled":
        status = 409
    elif exc.code == "not_found":
        status = 404
    elif exc.code in ("invalid_schedule", "invalid_payload"):
        status = 400
    else:
        status = 502
    raise HTTPException(status, str(exc))


def _task_history_service():
    from scheduler.task_history_service import task_history_service

    return task_history_service


def _raise_task_history_error(exc) -> None:
    if exc.code == "invalid_filter":
        status = 400
    elif exc.code == "not_found":
        status = 404
    else:
        status = 409
    raise HTTPException(status, str(exc))


class CronJobCreate(BaseModel):
    name: str = Field(..., description="任务名称")
    description: str = Field(default="", description="描述")
    agent_id: str = Field(default="main", description="Agent ID")
    enabled: bool = Field(default=True, description="是否启用")
    delete_after_run: bool = Field(default=False, alias="deleteAfterRun", description="一次性任务执行后自动删除")
    schedule: dict = Field(..., description="调度配置 {kind, at|everyMs|expr, tz?}")
    payload: dict = Field(..., description="Payload {kind: 'systemEvent', text: str}")

    model_config = {"populate_by_name": True}


class CronJobUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    agent_id: str | None = None
    enabled: bool | None = None
    delete_after_run: bool | None = Field(default=None, alias="deleteAfterRun")
    schedule: dict | None = None
    payload: dict | None = None

    model_config = {"populate_by_name": True}


@router.get("/cron/jobs")
async def list_cron_jobs():
    """列出所有 Cron 任务"""
    return [
        job.to_dict()
        for job in _cron_service().list_jobs()
    ]


@router.post("/cron/jobs")
async def create_cron_job(body: CronJobCreate):
    """创建 Cron 任务"""
    from scheduler.cron_service import CronServiceError

    try:
        job = _cron_service().create_job(
            name=body.name,
            description=body.description,
            agent_id=body.agent_id,
            enabled=body.enabled,
            delete_after_run=body.delete_after_run,
            schedule=body.schedule,
            payload=body.payload,
        )
    except CronServiceError as exc:
        _raise_cron_error(exc)
    return job.to_dict()


@router.get("/cron/jobs/{job_id}")
async def get_cron_job(job_id: str):
    """获取单个 Cron 任务"""
    from scheduler.cron_service import CronServiceError

    try:
        return _cron_service().get_job(job_id).to_dict()
    except CronServiceError as exc:
        _raise_cron_error(exc)


@router.patch("/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate):
    """更新 Cron 任务"""
    from scheduler.cron_service import CronServiceError

    try:
        job = _cron_service().update_job(
            job_id,
            name=body.name,
            description=body.description,
            agent_id=body.agent_id,
            enabled=body.enabled,
            delete_after_run=body.delete_after_run,
            schedule=body.schedule,
            payload=body.payload,
        )
    except CronServiceError as exc:
        _raise_cron_error(exc)
    return job.to_dict()


@router.delete("/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str):
    """删除 Cron 任务"""
    from scheduler.cron_service import CronServiceError

    try:
        _cron_service().delete_job(job_id)
    except CronServiceError as exc:
        _raise_cron_error(exc)
    return {"ok": True}


@router.post("/cron/jobs/{job_id}/run")
async def run_cron_job(job_id: str, mode: str = "force"):
    """手动触发 Cron 任务"""
    from scheduler.cron_service import CronServiceError

    try:
        _cron_service().trigger_job(job_id)
    except CronServiceError as exc:
        _raise_cron_error(exc)
    return {"ok": True, "message": "Triggered"}


# ---------------------------------------------------------------------------
# 任务历史 API (SQLite)
# ---------------------------------------------------------------------------

@router.get("/tasks/history")
async def get_task_history(
    agent_id: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """查询任务执行历史"""
    from scheduler.task_history_service import TaskHistoryError

    try:
        page = _task_history_service().query(
            agent_id=agent_id,
            kind=kind,
            status=status,
            limit=limit,
            offset=offset,
        )
    except TaskHistoryError as exc:
        _raise_task_history_error(exc)
    return {
        "items": page.items,
        "total": page.total,
        "limit": page.limit,
        "offset": page.offset,
    }


@router.get("/system-work/history")
async def get_system_work_history(
    kind: str | None = None,
    status: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """查询系统会话工作台账，便于排查 announce / heartbeat / cron 的投递状态。"""
    from sessions.session_work_store import session_work_store

    items = session_work_store.query(
        kind=kind,
        status=status,
        agent_id=agent_id,
        session_id=session_id,
        run_id=run_id,
        limit=limit,
        offset=offset,
    )
    total = session_work_store.count(
        kind=kind,
        status=status,
        agent_id=agent_id,
        session_id=session_id,
        run_id=run_id,
    )
    return {
        "items": [
            {
                "id": r.id,
                "kind": r.kind,
                "agent_id": r.agent_id,
                "session_id": r.session_id,
                "run_id": r.run_id,
                "status": r.status,
                "recover_on_restart": r.recover_on_restart,
                "created_at_ms": r.created_at_ms,
                "started_at_ms": r.started_at_ms,
                "finished_at_ms": r.finished_at_ms,
                "last_error": r.last_error,
                "content_preview": r.content[:200],
            }
            for r in items
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    from scheduler.task_history_service import TaskHistoryError

    try:
        result = _task_history_service().cancel(task_id)
    except TaskHistoryError as exc:
        _raise_task_history_error(exc)
    return {"ok": result.ok, "status": result.status}


# ---------------------------------------------------------------------------
# 自然语言提醒 (底层映射为一次性 Cron)
# ---------------------------------------------------------------------------

class ReminderCreate(BaseModel):
    text: str = Field(..., description="提醒内容")
    at: str = Field(..., description="提醒时间 (ISO 8601)")
    agent_id: str = Field(default="main")

    model_config = {"populate_by_name": True}


@router.post("/reminders")
async def create_reminder(body: ReminderCreate):
    """创建自然语言提醒 — 底层是一次性 at cron job"""
    from scheduler.cron_service import CronServiceError

    try:
        job = _cron_service().create_reminder(
            text=body.text,
            at=body.at,
            agent_id=body.agent_id,
        )
    except CronServiceError as exc:
        _raise_cron_error(exc)
    return {"ok": True, "job": job.to_dict()}
