"""SSE 事件流 — 用于前端实时接收 Agent 生命周期事件"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import (
    get_agent_manager,
    get_heartbeat_runner,
    get_session_manager,
)

router = APIRouter()


class SubagentKillRequest(BaseModel):
    target: str  # run_id or "all"
    session_id: str | None = None


class SubagentSteerRequest(BaseModel):
    run_id: str
    message: str


@router.get("/agents/{agent_id}/events")
async def agent_events(agent_id: str):
    """SSE 端点：订阅 Agent 的生命周期事件"""
    from infra.event_bus import event_bus

    queue = event_bus.subscribe(agent_id)

    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    data = json.dumps(event, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(agent_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agents/{agent_id}/usage")
async def agent_usage(agent_id: str, session_id: str | None = None):
    """获取 Agent 的 token 使用统计"""
    from infra.run_tracker import run_tracker
    from config import resolve_agent_config

    usage = run_tracker.get_cumulative_usage(agent_id, session_id)
    model = resolve_agent_config(agent_id).get("model", "deepseek-chat")

    return {
        **usage,
        "model": model,
    }


@router.get("/agents/{agent_id}/audit-log")
async def agent_audit_log(agent_id: str, limit: int = 50):
    """获取最近的审计日志"""
    from infra.audit_log import audit_logger
    return audit_logger.read_recent(agent_id, limit)


def _run_to_item(r, session_manager, time_module) -> dict:
    """将 SubagentRunRecord 转为 API 项"""
    elapsed = None
    duration_ms = None
    if r.started_at:
        end = r.ended_at or time_module.time()
        elapsed = int(end - r.started_at)
        duration_ms = max(0, int((end - r.started_at) * 1000))
    status = "running" if r.ended_at is None else (r.outcome or "completed")
    state = getattr(r, "state", "running" if r.ended_at is None else "succeeded")
    child_parts = r.child_session_key.split(":")
    child_session = child_parts[-1] if len(child_parts) >= 2 else r.child_session_key
    child_agent = child_parts[1] if len(child_parts) >= 2 else r.target_agent_id
    messages = []
    data = session_manager.load_session(child_session, child_agent)
    if data:
        for m in data.get("messages", []):
            messages.append({
                "role": m.get("role"),
                "content": (m.get("content", "") or "")[:500],
                "tool_calls": m.get("tool_calls"),
            })
    return {
        "run_id": r.run_id,
        "label": r.label or "子Agent",
        "task": r.task,
        "target_agent_id": r.target_agent_id,
        "status": status,
        "state": state,
        "terminal_reason": getattr(r, "terminal_reason", None),
        "elapsed": elapsed,
        "duration_ms": duration_ms,
        "started_at": r.started_at,
        "ended_at": r.ended_at,
        "result_summary": (r.result_summary or "")[:300],
        "messages": messages,
        "created_at": r.created_at,
        "spawn_depth": r.spawn_depth,
        "requester_session_key": r.requester_session_key,
        "child_session_key": r.child_session_key,
        "result_delivery_state": getattr(r, "result_delivery_state", "pending"),
        "delivery_work_id": getattr(r, "delivery_work_id", None),
        "announce_retry_count": getattr(r, "announce_retry_count", 0),
        "archive_at_ms": getattr(r, "archive_at_ms", None),
    }


def _count_active_descendants(
    records,
    subagent_service,
    requester_key: str,
    visited: set[str] | None = None,
) -> int:
    seen = set(visited or ())
    count = 0
    for record in records:
        if (
            record.requester_session_key != requester_key
            or record.run_id in seen
        ):
            continue
        seen.add(record.run_id)
        if record.ended_at is None:
            count += 1
        count += _count_active_descendants(
            records,
            subagent_service,
            subagent_service.child_requester_key(record),
            seen,
        )
    return count


def _build_subagent_tree(
    records,
    subagent_service,
    session_manager,
    agent_id: str,
    root_session_key: str,
    session_id_filter: str | None,
    time_module,
    cutoff: float | None = None,
) -> list[dict]:
    """递归构建子 Agent 树。cutoff 为 None 时不过滤；否则只包含 ended_at is None 或 ended_at >= cutoff 的 run"""
    tree: list[dict] = []
    for r in records:
        if r.requester_session_key != root_session_key:
            continue
        if session_id_filter and session_id_filter not in r.requester_session_key:
            continue
        if cutoff is not None and r.ended_at is not None and r.ended_at < cutoff:
            continue
        item = _run_to_item(r, session_manager, time_module)
        child_sk = subagent_service.child_requester_key(r)
        item["descendants_active_count"] = (
            _count_active_descendants(
                records,
                subagent_service,
                child_sk,
            )
        )
        item["children"] = _build_subagent_tree(
            records,
            subagent_service,
            session_manager,
            agent_id,
            child_sk,
            session_id_filter,
            time_module,
            cutoff=cutoff,
        )
        tree.append(item)
    tree.sort(key=lambda x: x["created_at"], reverse=True)
    return tree


@router.get("/agents/{agent_id}/subagents")
async def list_subagents(
    agent_id: str,
    session_id: str | None = None,
    include_recent_minutes: int | None = None,
    agent_manager: Any = Depends(get_agent_manager),
    session_manager: Any = Depends(get_session_manager),
):
    """获取子 Agent 列表及状态，返回树结构 + 扁平列表（按 requester_session_key 建树）

    include_recent_minutes: 只展示运行中 + 最近 N 分钟内完成的子 Agent，超过的不出现在 list 中。
    默认从 config.agents.defaults.subagents.recent_minutes 读取（30），API 参数可覆盖。
    """
    import time as time_module

    main_sid = session_manager.resolve_main_session_id(agent_id)
    effective_session_id = (
        (session_id or "").strip() or main_sid
    )
    result = agent_manager.subagent_service.list_runs(
        requester_agent_id=agent_id,
        requester_session_id=effective_session_id,
        recent_minutes=include_recent_minutes,
        recursive=True,
    )
    tree = _build_subagent_tree(
        result.records,
        agent_manager.subagent_service,
        session_manager,
        agent_id,
        result.requester_key,
        None,
        time_module,
    )

    flat: list[dict] = []

    def flatten(nodes: list[dict]) -> None:
        for n in nodes:
            flat.append({k: v for k, v in n.items() if k != "children"})
            if n.get("children"):
                flatten(n["children"])

    flatten(tree)
    flat.sort(key=lambda x: x["created_at"], reverse=True)

    return {
        "tree": tree,
        "flat": flat,
        "include_recent_minutes": result.recent_minutes,
    }


@router.post("/agents/{agent_id}/subagents/kill")
async def kill_subagents(
    agent_id: str,
    req: SubagentKillRequest,
    agent_manager: Any = Depends(get_agent_manager),
    session_manager: Any = Depends(get_session_manager),
):
    from subagents.subagent_service import SubagentServiceError

    session_id = (req.session_id or "").strip() or session_manager.resolve_main_session_id(agent_id)
    target = (req.target or "").strip()
    try:
        result = agent_manager.subagent_service.kill(
            requester_agent_id=agent_id,
            requester_session_id=session_id,
            target=target,
        )
    except SubagentServiceError as exc:
        return {"ok": False, "error": str(exc)}
    if target in ("all", "*"):
        return {
            "ok": True,
            "killed": result.killed,
            "scope": result.scope,
        }
    return {"ok": True, "run_id": result.run_id}


@router.post("/agents/{agent_id}/subagents/steer")
async def steer_subagent(
    agent_id: str,
    req: SubagentSteerRequest,
    agent_manager: Any = Depends(get_agent_manager),
):
    from subagents.subagent_service import SubagentServiceError

    try:
        result = agent_manager.subagent_service.steer(
            requester_agent_id=agent_id,
            requester_session_id=None,
            run_id=req.run_id,
            message=req.message,
        )
    except SubagentServiceError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "run_id": result.record.run_id,
        "replaced_run_id": result.replaced_run_id,
    }


@router.get("/agents/{agent_id}/status")
async def agent_status(
    agent_id: str,
    agent_manager: Any = Depends(get_agent_manager),
    heartbeat_runner: Any = Depends(get_heartbeat_runner),
):
    """获取 Agent 运行状态"""
    state = agent_manager.get_state(agent_id)
    return {
        "agent_id": agent_id,
        "total_turns": state.total_turns,
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
        "compaction_count": state.compaction_count,
        "thinking": state.thinking,
        "verbose": state.verbose,
        "reasoning": state.reasoning,
        "last_active": state.last_active,
        "heartbeat_active": agent_id in heartbeat_runner.active_agents,
    }
