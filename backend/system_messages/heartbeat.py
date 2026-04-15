"""Heartbeat 后台任务 — 主会话运行、HEARTBEAT_OK 剥离、事件存储"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import DEFAULT_HEARTBEAT_PROMPT, get_heartbeat_config, resolve_agent_workspace, list_agents
from system_messages.heartbeat_utils import (
    strip_heartbeat_token,
    is_heartbeat_content_effectively_empty,
    is_within_active_hours,
)
from sessions.session_manager import session_manager
from infra.audit_log import audit_logger

logger = logging.getLogger(__name__)


def _get_max_events_per_agent() -> int:
    """从配置获取心跳事件队列大小"""
    try:
        from config import get_config
        cfg = get_config()
        return cfg.get("agents", {}).get("defaults", {}).get("heartbeat", {}).get("maxEvents", 50)
    except Exception:
        return 50


@dataclass
class HeartbeatEvent:
    ts: int
    status: str  # ok-empty | ok-token | sent | skipped | failed
    reason: str | None = None
    preview: str | None = None
    duration_ms: int | None = None
    agent_id: str = ""


_events: dict[str, deque[HeartbeatEvent]] = {}


def _get_events(agent_id: str) -> deque[HeartbeatEvent]:
    if agent_id not in _events:
        _events[agent_id] = deque(maxlen=_get_max_events_per_agent())
    return _events[agent_id]


def emit_heartbeat_event(agent_id: str, evt: HeartbeatEvent) -> None:
    _get_events(agent_id).append(evt)
    try:
        from scheduler.task_store import task_store, TaskRecord, TaskKind, TaskStatus
        import uuid
        status_map = {
            "ok-empty": TaskStatus.SUCCESS,
            "ok-token": TaskStatus.SUCCESS,
            "sent": TaskStatus.SUCCESS,
            "skipped": TaskStatus.CANCELLED,
            "failed": TaskStatus.FAILED,
        }
        record = TaskRecord(
            id=str(uuid.uuid4()),
            kind=TaskKind.HEARTBEAT,
            agent_id=agent_id,
            name=f"heartbeat:{evt.status}",
            status=status_map.get(evt.status, TaskStatus.SUCCESS),
            created_at_ms=evt.ts,
            started_at_ms=evt.ts,
            ended_at_ms=evt.ts + (evt.duration_ms or 0),
            duration_ms=evt.duration_ms,
            preview=evt.preview,
            error=evt.reason if evt.status == "failed" else None,
        )
        task_store.insert(record)
    except Exception as e:
        logger.warning("Failed to persist heartbeat event to task_history: %s", e)


def get_heartbeat_history(agent_id: str, limit: int = 30) -> list[dict[str, Any]]:
    events = list(_get_events(agent_id))
    events = events[-limit:][::-1]
    return [
        {
            "ts": e.ts,
            "status": e.status,
            "reason": e.reason,
            "preview": e.preview,
            "duration_ms": e.duration_ms,
        }
        for e in events
    ]


class HeartbeatRunner:
    """为每个 Agent 管理周期性心跳任务（主会话、per-agent 配置）"""

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._config_version = 0

    async def start(self, agent_ids: list[str] | None = None) -> None:
        self._running = True
        ids = agent_ids or [a["id"] for a in list_agents()]
        for agent_id in ids:
            self._ensure_task(agent_id)
        logger.info(f"Heartbeat started for agents: {ids}")

    async def stop(self) -> None:
        self._running = False
        for agent_id, task in list(self._tasks.items()):
            task.cancel()
            logger.info(f"Heartbeat stopped for agent: {agent_id}")
        self._tasks.clear()

    def _ensure_task(self, agent_id: str) -> None:
        """确保心跳任务存在且正在运行，避免重复创建"""
        if agent_id in self._tasks:
            task = self._tasks[agent_id]
            if not task.done():
                return  # 任务存在且仍在运行，无需创建
            # 任务已完成，清理旧任务
            self._tasks.pop(agent_id, None)
        task = asyncio.create_task(self._heartbeat_loop(agent_id))
        self._tasks[agent_id] = task

    async def add_agent(self, agent_id: str) -> None:
        """新增 Agent 时加入心跳任务（供 API 调用）"""
        hb = get_heartbeat_config(agent_id)
        if hb.get("enabled") and hb.get("interval_seconds"):
            self._ensure_task(agent_id)

    def update_config(self) -> None:
        """配置热更新：根据当前 agents.list 与 heartbeat 配置调整任务"""
        self._config_version += 1
        ids = [a["id"] for a in list_agents()]
        for agent_id in ids:
            hb = get_heartbeat_config(agent_id)
            if not hb.get("enabled") or hb.get("interval_seconds") is None:
                task = self._tasks.pop(agent_id, None)
                if task:
                    task.cancel()
            else:
                self._ensure_task(agent_id)
        for agent_id in list(self._tasks.keys()):
            if agent_id not in ids:
                task = self._tasks.pop(agent_id, None)
                if task:
                    task.cancel()

    async def _heartbeat_loop(self, agent_id: str) -> None:
        last_run_at = 0.0
        while self._running:
            hb = get_heartbeat_config(agent_id)
            interval = hb.get("interval_seconds")
            if interval is None or interval <= 0:
                break
            try:
                poll_interval = 5
                waited = 0
                total_wait = max(1, int(interval))
                while waited < total_wait:
                    if not self._running:
                        break
                    step = min(poll_interval, total_wait - waited)
                    await asyncio.sleep(step)
                    waited += step
                    now = time.time()
                    if now - last_run_at >= interval:
                        break
                if not self._running:
                    break
                last_run_at = time.time()
                await self._run_heartbeat(agent_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error for {agent_id}: {e}")
                emit_heartbeat_event(
                    agent_id,
                    HeartbeatEvent(
                        ts=int(time.time() * 1000),
                        status="failed",
                        reason=str(e),
                        agent_id=agent_id,
                    ),
                )
                await asyncio.sleep(60)

    async def _run_heartbeat(self, agent_id: str) -> None:
        """在主会话上执行心跳；HEARTBEAT_OK 时 rollback 不持久化"""
        started = time.time()
        hb = get_heartbeat_config(agent_id)
        if not hb.get("enabled"):
            return
        session_id = session_manager.resolve_main_session_id(agent_id)
        workspace = resolve_agent_workspace(agent_id)
        heartbeat_md = workspace / "HEARTBEAT.md"

        # 静默时段
        active = hb.get("activeHours")
        from config import resolve_agent_config
        agent_cfg = resolve_agent_config(agent_id)
        tz = agent_cfg.get("user_timezone", "Asia/Shanghai")
        if not is_within_active_hours(active, tz):
            emit_heartbeat_event(
                agent_id,
                HeartbeatEvent(
                    ts=int(time.time() * 1000),
                    status="skipped",
                    reason="quiet-hours",
                    duration_ms=int((time.time() - started) * 1000),
                    agent_id=agent_id,
                ),
            )
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if heartbeat_md.exists():
            try:
                content = heartbeat_md.read_text(encoding="utf-8")
                if is_heartbeat_content_effectively_empty(content):
                    emit_heartbeat_event(
                        agent_id,
                        HeartbeatEvent(
                            ts=int(time.time() * 1000),
                            status="skipped",
                            reason="empty-heartbeat-file",
                            duration_ms=int((time.time() - started) * 1000),
                            agent_id=agent_id,
                        ),
                    )
                    return
            except Exception:
                pass

        prompt = hb.get("prompt") or DEFAULT_HEARTBEAT_PROMPT
        full_prompt = f"[心跳轮询] 当前时间: {now_str}。\n{prompt}"

        audit_logger.log(agent_id, "heartbeat_trigger", {"time": now_str})

        async def _handle_heartbeat_result(response: str) -> None:
            ack_max = hb.get("ackMaxChars", 300)
            should_skip, stripped = strip_heartbeat_token(response, max_ack_chars=ack_max)

            if should_skip:
                session_manager.rollback_last_turn(session_id, agent_id)
                status = "ok-empty" if not response.strip() else "ok-token"
                emit_heartbeat_event(
                    agent_id,
                    HeartbeatEvent(
                        ts=int(time.time() * 1000),
                        status=status,
                        duration_ms=int((time.time() - started) * 1000),
                        agent_id=agent_id,
                    ),
                )
                audit_logger.log(agent_id, "heartbeat_ok", {})
            else:
                target = hb.get("target", "webchat")
                if target == "webchat":
                    emit_heartbeat_event(
                        agent_id,
                        HeartbeatEvent(
                            ts=int(time.time() * 1000),
                            status="sent",
                            preview=stripped[:200] if stripped else None,
                            duration_ms=int((time.time() - started) * 1000),
                            agent_id=agent_id,
                        ),
                    )
                    from infra.event_bus import Events, event_bus
                    event_bus.emit(agent_id, Events.heartbeat_message(session_id=session_id, agent_id=agent_id))
                audit_logger.log(agent_id, "heartbeat_response", {"response": response[:500]})

        from sessions.session_dispatcher import PRIORITY_HEARTBEAT
        from sessions.session_work_delivery import session_work_delivery

        session_work_delivery.deliver(
            kind="heartbeat",
            priority=PRIORITY_HEARTBEAT,
            content=full_prompt,
            agent_id=agent_id,
            session_id=session_id,
            result_handler=_handle_heartbeat_result,
            on_failure=lambda: (
                emit_heartbeat_event(
                    agent_id,
                    HeartbeatEvent(
                        ts=int(time.time() * 1000),
                        status="skipped",
                        reason="session-busy",
                        duration_ms=int((time.time() - started) * 1000),
                        agent_id=agent_id,
                    ),
                ),
                audit_logger.log(agent_id, "heartbeat_skipped", {"reason": "session-busy"}),
            ),
        )

    @property
    def active_agents(self) -> list[str]:
        return list(self._tasks.keys())


heartbeat_runner = HeartbeatRunner()
