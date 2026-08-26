"""Heartbeat 后台任务 — 主会话运行、HEARTBEAT_OK 剥离、事件存储"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable

from config import DEFAULT_HEARTBEAT_PROMPT, get_heartbeat_config, resolve_agent_workspace, list_agents
from system_messages.heartbeat_utils import (
    is_heartbeat_content_effectively_empty,
    is_within_active_hours,
)
from sessions.session_work_policy import deliver_system_work
from sessions.system_work_dependencies import (
    SystemWorkDependencies,
    system_work_dependencies,
)
from infra.audit_log import audit_logger
from system_messages.heartbeat_history import (
    HeartbeatEvent,
    emit_heartbeat_event,
    get_heartbeat_history,
)
from system_messages.heartbeat_run_lifecycle import HeartbeatRunLifecycle

logger = logging.getLogger(__name__)


def _emit_heartbeat_message(agent_id: str, session_id: str) -> None:
    from infra.event_bus import Events, event_bus

    event_bus.emit(
        agent_id,
        Events.heartbeat_message(
            session_id=session_id,
            agent_id=agent_id,
        ),
    )


class HeartbeatRunner:
    """为每个 Agent 管理周期性心跳任务（主会话、per-agent 配置）"""

    def __init__(
        self,
        *,
        session_manager: Any | None = None,
        work_delivery: Any | None = None,
        dependencies: SystemWorkDependencies | None = None,
        event_sink: Callable[[str, HeartbeatEvent], None] | None = None,
        run_lifecycle: HeartbeatRunLifecycle | None = None,
    ) -> None:
        self._dependencies = SystemWorkDependencies.resolve(
            dependencies=dependencies,
            session_manager=session_manager,
            work_delivery=work_delivery,
        )
        self._event_sink = event_sink
        self._run_lifecycle = run_lifecycle
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    @property
    def session_manager(self) -> Any:
        return self._dependencies.session_manager

    @property
    def work_delivery(self) -> Any:
        return self._dependencies.work_delivery

    @property
    def event_sink(self) -> Callable[[str, HeartbeatEvent], None]:
        return self._event_sink if self._event_sink is not None else emit_heartbeat_event

    @property
    def run_lifecycle(self) -> HeartbeatRunLifecycle:
        if self._run_lifecycle is not None:
            return self._run_lifecycle
        return HeartbeatRunLifecycle(
            rollback_last_turn=lambda session_id, agent_id: (
                self.session_manager.rollback_last_turn(
                    session_id,
                    agent_id,
                )
            ),
            event_sink=lambda agent_id, event: self.event_sink(
                agent_id,
                event,
            ),
            audit_event=lambda agent_id, event_type, data: audit_logger.log(
                agent_id,
                event_type,
                data,
            ),
            emit_webchat_message=_emit_heartbeat_message,
            now=lambda: time.time(),
        )

    async def start(self, agent_ids: list[str] | None = None) -> None:
        self._running = True
        ids = agent_ids or [a["id"] for a in list_agents()]
        for agent_id in ids:
            self._ensure_task(agent_id)
        logger.info(f"Heartbeat started for agents: {ids}")

    async def stop(self) -> None:
        self._running = False
        tasks = list(self._tasks.items())
        for agent_id, task in tasks:
            task.cancel()
            logger.info(f"Heartbeat stopped for agent: {agent_id}")
        self._tasks.clear()
        if tasks:
            await asyncio.gather(
                *(task for _, task in tasks),
                return_exceptions=True,
            )

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
        # A restart must not trigger an immediate catch-up heartbeat.  The
        # timer is in-memory, so the first run should be scheduled one full
        # interval after this process starts.
        last_run_at = time.time()
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
                self.event_sink(
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
        session_id = self.session_manager.resolve_main_session_id(agent_id)
        workspace = resolve_agent_workspace(agent_id)
        heartbeat_md = workspace / "HEARTBEAT.md"

        # 静默时段
        active = hb.get("activeHours")
        from config import resolve_agent_config
        agent_cfg = resolve_agent_config(agent_id)
        tz = agent_cfg.get("user_timezone", "Asia/Shanghai")
        if not is_within_active_hours(active, tz):
            self.event_sink(
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
                    self.event_sink(
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

        callbacks = self.run_lifecycle.callbacks(
            agent_id=agent_id,
            session_id=session_id,
            config=hb,
            started_at=started,
        )

        deliver_system_work(
            self.work_delivery,
            kind="heartbeat",
            content=full_prompt,
            agent_id=agent_id,
            session_id=session_id,
            **callbacks,
        )

    @property
    def active_agents(self) -> list[str]:
        return list(self._tasks.keys())


heartbeat_runner = HeartbeatRunner(dependencies=system_work_dependencies)
