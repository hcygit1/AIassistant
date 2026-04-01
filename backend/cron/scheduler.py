"""Cron 调度器 — 到点触发 systemEvent 并唤醒心跳"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from croniter import croniter

from .store import load_cron_store, save_cron_store, resolve_cron_store_path
from .types import CronJob, CronSchedule, CronStore

logger = logging.getLogger(__name__)


def _compute_next_run(job: CronJob, now_ms: int, last_run_ms: int | None = None) -> int | None:
    """计算下次运行时间（毫秒）。支持 at、every、cron。"""
    if not job.enabled:
        return None
    sched = job.schedule
    if sched.kind == "at":
        if sched.at:
            try:
                from datetime import datetime, timezone
                at_dt = datetime.fromisoformat(sched.at.replace("Z", "+00:00"))
                at_ms = int(at_dt.timestamp() * 1000)
                if at_ms > now_ms:
                    return at_ms
            except (ValueError, TypeError) as e:
                logger.warning(f"Cron at {sched.at} parse error: {e}")
        return None
    if sched.kind == "every":
        every_ms = sched.every_ms or 0
        if every_ms <= 0:
            return None
        anchor = last_run_ms if last_run_ms is not None else job.created_at_ms or now_ms
        return anchor + every_ms
    if sched.kind == "cron" and sched.expr:
        try:
            from datetime import datetime, timezone
            now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
            it = croniter(sched.expr, now)
            next_dt = it.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception as e:
            logger.warning(f"Cron expr {sched.expr} error: {e}")
    return None


class CronScheduler:
    """Cron 调度器：后台循环检查 due jobs，触发 enqueue + request_heartbeat_now。"""

    def __init__(self, store_path: Path | None = None):
        self._store_path = store_path or resolve_cron_store_path()
        self._running = False
        self._task: asyncio.Task | None = None
        self._request_heartbeat_now: callable | None = None

    def set_request_heartbeat_now(self, fn: callable) -> None:
        """注入 request_heartbeat_now(agent_id, reason)。"""
        self._request_heartbeat_now = fn

    async def start(self) -> None:
        """启动调度循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Cron scheduler started")

    async def stop(self) -> None:
        """停止调度循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Cron scheduler stopped")

    async def _loop(self) -> None:
        """主循环：检查 due jobs，触发后重新计算 next_run。"""
        while self._running:
            try:
                store = load_cron_store(self._store_path)
                now_ms = int(time.time() * 1000)
                due_jobs: list[CronJob] = []
                for job in store.jobs:
                    if not job.enabled:
                        continue
                    next_ms = job.next_run_at_ms
                    if next_ms is None:
                        next_ms = _compute_next_run(job, now_ms, job.last_run_at_ms)
                        job.next_run_at_ms = next_ms
                    if next_ms is not None and next_ms <= now_ms:
                        due_jobs.append(job)
                if not store.jobs:
                    await asyncio.sleep(60)
                    continue
                to_remove: list[str] = []
                for job in due_jobs:
                    try:
                        await self._fire_job(job)
                        job.last_run_at_ms = now_ms
                        job.last_run_status = "ok"
                    except Exception as e:
                        logger.exception(f"Cron job {job.id} failed: {e}")
                        job.last_run_status = "error"
                        # 继续处理下一个 job，不影响其他任务
                    if job.schedule.kind == "at":
                        job.enabled = False
                        job.next_run_at_ms = None
                        if job.delete_after_run:
                            to_remove.append(job.id)
                    else:
                        job.next_run_at_ms = _compute_next_run(
                            job, now_ms, job.last_run_at_ms
                        )
                for jid in to_remove:
                    store.jobs = [j for j in store.jobs if j.id != jid]
                save_cron_store(store, self._store_path)
                # 计算最近下次触发时间，sleep 到那时（最多 60s 轮询一次）
                next_wake_ms = now_ms + 60_000
                for job in store.jobs:
                    if job.enabled and job.next_run_at_ms:
                        if job.next_run_at_ms < next_wake_ms:
                            next_wake_ms = job.next_run_at_ms
                sleep_s = max(1, min(60, (next_wake_ms - now_ms) / 1000))
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cron loop error: {e}")
                await asyncio.sleep(10)

    async def _fire_job(self, job: CronJob) -> None:
        """触发 job：enqueue_system_event + request_heartbeat_now。
        确保事件入队和心跳唤醒的一致性。
        """
        if job.payload.kind != "systemEvent" or not job.payload.text.strip():
            return
        from infra.system_events import enqueue_system_event
        from graph.session_manager import session_manager
        agent_id = job.agent_id or "main"
        main_sid = session_manager.resolve_main_session_id(agent_id)
        session_key = session_manager.session_key_from_session_id(agent_id, main_sid)

        # 标记是否成功入队
        event_enqueued = False
        heartbeat_ok = False

        try:
            enqueue_system_event(
                job.payload.text,
                session_key=session_key,
                context_key=f"cron:{job.id}",
            )
            event_enqueued = True
            logger.debug(f"Cron event enqueued: {job.id}")
        except Exception as e:
            logger.error(f"Failed to enqueue cron event {job.id}: {e}")
            return  # 入队失败，直接返回

        # 尝试唤醒心跳，带重试机制
        if self._request_heartbeat_now:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self._request_heartbeat_now(agent_id, f"cron:{job.id}")
                    heartbeat_ok = True
                    break
                except Exception as e:
                    logger.warning(f"request_heartbeat_now failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))  # 指数退避

            if not heartbeat_ok:
                # 心跳唤醒失败，记录错误但事件已入队，心跳会在下次轮询时处理
                logger.error(f"Failed to wake heartbeat for cron job {job.id} after {max_retries} attempts. "
                           f"Event is enqueued and will be processed on next heartbeat cycle.")
        else:
            logger.warning(f"request_heartbeat_now not configured for cron job {job.id}")

        logger.info(f"Cron job {job.id} fired: {job.payload.text[:50]}... "
                   f"(event_enqueued={event_enqueued}, heartbeat_ok={heartbeat_ok})")
