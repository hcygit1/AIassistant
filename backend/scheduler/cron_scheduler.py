"""Cron 调度器 — 到点触发提醒投递"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from croniter import croniter

from .cron_store import load_cron_store, save_cron_store, resolve_cron_store_path
from .cron_types import CronJob, CronSchedule, CronStore

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
    """Cron 调度器：后台循环检查 due jobs，并将提醒直接投递为会话工作项。"""

    def __init__(self, store_path: Path | None = None):
        self._store_path = store_path or resolve_cron_store_path()
        self._running = False
        self._task: asyncio.Task | None = None

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
        """触发 job：直接投递为 cron SessionWorkItem。"""
        if job.payload.kind != "systemEvent" or not job.payload.text.strip():
            return
        from system_messages.reminder_delivery import reminder_delivery_service

        agent_id = job.agent_id or "main"

        try:
            reminder_delivery_service.deliver_cron_reminder(
                agent_id=agent_id,
                text=job.payload.text,
                run_id=job.id,
            )
            logger.debug(f"Cron reminder submitted: {job.id}")
        except Exception as e:
            logger.error(f"Failed to submit cron reminder {job.id}: {e}")
            raise

        logger.info(f"Cron job {job.id} fired: {job.payload.text[:50]}...")
