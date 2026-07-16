"""Cron 调度器 — 到点触发提醒投递"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .cron_service import (
    CronService,
    compute_next_run,
    cron_service,
)
from .cron_types import CronJob

logger = logging.getLogger(__name__)


def _compute_next_run(
    job: CronJob,
    now_ms: int,
    last_run_ms: int | None = None,
) -> int | None:
    """兼容入口；调度计算由 CronService 模块统一实现。"""
    return compute_next_run(job, now_ms, last_run_ms)


class CronScheduler:
    """Cron 调度器：后台循环检查 due jobs，并将提醒直接投递为会话工作项。"""

    def __init__(
        self,
        store_path: Path | None = None,
        *,
        service: CronService | None = None,
        now_ms=None,
    ):
        if service is not None:
            self._service = service
        elif store_path is not None:
            self._service = CronService(
                resolve_store_path=lambda: store_path
            )
        else:
            self._service = cron_service
        self._now_ms = now_ms or (
            lambda: int(time.time() * 1000)
        )
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
                await asyncio.sleep(self.tick())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cron loop error: {e}")
                await asyncio.sleep(10)

    def tick(self) -> float:
        now_ms = self._now_ms()
        result = self._service.process_due_jobs(now_ms=now_ms)
        if result.fired or result.failed:
            logger.info(
                "Cron tick: fired=%d failed=%d",
                result.fired,
                result.failed,
            )
        return max(
            1,
            min(
                60,
                (result.next_wake_at_ms - now_ms) / 1000,
            ),
        )
