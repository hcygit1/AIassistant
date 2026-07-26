"""Application startup and shutdown orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI


logger = logging.getLogger(__name__)


class ApplicationLifecycle:
    """Coordinate backend runtime services without owning FastAPI routing."""

    def __init__(
        self,
        *,
        load_config: Callable[[], Any],
        setup_logging: Callable[[], None],
        scan_skills: Callable[[], Any],
        agent_manager: Any,
        skills_watcher: Any,
        configure_work_recovery: Callable[[], None],
        work_delivery_provider: Callable[[], Any],
        list_agents: Callable[[], list[dict[str, Any]]],
        heartbeat_runner: Any,
        start_subagent_archive: Callable[[], None],
        stop_subagent_archive: Callable[[], None],
        get_config: Callable[[], dict[str, Any]],
        cron_scheduler_factory: Callable[[], Any],
        resume_subagent_runs: Callable[[], Awaitable[None]],
        data_dir: Path,
        log: Any = logger,
    ) -> None:
        self._load_config = load_config
        self._setup_logging = setup_logging
        self._scan_skills = scan_skills
        self._agent_manager = agent_manager
        self._skills_watcher = skills_watcher
        self._configure_work_recovery = configure_work_recovery
        self._work_delivery_provider = work_delivery_provider
        self._list_agents = list_agents
        self._heartbeat_runner = heartbeat_runner
        self._start_subagent_archive = start_subagent_archive
        self._stop_subagent_archive = stop_subagent_archive
        self._get_config = get_config
        self._cron_scheduler_factory = cron_scheduler_factory
        self._resume_subagent_runs = resume_subagent_runs
        self._data_dir = data_dir
        self._logger = log

    async def start(self, application: FastAPI) -> None:
        application.state.cron_scheduler = None
        self._load_config()
        self._setup_logging()

        self._scan_skills()
        await self._agent_manager.initialize(str(self._data_dir))
        self._skills_watcher.start()

        work_delivery = self._work_delivery_provider()
        self._configure_work_recovery()
        failed_work_count = work_delivery.fail_unrecoverable_pending()
        if failed_work_count:
            self._logger.info(
                "Marked %s interrupted non-recoverable system work items as failed",
                failed_work_count,
            )

        agent_ids = [agent["id"] for agent in self._list_agents()]
        await self._heartbeat_runner.start(agent_ids)
        self._logger.info("Heartbeat started for agents: %s", agent_ids)

        recovered_work_count = work_delivery.recover_pending_work()
        if recovered_work_count:
            self._logger.info(
                "Recovered %s pending system work items",
                recovered_work_count,
            )

        self._start_subagent_archive()

        config = self._get_config()
        cron_config = config.get("cron") or {}
        if cron_config.get("enabled"):
            cron_scheduler = self._cron_scheduler_factory()
            application.state.cron_scheduler = cron_scheduler
            await cron_scheduler.start()
            self._logger.info("Cron scheduler started")

        try:
            await self._resume_subagent_runs()
        except Exception as error:
            self._logger.warning("Subagent resume failed: %s", error)

    async def stop(self, application: FastAPI) -> None:
        try:
            self._skills_watcher.stop()
        except Exception:
            self._logger.exception("Failed to stop skills watcher")

        try:
            self._stop_subagent_archive()
        except Exception:
            self._logger.exception("Failed to stop subagent archive")

        try:
            cron_scheduler = getattr(
                application.state,
                "cron_scheduler",
                None,
            )
            if cron_scheduler:
                await cron_scheduler.stop()
        except Exception:
            self._logger.exception("Failed to stop cron scheduler")

        try:
            await self._heartbeat_runner.stop()
            self._logger.info("Heartbeat stopped")
        except Exception:
            self._logger.exception("Failed to stop heartbeat")

        try:
            await self._agent_manager.close(timeout=30)
        except Exception:
            self._logger.exception("Failed to close agent manager")
