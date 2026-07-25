"""Manual Cron claim and immediate wake commands."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

from scheduler.cron_errors import CronServiceError
from scheduler.cron_job_catalog import CronJobCatalog
from scheduler.cron_types import CronStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunReceipt:
    job_id: str
    queue_position: int


class CronRunCommands:
    def __init__(
        self,
        *,
        transaction: Callable[
            [], ContextManager[tuple[CronStore, Path]]
        ],
        save_store: Callable[[CronStore, Path], None],
        ensure_enabled: Callable[[], None],
        now_ms: Callable[[], int],
        token_factory: Callable[[], str],
        run_lifecycle: Any,
        deliver: Callable[..., int],
    ) -> None:
        self._transaction = transaction
        self._save_store = save_store
        self._ensure_enabled = ensure_enabled
        self._now_ms = now_ms
        self._token_factory = token_factory
        self._run_lifecycle = run_lifecycle
        self._deliver = deliver

    def trigger_job(
        self,
        job_id: str,
        *,
        agent_id: str | None = None,
    ) -> RunReceipt:
        self._ensure_enabled()
        now_ms = self._now_ms()
        token = self._token_factory()
        with self._transaction() as (store, path):
            job = CronJobCatalog.find_in_store(
                store,
                job_id,
                agent_id=agent_id,
            )
            if job is None:
                raise CronServiceError(
                    "not_found",
                    f"Job {job_id} not found",
                )
            if job.active_run_token:
                raise CronServiceError(
                    "busy",
                    f"Job {job_id} is already being triggered",
                )
            job.active_run_token = token
            job.active_run_work_id = None
            job.active_run_due_at_ms = None
            job.active_run_schedule_revision = job.schedule_revision
            job.last_run_at_ms = now_ms
            job.last_run_status = "running"
            claimed = copy.deepcopy(job)
            self._save_store(store, path)

        try:
            position = self._run_lifecycle.deliver_job(
                claimed,
                token,
                attempted_at_ms=now_ms,
            )
        except Exception as exc:
            try:
                self._run_lifecycle.finalize_claim(
                    claimed.id,
                    token,
                    status="error",
                )
            except Exception:
                logger.exception(
                    "Failed to finalize manual Cron claim job=%s token=%s",
                    claimed.id,
                    token,
                )
            raise CronServiceError(
                "delivery_failed",
                f"Failed to trigger {job_id}: {exc}",
            ) from exc
        return RunReceipt(claimed.id, position)

    def wake(self, *, agent_id: str, text: str) -> RunReceipt:
        self._ensure_enabled()
        reminder_text = (text or "").strip()
        if not reminder_text:
            raise CronServiceError(
                "invalid_payload",
                "reminder text is required",
            )
        position = self._deliver(
            agent_id=agent_id or "main",
            text=reminder_text,
            run_id="cron:wake",
        )
        return RunReceipt("cron:wake", position)
