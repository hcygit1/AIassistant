"""Claim and dispatch due Cron jobs."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

from scheduler.cron_schedule import compute_next_run
from scheduler.cron_types import CronJob, CronStore


@dataclass(frozen=True, slots=True)
class ProcessDueResult:
    fired: int
    failed: int
    next_wake_at_ms: int


class CronDueProcessor:
    STALE_CLAIM_AFTER_MS = 5 * 60_000
    MAX_WAKE_DELAY_MS = 60_000

    def __init__(
        self,
        *,
        transaction: Callable[[], ContextManager[tuple[CronStore, Any]]],
        save_store: Callable[[CronStore, Any], None],
        deliver_job: Callable[..., int],
        finalize_claim: Callable[..., bool],
        next_run: Callable[[CronJob, int, int | None], int | None] = (
            compute_next_run
        ),
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._transaction = transaction
        self._save_store = save_store
        self._deliver_job = deliver_job
        self._finalize_claim = finalize_claim
        self._next_run = next_run
        self._token_factory = token_factory or (lambda: uuid.uuid4().hex)

    def process(self, current_ms: int) -> ProcessDueResult:
        fired = 0
        failed = 0
        claims: list[tuple[CronJob, str]] = []
        with self._transaction() as (store, path):
            for job in store.jobs:
                if job.active_run_token:
                    if (
                        job.last_run_at_ms is not None
                        and current_ms - job.last_run_at_ms
                        < self.STALE_CLAIM_AFTER_MS
                    ):
                        continue
                    schedule_unchanged = (
                        job.active_run_schedule_revision is None
                        or job.active_run_schedule_revision
                        == job.schedule_revision
                    )
                    stale_due_at_ms = job.active_run_due_at_ms
                    job.active_run_token = None
                    job.active_run_work_id = None
                    job.active_run_due_at_ms = None
                    job.active_run_schedule_revision = None
                    job.last_run_status = "error"
                    if (
                        stale_due_at_ms is not None
                        and schedule_unchanged
                    ):
                        job.next_run_at_ms = current_ms
                if not job.enabled:
                    continue
                if job.next_run_at_ms is None:
                    job.next_run_at_ms = self._next_run(
                        job,
                        current_ms,
                        job.last_run_at_ms,
                    )
                if (
                    job.next_run_at_ms is None
                    or job.next_run_at_ms > current_ms
                ):
                    continue
                token = self._token_factory()
                due_at_ms = job.next_run_at_ms
                job.active_run_token = token
                job.active_run_work_id = None
                job.active_run_due_at_ms = due_at_ms
                job.active_run_schedule_revision = job.schedule_revision
                job.last_run_at_ms = current_ms
                job.last_run_status = "running"
                if job.schedule.kind == "at":
                    job.next_run_at_ms = None
                else:
                    job.next_run_at_ms = self._next_run(
                        job,
                        current_ms,
                        current_ms,
                    )
                claims.append((copy.deepcopy(job), token))
            self._save_store(store, path)

        for claimed, token in claims:
            try:
                self._deliver_job(
                    claimed,
                    token,
                    attempted_at_ms=current_ms,
                )
            except Exception:
                failed += 1
                self._finalize_claim(
                    claimed.id,
                    token,
                    status="error",
                    attempted_at_ms=current_ms,
                )
            else:
                fired += 1

        with self._transaction() as (store, _):
            next_wake = current_ms + self.MAX_WAKE_DELAY_MS
            for job in store.jobs:
                if job.enabled and job.next_run_at_ms is not None:
                    next_wake = min(next_wake, job.next_run_at_ms)
        return ProcessDueResult(fired, failed, next_wake)
