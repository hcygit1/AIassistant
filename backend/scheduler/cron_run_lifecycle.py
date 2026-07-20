"""Cron claim delivery, work binding, recovery, and terminal settlement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ContextManager

from scheduler.cron_types import CronJob, CronStore


class CronRunLifecycle:
    def __init__(
        self,
        *,
        transaction: Callable[
            [],
            ContextManager[tuple[CronStore, Path]],
        ],
        save_store: Callable[[CronStore, Path], None],
        list_jobs: Callable[[], list[CronJob]],
        deliver: Callable[..., int],
        retry_delay_ms: int,
    ) -> None:
        self._transaction = transaction
        self._save_store = save_store
        self._list_jobs = list_jobs
        self._deliver = deliver
        self._retry_delay_ms = retry_delay_ms

    def finalize_claim(
        self,
        job_id: str,
        token: str,
        *,
        status: str,
        attempted_at_ms: int | None = None,
    ) -> bool:
        with self._transaction() as (store, path):
            job = self._find_job(store, job_id)
            if job is None or job.active_run_token != token:
                return False
            due_at_ms = job.active_run_due_at_ms
            schedule_unchanged = (
                job.active_run_schedule_revision is None
                or job.active_run_schedule_revision == job.schedule_revision
            )
            job.active_run_token = None
            job.active_run_work_id = None
            job.active_run_due_at_ms = None
            job.active_run_schedule_revision = None
            job.last_run_status = status
            if (
                due_at_ms is not None
                and schedule_unchanged
                and job.schedule.kind == "at"
            ):
                if status == "ok":
                    job.enabled = False
                    job.next_run_at_ms = None
                    if job.delete_after_run:
                        store.jobs = [
                            current
                            for current in store.jobs
                            if current.id != job.id
                        ]
                else:
                    job.next_run_at_ms = (
                        (attempted_at_ms or due_at_ms)
                        + self._retry_delay_ms
                    )
            self._save_store(store, path)
            return True

    def bind_claim_work(
        self,
        job_id: str,
        token: str,
        work_id: str,
    ) -> bool:
        normalized_work_id = (work_id or "").strip()
        if not normalized_work_id:
            return False
        with self._transaction() as (store, path):
            job = self._find_job(store, job_id)
            if job is None or job.active_run_token != token:
                return False
            job.active_run_work_id = normalized_work_id
            self._save_store(store, path)
            return True

    def claim_callbacks(
        self,
        job_id: str,
        token: str,
        *,
        attempted_at_ms: int | None,
    ) -> dict[str, Callable[..., Any]]:
        return {
            "on_success": lambda: self.finalize_claim(
                job_id,
                token,
                status="ok",
                attempted_at_ms=attempted_at_ms,
            ),
            "on_failure": lambda: self.finalize_claim(
                job_id,
                token,
                status="error",
                attempted_at_ms=attempted_at_ms,
            ),
            "on_cancel": lambda: self.finalize_claim(
                job_id,
                token,
                status="error",
                attempted_at_ms=attempted_at_ms,
            ),
        }

    def recovery_callbacks(
        self,
        job_id: str,
        work_id: str,
    ) -> dict[str, Callable[..., Any]] | None:
        normalized_work_id = (work_id or "").strip()
        if not normalized_work_id:
            return None
        with self._transaction() as (store, path):
            job = self._find_job(store, job_id)
            if job is None or not job.active_run_token:
                return {}
            if job.active_run_work_id not in (
                None,
                normalized_work_id,
            ):
                return None
            if job.active_run_work_id is None:
                job.active_run_work_id = normalized_work_id
                self._save_store(store, path)
            token = job.active_run_token
            attempted_at_ms = job.last_run_at_ms
        return self.claim_callbacks(
            job_id,
            token,
            attempted_at_ms=attempted_at_ms,
        )

    def reconcile_active_work(
        self,
        get_work: Callable[[str], Any],
    ) -> int:
        reconciled = 0
        for job in self._list_jobs():
            token = job.active_run_token
            work_id = job.active_run_work_id
            if not token or not work_id:
                continue
            record = get_work(work_id)
            if record is None:
                continue
            work_status = str(getattr(record, "status", "") or "")
            if work_status == "done":
                status = "ok"
            elif work_status in ("failed", "cancelled"):
                status = "error"
            else:
                continue
            if self.finalize_claim(
                job.id,
                token,
                status=status,
                attempted_at_ms=job.last_run_at_ms,
            ):
                reconciled += 1
        return reconciled

    def deliver_job(
        self,
        job: CronJob,
        token: str,
        *,
        attempted_at_ms: int | None,
    ) -> int:
        callbacks = self.claim_callbacks(
            job.id,
            token,
            attempted_at_ms=attempted_at_ms,
        )
        return self._deliver(
            agent_id=job.agent_id or "main",
            text=job.payload.text,
            run_id=job.id,
            on_record_created=lambda record: self.bind_claim_work(
                job.id,
                token,
                getattr(record, "id", ""),
            ),
            **callbacks,
        )

    @staticmethod
    def _find_job(store: CronStore, job_id: str) -> CronJob | None:
        return next((job for job in store.jobs if job.id == job_id), None)
