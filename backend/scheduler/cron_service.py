"""Cron 任务定义、调度计算与触发编排。"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator

from scheduler.cron_errors import CronServiceError
from scheduler.cron_run_lifecycle import CronRunLifecycle
from scheduler.cron_schedule import (
    build_payload as build_cron_payload,
    build_schedule as build_cron_schedule,
    compute_next_run,
    schedule_state as cron_schedule_state,
)
from scheduler.cron_types import (
    CronJob,
    CronPayload,
    CronSchedule,
    CronStore,
)


@dataclass(frozen=True, slots=True)
class RunReceipt:
    job_id: str
    queue_position: int


@dataclass(frozen=True, slots=True)
class ProcessDueResult:
    fired: int
    failed: int
    next_wake_at_ms: int


class CronService:
    RETRY_DELAY_MS = 60_000

    def __init__(
        self,
        *,
        load_store: Callable[[Path], CronStore] | None = None,
        save_store: Callable[[CronStore, Path], None] | None = None,
        resolve_store_path: Callable[[], Path] | None = None,
        store_transaction: (
            Callable[[Path], ContextManager[None]] | None
        ) = None,
        is_enabled: Callable[[], bool] | None = None,
        deliver: Callable[..., int] | None = None,
        now_ms: Callable[[], int] | None = None,
        id_factory: Callable[[], str] | None = None,
        default_timezone: Callable[[], str] | None = None,
        run_lifecycle: CronRunLifecycle | None = None,
    ) -> None:
        if load_store is None:
            from scheduler.cron_store import load_cron_store

            load_store = load_cron_store
        if save_store is None:
            from scheduler.cron_store import save_cron_store

            save_store = save_cron_store
        if resolve_store_path is None:
            resolve_store_path = self._default_store_path
        if store_transaction is None:
            from scheduler.cron_store import cron_store_transaction

            store_transaction = cron_store_transaction
        if is_enabled is None:
            from config import is_cron_enabled

            is_enabled = is_cron_enabled
        if deliver is None:
            from system_messages.reminder_delivery import (
                reminder_delivery_service,
            )

            deliver = (
                reminder_delivery_service.deliver_cron_reminder
            )

        self._load_store = load_store
        self._save_store = save_store
        self._resolve_store_path = resolve_store_path
        self._store_transaction = store_transaction
        self._is_enabled = is_enabled
        self._deliver = deliver
        self._now_ms = now_ms or (
            lambda: int(time.time() * 1000)
        )
        self._id_factory = id_factory or (
            lambda: uuid.uuid4().hex[:12]
        )
        self._default_timezone = (
            default_timezone or self._resolve_default_timezone
        )
        self._lock = threading.RLock()
        self._run_lifecycle = run_lifecycle or CronRunLifecycle(
            transaction=lambda: self._transaction(),
            save_store=lambda store, path: self._save(store, path),
            list_jobs=lambda: self.list_jobs(),
            deliver=lambda **kwargs: self._deliver(**kwargs),
            retry_delay_ms=self.RETRY_DELAY_MS,
        )

    def is_enabled(self) -> bool:
        return bool(self._is_enabled())

    @staticmethod
    def _default_store_path() -> Path:
        from config import get_config
        from scheduler.cron_store import resolve_cron_store_path

        config = get_config() or {}
        override = (config.get("cron") or {}).get("store")
        return resolve_cron_store_path(override)

    @staticmethod
    def _resolve_default_timezone() -> str:
        from config import get_config

        config = get_config() or {}
        return str(
            config.get("agents", {})
            .get("defaults", {})
            .get("user_timezone", "UTC")
            or "UTC"
        )

    def list_jobs(
        self,
        *,
        agent_id: str | None = None,
    ) -> list[CronJob]:
        with self._transaction() as (store, _):
            jobs = store.jobs
            if agent_id is not None:
                jobs = [
                    job for job in jobs
                    if (job.agent_id or "main") == agent_id
                ]
            return copy.deepcopy(jobs)

    def find_job(
        self,
        job_id: str,
        *,
        agent_id: str | None = None,
    ) -> CronJob | None:
        with self._transaction() as (store, _):
            job = self._find(
                store,
                job_id,
                agent_id=agent_id,
            )
            return copy.deepcopy(job) if job is not None else None

    def get_job(
        self,
        job_id: str,
        *,
        agent_id: str | None = None,
    ) -> CronJob:
        job = self.find_job(job_id, agent_id=agent_id)
        if job is None:
            raise CronServiceError(
                "not_found",
                f"Job {job_id} not found",
            )
        return job

    def create_job(
        self,
        *,
        name: str,
        agent_id: str = "main",
        schedule: dict[str, Any],
        payload: dict[str, Any],
        description: str = "",
        enabled: bool = True,
        delete_after_run: bool = False,
        id_prefix: str = "cron",
    ) -> CronJob:
        self._ensure_enabled()
        now_ms = self._now_ms()
        cron_schedule = self._build_schedule(
            schedule,
            now_ms=now_ms,
        )
        cron_payload = self._build_payload(payload)
        with self._transaction() as (store, path):
            job = CronJob(
                id=f"{id_prefix}-{self._id_factory()}",
                name=(name or "").strip(),
                description=(description or "").strip(),
                agent_id=(agent_id or "main").strip() or "main",
                enabled=bool(enabled),
                delete_after_run=bool(delete_after_run),
                schedule=cron_schedule,
                payload=cron_payload,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
            job.next_run_at_ms = compute_next_run(
                job,
                now_ms,
                None,
            )
            store.jobs.append(job)
            self._save(store, path)
            return copy.deepcopy(job)

    def create_reminder(
        self,
        *,
        text: str,
        at: str,
        agent_id: str = "main",
    ) -> CronJob:
        reminder_text = (text or "").strip()
        return self.create_job(
            name=f"提醒: {reminder_text[:40]}",
            description=reminder_text,
            agent_id=agent_id,
            enabled=True,
            delete_after_run=True,
            schedule={"kind": "at", "at": at},
            payload={
                "kind": "systemEvent",
                "text": reminder_text,
            },
            id_prefix="reminder",
        )

    def update_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        agent_id: str | None = None,
        enabled: bool | None = None,
        delete_after_run: bool | None = None,
        schedule: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        scope_agent_id: str | None = None,
    ) -> CronJob:
        self._ensure_enabled()
        now_ms = self._now_ms()
        with self._transaction() as (store, path):
            job = self._find(
                store,
                job_id,
                agent_id=scope_agent_id,
            )
            if job is None:
                raise self._not_found(job_id)
            schedule_state = self._schedule_state(job)
            if name is not None:
                job.name = str(name).strip()
            if description is not None:
                job.description = str(description).strip()
            if agent_id is not None:
                job.agent_id = str(agent_id).strip() or "main"
            if enabled is not None:
                job.enabled = bool(enabled)
            if delete_after_run is not None:
                job.delete_after_run = bool(delete_after_run)
            if schedule is not None:
                job.schedule = self._build_schedule(
                    schedule,
                    current=job.schedule,
                    now_ms=now_ms,
                )
            if payload is not None:
                job.payload = self._build_payload(payload)
            if self._schedule_state(job) != schedule_state:
                job.schedule_revision += 1
            job.updated_at_ms = now_ms
            job.next_run_at_ms = compute_next_run(
                job,
                now_ms,
                job.last_run_at_ms,
            )
            self._save(store, path)
            return copy.deepcopy(job)

    def delete_job(
        self,
        job_id: str,
        *,
        agent_id: str | None = None,
    ) -> bool:
        self._ensure_enabled()
        with self._transaction() as (store, path):
            job = self._find(store, job_id, agent_id=agent_id)
            if job is None:
                raise self._not_found(job_id)
            store.jobs = [
                current
                for current in store.jobs
                if current.id != job.id
            ]
            self._save(store, path)
            return True

    def trigger_job(
        self,
        job_id: str,
        *,
        agent_id: str | None = None,
    ) -> RunReceipt:
        self._ensure_enabled()
        now_ms = self._now_ms()
        token = uuid.uuid4().hex
        with self._transaction() as (store, path):
            job = self._find(store, job_id, agent_id=agent_id)
            if job is None:
                raise self._not_found(job_id)
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
            self._save(store, path)

        try:
            position = self._deliver_job(
                claimed,
                token,
                attempted_at_ms=now_ms,
            )
        except Exception as exc:
            self._finalize_claim(
                claimed.id,
                token,
                status="error",
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

    def process_due_jobs(
        self,
        *,
        now_ms: int | None = None,
    ) -> ProcessDueResult:
        current_ms = (
            self._now_ms() if now_ms is None else now_ms
        )
        fired = 0
        failed = 0
        claims: list[tuple[CronJob, str]] = []
        with self._transaction() as (store, path):
            for job in store.jobs:
                if job.active_run_token:
                    if (
                        job.last_run_at_ms is not None
                        and current_ms - job.last_run_at_ms
                        < 5 * 60_000
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
                    job.next_run_at_ms = compute_next_run(
                        job,
                        current_ms,
                        job.last_run_at_ms,
                    )
                if (
                    job.next_run_at_ms is None
                    or job.next_run_at_ms > current_ms
                ):
                    continue
                token = uuid.uuid4().hex
                due_at_ms = job.next_run_at_ms
                job.active_run_token = token
                job.active_run_work_id = None
                job.active_run_due_at_ms = due_at_ms
                job.active_run_schedule_revision = (
                    job.schedule_revision
                )
                job.last_run_at_ms = current_ms
                job.last_run_status = "running"
                if job.schedule.kind == "at":
                    job.next_run_at_ms = None
                else:
                    job.next_run_at_ms = compute_next_run(
                        job,
                        current_ms,
                        current_ms,
                    )
                claims.append((copy.deepcopy(job), token))
            self._save(store, path)

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
            next_wake = current_ms + 60_000
            for job in store.jobs:
                if (
                    job.enabled
                    and job.next_run_at_ms is not None
                ):
                    next_wake = min(
                        next_wake,
                        job.next_run_at_ms,
                    )
        return ProcessDueResult(fired, failed, next_wake)

    def _finalize_claim(
        self,
        job_id: str,
        token: str,
        *,
        status: str,
        attempted_at_ms: int | None = None,
    ) -> bool:
        return self._run_lifecycle.finalize_claim(
            job_id,
            token,
            status=status,
            attempted_at_ms=attempted_at_ms,
        )

    def _build_schedule(
        self,
        data: dict[str, Any],
        *,
        current: CronSchedule | None = None,
        now_ms: int,
    ) -> CronSchedule:
        return build_cron_schedule(
            data,
            current=current,
            now_ms=now_ms,
            default_timezone=self._default_timezone,
        )

    @staticmethod
    def _build_payload(data: dict[str, Any]) -> CronPayload:
        return build_cron_payload(data)

    def _bind_claim_work(
        self,
        job_id: str,
        token: str,
        work_id: str,
    ) -> bool:
        return self._run_lifecycle.bind_claim_work(
            job_id,
            token,
            work_id,
        )

    def _claim_callbacks(
        self,
        job_id: str,
        token: str,
        *,
        attempted_at_ms: int | None,
    ) -> dict[str, Callable[..., Any]]:
        return self._run_lifecycle.claim_callbacks(
            job_id,
            token,
            attempted_at_ms=attempted_at_ms,
        )

    def recovery_callbacks(
        self,
        job_id: str,
        work_id: str,
    ) -> dict[str, Callable[..., Any]] | None:
        return self._run_lifecycle.recovery_callbacks(
            job_id,
            work_id,
        )

    def reconcile_active_work(
        self,
        get_work: Callable[[str], Any],
    ) -> int:
        return self._run_lifecycle.reconcile_active_work(get_work)

    def _deliver_job(
        self,
        job: CronJob,
        token: str,
        *,
        attempted_at_ms: int | None,
    ) -> int:
        return self._run_lifecycle.deliver_job(
            job,
            token,
            attempted_at_ms=attempted_at_ms,
        )

    @staticmethod
    def _schedule_state(job: CronJob) -> tuple[Any, ...]:
        return cron_schedule_state(job)

    @contextmanager
    def _transaction(
        self,
    ) -> Iterator[tuple[CronStore, Path]]:
        path = self._resolve_store_path()
        with self._lock:
            with self._store_transaction(path):
                yield self._load_store(path), path

    def _save(self, store: CronStore, path: Path) -> None:
        self._save_store(store, path)

    @staticmethod
    def _find(
        store: CronStore,
        job_id: str,
        *,
        agent_id: str | None,
    ) -> CronJob | None:
        target = (job_id or "").strip()
        for job in store.jobs:
            if job.id != target:
                continue
            if (
                agent_id is not None
                and (job.agent_id or "main") != agent_id
            ):
                continue
            return job
        return None

    def _ensure_enabled(self) -> None:
        if not self._is_enabled():
            raise CronServiceError(
                "disabled",
                "cron is disabled "
                "(config.cron.enabled=false)",
            )

    @staticmethod
    def _not_found(job_id: str) -> CronServiceError:
        return CronServiceError(
            "not_found",
            f"Job {job_id} not found",
        )


cron_service = CronService()
