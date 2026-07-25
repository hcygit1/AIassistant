"""Transactional CRUD operations for persisted Cron job definitions."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, ContextManager

from scheduler.cron_types import (
    CronJob,
    CronPayload,
    CronSchedule,
    CronStore,
)


class CronJobCatalog:
    def __init__(
        self,
        *,
        transaction: Callable[
            [], ContextManager[tuple[CronStore, Path]]
        ],
        save_store: Callable[[CronStore, Path], None],
        ensure_enabled: Callable[[], None],
        now_ms: Callable[[], int],
        id_factory: Callable[[], str],
        build_schedule: Callable[..., CronSchedule],
        build_payload: Callable[[dict[str, Any]], CronPayload],
        next_run: Callable[[CronJob, int, int | None], int | None],
        schedule_state: Callable[[CronJob], tuple[Any, ...]],
        not_found: Callable[[str], Exception],
    ) -> None:
        self._transaction = transaction
        self._save_store = save_store
        self._ensure_enabled = ensure_enabled
        self._now_ms = now_ms
        self._id_factory = id_factory
        self._build_schedule = build_schedule
        self._build_payload = build_payload
        self._next_run = next_run
        self._schedule_state = schedule_state
        self._not_found = not_found

    def list_jobs(
        self,
        *,
        agent_id: str | None = None,
    ) -> list[CronJob]:
        with self._transaction() as (store, _):
            jobs = store.jobs
            if agent_id is not None:
                jobs = [
                    job
                    for job in jobs
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
            job = self.find_in_store(
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
            raise self._not_found(job_id)
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
            job.next_run_at_ms = self._next_run(
                job,
                now_ms,
                None,
            )
            store.jobs.append(job)
            self._save_store(store, path)
            return copy.deepcopy(job)

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
            job = self.find_in_store(
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
            job.next_run_at_ms = self._next_run(
                job,
                now_ms,
                job.last_run_at_ms,
            )
            self._save_store(store, path)
            return copy.deepcopy(job)

    def delete_job(
        self,
        job_id: str,
        *,
        agent_id: str | None = None,
    ) -> bool:
        self._ensure_enabled()
        with self._transaction() as (store, path):
            job = self.find_in_store(
                store,
                job_id,
                agent_id=agent_id,
            )
            if job is None:
                raise self._not_found(job_id)
            store.jobs = [
                current
                for current in store.jobs
                if current.id != job.id
            ]
            self._save_store(store, path)
            return True

    @staticmethod
    def find_in_store(
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
