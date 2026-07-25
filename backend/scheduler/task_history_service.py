"""Unified read model and cancellation boundary for background tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scheduler.task_store import TaskKind, TaskStatus
from sessions.session_work_record import SessionWorkRecord


class TaskHistoryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TaskHistoryPage:
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class TaskCancellationResult:
    ok: bool
    status: str


_KINDS = {kind.value for kind in TaskKind}
_STATUSES = {status.value for status in TaskStatus}
_WORK_STATUS_TO_TASK_STATUS = {
    "queued": TaskStatus.PENDING.value,
    "running": TaskStatus.RUNNING.value,
    "done": TaskStatus.SUCCESS.value,
    "failed": TaskStatus.FAILED.value,
    "cancelled": TaskStatus.CANCELLED.value,
}
_TASK_STATUS_TO_WORK_STATUS = {
    task_status: work_status
    for work_status, task_status in _WORK_STATUS_TO_TASK_STATUS.items()
}


class TaskHistoryService:
    def __init__(
        self,
        *,
        task_store=None,
        work_store=None,
        dispatcher_manager=None,
        runtime=None,
    ) -> None:
        if task_store is None:
            from scheduler.task_store import task_store
        if runtime is not None and (
            work_store is not None or dispatcher_manager is not None
        ):
            raise ValueError(
                "runtime cannot be combined with legacy dependencies"
            )
        if (
            runtime is None
            and work_store is None
            and dispatcher_manager is None
        ):
            from sessions.session_work_runtime import session_work_runtime

            runtime = session_work_runtime
        if runtime is not None:
            work_store = runtime.work_store
            dispatcher_manager = runtime.dispatcher_manager
        else:
            if work_store is None:
                work_store = getattr(dispatcher_manager, "work_store", None)
            if work_store is None:
                from sessions.session_work_store import session_work_store

                work_store = session_work_store
            if dispatcher_manager is None:
                from sessions.session_dispatcher_manager import (
                    dispatcher_manager,
                )

        self._task_store = task_store
        self._work_store = work_store
        self._dispatcher_manager = dispatcher_manager

    def query(
        self,
        *,
        agent_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TaskHistoryPage:
        self._validate_filters(kind, status, limit, offset)
        fetch_limit = limit + offset
        items: list[dict[str, Any]] = []
        total = 0

        if kind in (None, TaskKind.HEARTBEAT.value):
            task_status = TaskStatus(status) if status else None
            task_rows = self._task_store.query(
                agent_id=agent_id,
                kind=TaskKind.HEARTBEAT,
                status=task_status,
                limit=fetch_limit,
                offset=0,
            )
            items.extend(task_rows)
            total += self._task_store.count(
                agent_id=agent_id,
                kind=TaskKind.HEARTBEAT,
                status=task_status,
            )

        work_query = self._work_query(kind, status)
        if work_query is not None:
            work_rows = self._work_store.query(
                agent_id=agent_id,
                limit=fetch_limit,
                offset=0,
                **work_query,
            )
            items.extend(self._work_to_item(row) for row in work_rows)
            total += self._work_store.count(
                agent_id=agent_id,
                **work_query,
            )

        items.sort(
            key=lambda item: (
                int(item.get("created_at_ms") or 0),
                str(item.get("id") or ""),
            ),
            reverse=True,
        )
        return TaskHistoryPage(
            items=items[offset:offset + limit],
            total=total,
            limit=limit,
            offset=offset,
        )

    def cancel(self, task_id: str) -> TaskCancellationResult:
        work = self._work_store.get(task_id)
        if work is not None:
            if work.status == "running":
                raise TaskHistoryError(
                    "running",
                    "Running session work cannot be cancelled",
                )
            if work.status != "queued":
                raise TaskHistoryError(
                    "not_cancellable",
                    f"Task {task_id} is already {work.status}",
                )
            if not self._work_store.cancel_queued(task_id):
                current = self._work_store.get(task_id)
                code = (
                    "running"
                    if current is not None and current.status == "running"
                    else "not_cancellable"
                )
                raise TaskHistoryError(
                    code,
                    f"Task {task_id} is no longer queued",
                )
            self._dispatcher_manager.cancel_work(
                work.agent_id,
                work.session_id,
                work.id,
            )
            return TaskCancellationResult(True, "cancelled")

        task = self._task_store.get(task_id)
        if task is not None:
            raise TaskHistoryError(
                "not_cancellable",
                f"Task {task_id} is already {task['status']}",
            )
        raise TaskHistoryError(
            "not_found",
            f"Task {task_id} not found",
        )

    @staticmethod
    def _validate_filters(
        kind: str | None,
        status: str | None,
        limit: int,
        offset: int,
    ) -> None:
        if kind is not None and kind not in _KINDS:
            raise TaskHistoryError("invalid_filter", f"Unknown task kind: {kind}")
        if status is not None and status not in _STATUSES:
            raise TaskHistoryError(
                "invalid_filter",
                f"Unknown task status: {status}",
            )
        if limit < 1 or offset < 0:
            raise TaskHistoryError(
                "invalid_filter",
                "limit must be positive and offset cannot be negative",
            )

    @staticmethod
    def _work_query(
        kind: str | None,
        status: str | None,
    ) -> dict[str, Any] | None:
        native_status = (
            _TASK_STATUS_TO_WORK_STATUS.get(status)
            if status is not None
            else None
        )
        if status is not None and native_status is None:
            return None
        query: dict[str, Any] = {"status": native_status}
        if kind is None:
            query["kinds"] = ["cron", "announce"]
        elif kind == TaskKind.CRON.value:
            query["kind"] = "cron"
            query["exclude_run_id_prefix"] = "reminder-"
        elif kind == TaskKind.REMINDER.value:
            query["kind"] = "cron"
            query["run_id_prefix"] = "reminder-"
        elif kind == TaskKind.SYSTEM.value:
            query["kind"] = "announce"
        else:
            return None
        return query

    @staticmethod
    def _work_to_item(record: SessionWorkRecord) -> dict[str, Any]:
        kind = record.kind
        if kind == "cron" and (record.run_id or "").startswith(
            "reminder-"
        ):
            kind = TaskKind.REMINDER.value
        elif kind == "announce":
            kind = TaskKind.SYSTEM.value
        started_at = record.started_at_ms
        ended_at = record.finished_at_ms
        duration = (
            ended_at - started_at
            if started_at is not None and ended_at is not None
            else None
        )
        return {
            "id": record.id,
            "kind": kind,
            "agent_id": record.agent_id,
            "name": (
                f"{kind}:{record.run_id}"
                if record.run_id
                else kind
            ),
            "status": _WORK_STATUS_TO_TASK_STATUS.get(
                record.status,
                TaskStatus.FAILED.value,
            ),
            "created_at_ms": record.created_at_ms,
            "started_at_ms": started_at,
            "ended_at_ms": ended_at,
            "duration_ms": duration,
            "retry_count": 0,
            "max_retries": 0,
            "error": record.last_error,
            "preview": record.content[:200] or None,
            "source_job_id": record.run_id,
        }


task_history_service = TaskHistoryService()
