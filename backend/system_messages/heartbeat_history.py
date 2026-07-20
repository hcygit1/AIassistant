"""Heartbeat outcome history and task-history projection."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger(__name__)


def _get_max_events_per_agent() -> int:
    try:
        from config import get_config

        cfg = get_config()
        return (
            cfg.get("agents", {})
            .get("defaults", {})
            .get("heartbeat", {})
            .get("maxEvents", 50)
        )
    except Exception:
        return 50


@dataclass
class HeartbeatEvent:
    ts: int
    status: str
    reason: str | None = None
    preview: str | None = None
    duration_ms: int | None = None
    agent_id: str = ""


class HeartbeatHistory:
    """Own heartbeat outcome history and its durable task projection."""

    def __init__(
        self,
        *,
        task_store: Any | None = None,
        max_events_resolver: Callable[[], int] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._task_store = task_store
        self._max_events_resolver = (
            max_events_resolver or _get_max_events_per_agent
        )
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._events: dict[str, deque[HeartbeatEvent]] = {}

    @property
    def task_store(self) -> Any:
        if self._task_store is None:
            from scheduler.task_store import task_store

            return task_store
        return self._task_store

    def _events_for(self, agent_id: str) -> deque[HeartbeatEvent]:
        if agent_id not in self._events:
            self._events[agent_id] = deque(
                maxlen=self._max_events_resolver()
            )
        return self._events[agent_id]

    def emit(self, agent_id: str, event: HeartbeatEvent) -> None:
        self._events_for(agent_id).append(event)
        try:
            self.task_store.insert(self._to_task_record(agent_id, event))
        except Exception as exc:
            logger.warning(
                "Failed to persist heartbeat event to task_history: %s",
                exc,
            )

    def get(self, agent_id: str, limit: int = 30) -> list[dict[str, Any]]:
        events = list(self._events_for(agent_id))
        events = events[-limit:][::-1]
        return [
            {
                "ts": event.ts,
                "status": event.status,
                "reason": event.reason,
                "preview": event.preview,
                "duration_ms": event.duration_ms,
            }
            for event in events
        ]

    def _to_task_record(
        self,
        agent_id: str,
        event: HeartbeatEvent,
    ) -> Any:
        from scheduler.task_store import TaskKind, TaskRecord, TaskStatus

        status_map = {
            "ok-empty": TaskStatus.SUCCESS,
            "ok-token": TaskStatus.SUCCESS,
            "sent": TaskStatus.SUCCESS,
            "skipped": TaskStatus.CANCELLED,
            "failed": TaskStatus.FAILED,
        }
        return TaskRecord(
            id=self._id_factory(),
            kind=TaskKind.HEARTBEAT,
            agent_id=agent_id,
            name=f"heartbeat:{event.status}",
            status=status_map.get(event.status, TaskStatus.SUCCESS),
            created_at_ms=event.ts,
            started_at_ms=event.ts,
            ended_at_ms=event.ts + (event.duration_ms or 0),
            duration_ms=event.duration_ms,
            preview=event.preview,
            error=event.reason if event.status == "failed" else None,
        )


heartbeat_history = HeartbeatHistory()


def emit_heartbeat_event(agent_id: str, event: HeartbeatEvent) -> None:
    heartbeat_history.emit(agent_id, event)


def get_heartbeat_history(
    agent_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    return heartbeat_history.get(agent_id, limit=limit)
