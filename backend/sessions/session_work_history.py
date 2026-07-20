"""Read model for the session-work history endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sessions.session_work_runtime import (
    SessionWorkRuntime,
    session_work_runtime,
)


@dataclass(frozen=True, slots=True)
class SessionWorkHistoryPage:
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class SessionWorkHistoryService:
    """Query session-work records without exposing the runtime store to APIs."""

    def __init__(self, *, runtime: SessionWorkRuntime | None = None) -> None:
        self._runtime = runtime or session_work_runtime
        self._work_store = self._runtime.work_store

    def query(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SessionWorkHistoryPage:
        filters = {
            "kind": kind,
            "status": status,
            "agent_id": agent_id,
            "session_id": session_id,
            "run_id": run_id,
        }
        records = self._work_store.query(
            **filters,
            limit=limit,
            offset=offset,
        )
        total = self._work_store.count(**filters)
        return SessionWorkHistoryPage(
            items=[self._to_item(record) for record in records],
            total=total,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def _to_item(record: Any) -> dict[str, Any]:
        return {
            "id": record.id,
            "kind": record.kind,
            "agent_id": record.agent_id,
            "session_id": record.session_id,
            "run_id": record.run_id,
            "status": record.status,
            "recover_on_restart": record.recover_on_restart,
            "created_at_ms": record.created_at_ms,
            "started_at_ms": record.started_at_ms,
            "finished_at_ms": record.finished_at_ms,
            "last_error": record.last_error,
            "content_preview": record.content[:200],
        }


session_work_history_service = SessionWorkHistoryService()
