"""Read-only queries for subagent run records."""

from __future__ import annotations

import time
from typing import Any, Callable


class SubagentRunQueryService:
    """Provide locked snapshots without owning run mutations or persistence."""

    def __init__(
        self,
        *,
        store: Any,
        snapshot: Callable[[Any], Any],
        now: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._snapshot = snapshot
        self._now = now or time.time

    def get_run(self, run_id: str) -> Any | None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            return (
                self._snapshot(record)
                if record is not None
                else None
            )

    def list_runs(self) -> list[Any]:
        """Return snapshots in the store's canonical insertion order."""
        with self._store.locked_records() as runs:
            return [
                self._snapshot(record)
                for record in runs.values()
            ]

    def list_run_entries(self) -> list[tuple[str, Any]]:
        """Return canonical registry keys with record snapshots."""
        with self._store.locked_records() as runs:
            return [
                (run_id, self._snapshot(record))
                for run_id, record in runs.items()
            ]

    def list_runs_for_requester(
        self,
        requester_key: str,
        include_recent_minutes: int = 30,
    ) -> list[Any]:
        cutoff = self._now() - include_recent_minutes * 60
        with self._store.locked_records() as runs:
            records = [
                record
                for record in runs.values()
                if record.requester_session_key == requester_key
                and (
                    record.ended_at is None
                    or record.ended_at >= cutoff
                )
            ]
            records.sort(
                key=lambda record: record.created_at,
                reverse=True,
            )
            return [self._snapshot(record) for record in records]

    def count_active_for_requester(self, requester_key: str) -> int:
        with self._store.locked_records() as runs:
            return sum(
                1
                for record in runs.values()
                if (
                    record.requester_session_key == requester_key
                    and record.ended_at is None
                )
            )
