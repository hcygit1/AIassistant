"""Expiry cleanup and archive orchestration for subagent runs."""

from __future__ import annotations

import time
from typing import Any, Callable


class SubagentRunArchiveService:
    """Remove expired terminal runs and notify session archiving hooks."""

    def __init__(
        self,
        *,
        store: Any,
        state: Any,
        persist: Callable[[], None],
        now: Callable[[], float] | None = None,
        emit_event: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._store = store
        self._state = state
        self._persist = persist
        self._now = now or time.time
        self._emit_event = emit_event or self._emit_archived_event

    @staticmethod
    def _emit_archived_event(run_id: str, record: Any) -> None:
        try:
            from infra.event_bus import Events, event_bus

            event_bus.emit(
                record.requester_agent_id,
                Events.subagent_archived(
                    run_id=run_id,
                    child_session_key=record.child_session_key,
                ),
            )
        except Exception:
            pass

    def sweep_expired(
        self,
        on_expire: Callable[[Any], None] | None = None,
    ) -> int:
        now_ms = self._now() * 1000
        with self._store.locked_records() as runs:
            to_remove: list[tuple[str, Any]] = []
            for run_id, record in runs.items():
                if (
                    record.archive_at_ms is None
                    or record.archive_at_ms > now_ms
                ):
                    continue
                if record.ended_at is None:
                    continue
                self._state.mark_archived(record)
                to_remove.append((run_id, record))
            for run_id, _record in to_remove:
                runs.pop(run_id, None)

        for run_id, record in to_remove:
            try:
                self._emit_event(run_id, record)
            except Exception:
                pass
            if on_expire:
                try:
                    on_expire(record)
                except Exception:
                    pass
        if to_remove:
            self._persist()
        return len(to_remove)
