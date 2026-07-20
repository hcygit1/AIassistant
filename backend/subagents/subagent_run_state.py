"""State transitions for subagent runs and result delivery."""

from __future__ import annotations

import time
from typing import Any, Callable

from infra.state_machine import (
    SUBAGENT_ANNOUNCE_TRANSITIONS,
    SUBAGENT_RUN_TRANSITIONS,
    transition,
)


class SubagentRunStateService:
    """Mutate run state while keeping persistence outside the state model."""

    MAX_ANNOUNCE_RETRIES = 3
    ANNOUNCE_RETRY_EXPIRE_MS = 5 * 60 * 1000

    def __init__(
        self,
        *,
        store: Any,
        persist: Callable[[], None],
        now: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._persist = persist
        self._now = now or time.time

    def mark_started(self, run_id: str) -> None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if not record or record.ended_at is not None:
                return
            record.started_at = self._now()
            transition(
                record,
                "state",
                "running",
                table=SUBAGENT_RUN_TRANSITIONS,
            )
        self._persist()

    def mark_completed(
        self,
        run_id: str,
        result_summary: str = "",
        outcome: str = "completed",
        terminal_reason: str | None = None,
    ) -> None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if not record or record.ended_at is not None:
                return
            record.ended_at = self._now()
            record.outcome = outcome
            record.result_summary = result_summary[:1000]
            transition(
                record,
                "state",
                "succeeded",
                table=SUBAGENT_RUN_TRANSITIONS,
            )
            record.terminal_reason = terminal_reason
            transition(
                record,
                "result_delivery_state",
                "pending",
                table=SUBAGENT_ANNOUNCE_TRANSITIONS,
            )
        self._persist()

    def mark_terminated(self, run_id: str, reason: str = "killed") -> None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if not record or record.ended_at is not None:
                return
            self.terminate_record(record, reason)
        self._persist()

    def terminate_record(self, record: Any, reason: str) -> None:
        record.ended_at = self._now()
        record.outcome = reason
        lowered = (reason or "").lower()
        if "timeout" in lowered:
            new_state = "timed_out"
        elif "killed" in lowered or "cancel" in lowered:
            new_state = "cancelled"
        elif "orphaned" in lowered:
            new_state = "orphaned"
        elif "restart-interrupted" in lowered:
            new_state = "interrupted"
        else:
            new_state = "failed"
        transition(
            record,
            "state",
            new_state,
            table=SUBAGENT_RUN_TRANSITIONS,
        )
        record.terminal_reason = reason

    def mark_announce_retry(self, run_id: str) -> bool:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if not record:
                return False
            now_ms = self._now() * 1000
            if record.announce_retry_count >= self.MAX_ANNOUNCE_RETRIES:
                return False
            if (
                record.ended_at
                and now_ms - record.ended_at * 1000
                > self.ANNOUNCE_RETRY_EXPIRE_MS
            ):
                return False
            record.announce_retry_count = (
                getattr(record, "announce_retry_count", 0) + 1
            )
            record.last_announce_retry_at = self._now()
            transition(
                record,
                "result_delivery_state",
                "retrying",
                table=SUBAGENT_ANNOUNCE_TRANSITIONS,
            )
        self._persist()
        return True

    def mark_result_delivery_delivered(self, run_id: str) -> None:
        self.set_result_delivery_state(run_id, "delivered")

    def mark_result_delivery_dropped(self, run_id: str) -> None:
        self.set_result_delivery_state(run_id, "dropped")

    def set_result_delivery_state(self, run_id: str, new_state: str) -> None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if not record:
                return
            transition(
                record,
                "result_delivery_state",
                new_state,
                table=SUBAGENT_ANNOUNCE_TRANSITIONS,
            )
        self._persist()

    def set_delivery_work_id(self, run_id: str, work_id: str | None) -> None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if not record:
                return
            record.delivery_work_id = (work_id or "").strip() or None
        self._persist()

    def mark_archived(self, record: Any) -> None:
        transition(
            record,
            "state",
            "archived",
            table=SUBAGENT_RUN_TRANSITIONS,
        )
        record.ended_at = self._now()
