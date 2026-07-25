"""State transition rules for persisted session work."""

from __future__ import annotations

from collections.abc import Callable


ExecuteUpdate = Callable[[str, tuple[object, ...]], int]


class SessionWorkTransitions:
    def __init__(
        self,
        *,
        execute_update: ExecuteUpdate,
        now_ms: Callable[[], int],
    ) -> None:
        self._execute_update = execute_update
        self._now_ms = now_ms

    def mark_running(self, work_id: str) -> bool:
        rowcount = self._execute_update(
            """UPDATE session_work
            SET status='running', started_at_ms=?, last_error=NULL
            WHERE id=? AND status IN ('queued', 'running')""",
            (self._now_ms(), work_id),
        )
        return rowcount == 1

    def cancel_queued(self, work_id: str) -> bool:
        rowcount = self._execute_update(
            """UPDATE session_work
            SET status='cancelled', finished_at_ms=?
            WHERE id=? AND status='queued'""",
            (self._now_ms(), work_id),
        )
        return rowcount == 1

    def mark_cancelled(self, work_id: str) -> None:
        self._execute_update(
            """UPDATE session_work
            SET status='cancelled', finished_at_ms=?
            WHERE id=? AND status IN ('queued', 'running')""",
            (self._now_ms(), work_id),
        )

    def requeue_for_recovery(self, work_id: str) -> bool:
        rowcount = self._execute_update(
            """UPDATE session_work
            SET status='queued', started_at_ms=NULL,
                finished_at_ms=NULL, last_error=NULL
            WHERE id=? AND recover_on_restart=1
                AND status IN ('queued', 'running')""",
            (work_id,),
        )
        return rowcount == 1

    def fail_unrecoverable_pending(self, error: str) -> int:
        return self._execute_update(
            """UPDATE session_work
            SET status='failed', finished_at_ms=?, last_error=?
            WHERE recover_on_restart=0
                AND status IN ('queued', 'running')""",
            (self._now_ms(), error),
        )

    def mark_done(self, work_id: str) -> None:
        self._execute_update(
            """UPDATE session_work
            SET status='done', finished_at_ms=?, last_error=NULL
            WHERE id=?""",
            (self._now_ms(), work_id),
        )

    def mark_failed(self, work_id: str, error: str | None = None) -> None:
        self._execute_update(
            """UPDATE session_work
            SET status='failed', finished_at_ms=?, last_error=?
            WHERE id=?""",
            (self._now_ms(), error, work_id),
        )
