"""Coordinated cleanup for one session runtime."""

from __future__ import annotations

from typing import Any


class SessionRuntimeCleanupService:
    """Close a dispatcher before dropping its shared lock and turn state."""

    def __init__(
        self,
        *,
        dispatcher_manager: Any,
        lock_manager: Any,
        turn_coordinator: Any,
    ) -> None:
        self._dispatcher_manager = dispatcher_manager
        self._lock_manager = lock_manager
        self._turn_coordinator = turn_coordinator

    def cleanup(self, agent_id: str, session_id: str) -> None:
        def finalize_cleanup() -> None:
            self._lock_manager.cleanup(agent_id, session_id)
            self._turn_coordinator.clear_session(agent_id, session_id)

        cleanup_when_closed = getattr(
            self._dispatcher_manager,
            "cleanup_when_closed",
            None,
        )
        if callable(cleanup_when_closed):
            cleanup_when_closed(
                agent_id,
                session_id,
                on_closed=finalize_cleanup,
            )
            return

        self._dispatcher_manager.cleanup(agent_id, session_id)
        finalize_cleanup()
