from __future__ import annotations

import asyncio
import time
import uuid

from .runtime import UserTurnRuntime


class UserTurnCoordinator:
    """Authoritative lifecycle/state owner for user turns."""

    def __init__(self) -> None:
        self._runtimes: dict[str, UserTurnRuntime] = {}
        self._session_to_turn: dict[str, str] = {}

    def _session_key(self, agent_id: str, session_id: str) -> str:
        return f"{agent_id}:{session_id}"

    def _release_session(self, turn_id: str) -> None:
        runtime = self._runtimes.get(turn_id)
        if not runtime:
            return
        key = self._session_key(runtime.agent_id, runtime.session_id)
        if self._session_to_turn.get(key) == turn_id:
            self._session_to_turn.pop(key, None)

    def _purge_turn(self, turn_id: str) -> None:
        self._release_session(turn_id)
        self._runtimes.pop(turn_id, None)

    def create_queued(
        self,
        agent_id: str,
        session_id: str,
        stream_queue: asyncio.Queue[str | None] | None = None,
        turn_id: str | None = None,
    ) -> UserTurnRuntime:
        if self.has_active_user_turn(agent_id, session_id):
            raise RuntimeError("active user turn already exists for session")
        queue = stream_queue or asyncio.Queue()
        resolved_turn_id = turn_id or str(uuid.uuid4())
        runtime = UserTurnRuntime(
            turn_id=resolved_turn_id,
            agent_id=agent_id,
            session_id=session_id,
            status="queued",
            stream_queue=queue,
            created_at=time.time(),
        )
        self._runtimes[resolved_turn_id] = runtime
        self._session_to_turn[self._session_key(agent_id, session_id)] = resolved_turn_id
        return runtime

    def has_active_user_turn(self, agent_id: str, session_id: str) -> bool:
        key = self._session_key(agent_id, session_id)
        turn_id = self._session_to_turn.get(key)
        if not turn_id:
            return False
        runtime = self._runtimes.get(turn_id)
        if not runtime:
            self._session_to_turn.pop(key, None)
            return False
        return runtime.status in ("queued", "running")

    def get(self, turn_id: str) -> UserTurnRuntime | None:
        return self._runtimes.get(turn_id)

    def get_pending_for_session(self, agent_id: str, session_id: str) -> UserTurnRuntime | None:
        key = self._session_key(agent_id, session_id)
        turn_id = self._session_to_turn.get(key)
        if not turn_id:
            return None
        runtime = self._runtimes.get(turn_id)
        if not runtime:
            self._session_to_turn.pop(key, None)
            return None
        if runtime.status not in ("queued", "running"):
            return None
        return runtime

    def current_turn_id_for_session(self, agent_id: str, session_id: str) -> str | None:
        runtime = self.get_pending_for_session(agent_id, session_id)
        return runtime.turn_id if runtime else None

    def set_running(self, turn_id: str) -> None:
        runtime = self._runtimes.get(turn_id)
        if runtime:
            runtime.status = "running"
            runtime.cancel_reason = None

    def bind_execution_task(self, turn_id: str, task: asyncio.Task | None) -> None:
        runtime = self._runtimes.get(turn_id)
        if runtime:
            runtime.execution_task = task

    def abort_turn(
        self,
        agent_id: str,
        session_id: str,
        *,
        turn_id: str = "",
        user_initiated: bool = True,
    ) -> bool:
        target_turn_id = (turn_id or "").strip() or self.current_turn_id_for_session(agent_id, session_id)
        if not target_turn_id:
            return False

        runtime = self._runtimes.get(target_turn_id)
        if not runtime:
            return False
        if runtime.agent_id != agent_id or runtime.session_id != session_id:
            return False

        task = runtime.execution_task
        if not task or task.done():
            return False
        runtime.cancel_reason = "stopped_by_user" if user_initiated else "client_disconnected"
        task.cancel()
        return True

    def get_cancel_reason(self, turn_id: str) -> str | None:
        runtime = self._runtimes.get(turn_id)
        return runtime.cancel_reason if runtime else None

    def set_done(self, turn_id: str) -> None:
        runtime = self._runtimes.get(turn_id)
        if runtime:
            runtime.status = "done"
            runtime.execution_task = None
        self._purge_turn(turn_id)

    def set_error(self, turn_id: str, message: str) -> None:
        runtime = self._runtimes.get(turn_id)
        if runtime:
            runtime.status = "error"
            runtime.error = message
            runtime.execution_task = None
        self._purge_turn(turn_id)

    def set_cancelled(self, turn_id: str) -> None:
        runtime = self._runtimes.get(turn_id)
        if runtime:
            runtime.status = "cancelled"
            runtime.execution_task = None
        self._purge_turn(turn_id)

    def clear_session(self, agent_id: str, session_id: str) -> None:
        key = self._session_key(agent_id, session_id)
        turn_id = self._session_to_turn.pop(key, None)
        if turn_id:
            self._runtimes.pop(turn_id, None)


user_turn_coordinator = UserTurnCoordinator()
