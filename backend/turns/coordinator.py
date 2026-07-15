from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict

from .runtime import TerminalTurnStatus, TerminalUserTurn, UserTurnRuntime


DEFAULT_MAX_TERMINAL_TURNS = 1000


class UserTurnCoordinator:
    """Authoritative lifecycle/state owner for user turns."""

    def __init__(self, max_terminal_turns: int = DEFAULT_MAX_TERMINAL_TURNS) -> None:
        self._runtimes: dict[str, UserTurnRuntime] = {}
        self._session_to_turn: dict[str, str] = {}
        self._terminal_turns: OrderedDict[str, TerminalUserTurn] = OrderedDict()
        self._max_terminal_turns = max(0, max_terminal_turns)

    def _session_key(self, agent_id: str, session_id: str) -> str:
        return f"{agent_id}:{session_id}"

    def _release_session(self, turn_id: str) -> None:
        runtime = self._runtimes.get(turn_id)
        if not runtime:
            return
        key = self._session_key(runtime.agent_id, runtime.session_id)
        if self._session_to_turn.get(key) == turn_id:
            self._session_to_turn.pop(key, None)

    def _finish_turn(
        self,
        turn_id: str,
        status: TerminalTurnStatus,
        *,
        error: str | None = None,
    ) -> None:
        runtime = self._runtimes.get(turn_id)
        if not runtime:
            return

        self._release_session(turn_id)
        runtime.status = status
        runtime.error = error
        runtime.execution_task = None
        self._runtimes.pop(turn_id, None)

        if self._max_terminal_turns == 0:
            return
        self._terminal_turns[turn_id] = TerminalUserTurn(
            turn_id=runtime.turn_id,
            agent_id=runtime.agent_id,
            session_id=runtime.session_id,
            status=status,
            created_at=runtime.created_at,
            finished_at=time.time(),
            error=runtime.error,
            cancel_reason=runtime.cancel_reason,
        )
        self._terminal_turns.move_to_end(turn_id)
        while len(self._terminal_turns) > self._max_terminal_turns:
            self._terminal_turns.popitem(last=False)

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
        self._terminal_turns.pop(resolved_turn_id, None)
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

    def get(self, turn_id: str) -> UserTurnRuntime | TerminalUserTurn | None:
        return self._runtimes.get(turn_id) or self._terminal_turns.get(turn_id)

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
        runtime = self.get(turn_id)
        return runtime.cancel_reason if runtime else None

    def set_done(self, turn_id: str) -> None:
        self._finish_turn(turn_id, "done")

    def set_error(self, turn_id: str, message: str) -> None:
        self._finish_turn(turn_id, "error", error=message)

    def set_cancelled(self, turn_id: str) -> None:
        self._finish_turn(turn_id, "cancelled")

    def clear_session(self, agent_id: str, session_id: str) -> None:
        key = self._session_key(agent_id, session_id)
        turn_id = self._session_to_turn.pop(key, None)
        if turn_id:
            self._runtimes.pop(turn_id, None)
        for retained_turn_id, retained in list(self._terminal_turns.items()):
            if retained.agent_id == agent_id and retained.session_id == session_id:
                self._terminal_turns.pop(retained_turn_id, None)


user_turn_coordinator = UserTurnCoordinator()
