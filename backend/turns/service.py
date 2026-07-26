from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException

from .events import TurnEvent


class UserTurnService:
    """Application-facing orchestration layer for user turns."""

    def __init__(
        self,
        *,
        coordinator: Any | None = None,
        lock_manager: Any | None = None,
        dispatcher_manager: Any | None = None,
        runtime: Any | None = None,
    ) -> None:
        if runtime is not None and (
            lock_manager is not None or dispatcher_manager is not None
        ):
            raise ValueError(
                "runtime cannot be combined with legacy dependencies"
            )
        self._coordinator = coordinator
        self._runtime = runtime
        self._dispatcher_manager = dispatcher_manager
        self._lock_manager = lock_manager
        if self._lock_manager is None and dispatcher_manager is not None:
            self._lock_manager = getattr(
                dispatcher_manager,
                "lock_manager",
                None,
            )

    @property
    def coordinator(self) -> Any:
        if self._coordinator is None:
            from turns.coordinator import user_turn_coordinator

            return user_turn_coordinator
        return self._coordinator

    @property
    def runtime(self) -> Any | None:
        if self._runtime is not None:
            return self._runtime
        if (
            self._lock_manager is not None
            or self._dispatcher_manager is not None
        ):
            return None
        from sessions.session_work_runtime import session_work_runtime

        return session_work_runtime

    @property
    def lock_manager(self) -> Any:
        runtime = self.runtime
        if runtime is not None:
            return runtime.lock_manager
        if self._lock_manager is None:
            from sessions.session_lock_manager import session_lock_manager

            return session_lock_manager
        return self._lock_manager

    @property
    def dispatcher_manager(self) -> Any:
        runtime = self.runtime
        if runtime is not None:
            return runtime.dispatcher_manager
        if self._dispatcher_manager is None:
            from sessions.session_dispatcher_manager import dispatcher_manager

            return dispatcher_manager
        return self._dispatcher_manager

    def _get_dispatcher(self, agent_id: str, session_id: str) -> Any:
        session_lock = self.lock_manager.get_lock(agent_id, session_id)
        return self.dispatcher_manager.get(
            agent_id,
            session_id,
            session_lock.lock,
        )

    async def submit(self, message: str, agent_id: str, session_id: str) -> dict:
        from sessions.session_work_item import SessionWorkItem
        from sessions.session_work_queue import PRIORITY_USER

        cleaned = (message or "").strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="message is required")

        if self.coordinator.has_active_user_turn(agent_id, session_id):
            raise HTTPException(
                status_code=409,
                detail="A user turn is already queued or running for this session",
            )

        try:
            runtime = self.coordinator.create_queued(agent_id, session_id)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        dispatcher = self._get_dispatcher(agent_id, session_id)
        task = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content=cleaned,
            agent_id=agent_id,
            session_id=session_id,
            prompt_mode="full",
            persist_role="user",
            turn_id=runtime.turn_id,
            stream_queue=runtime.stream_queue,
        )
        dispatcher.submit(task)
        position = dispatcher.turn_queue_position(runtime.turn_id) or 1
        return {
            "turn_id": runtime.turn_id,
            "position": position,
            "status": "queued",
            "session_id": session_id,
        }

    async def status(self, turn_id: str) -> dict:
        runtime = self.coordinator.get(turn_id)
        if not runtime:
            raise HTTPException(status_code=404, detail="unknown turn_id")

        position = 0
        if runtime.status == "queued":
            dispatcher = self._get_dispatcher(
                runtime.agent_id,
                runtime.session_id,
            )
            pos = dispatcher.turn_queue_position(turn_id)
            position = pos if pos is not None else 0

        return {
            "turn_id": runtime.turn_id,
            "status": runtime.status,
            "position": position,
            "session_id": runtime.session_id,
            "agent_id": runtime.agent_id,
            "error": runtime.error,
        }

    async def pending(self, agent_id: str, session_id: str) -> dict:
        runtime = self.coordinator.get_pending_for_session(agent_id, session_id)
        if not runtime:
            return {"turn_id": None, "status": None, "session_id": session_id}

        dispatcher = self._get_dispatcher(agent_id, session_id)
        position = 0
        if runtime.status == "queued":
            pos = dispatcher.turn_queue_position(runtime.turn_id)
            position = pos if pos is not None else 0
        return {
            "turn_id": runtime.turn_id,
            "status": runtime.status,
            "position": position,
            "session_id": runtime.session_id,
            "agent_id": runtime.agent_id,
        }

    async def stream(self, turn_id: str) -> AsyncGenerator[TurnEvent, None]:
        runtime = self.coordinator.get(turn_id)
        if not runtime:
            yield TurnEvent.error("unknown turn_id")
            return

        if runtime.status == "error":
            yield TurnEvent.error(runtime.error or "user turn failed")
            return
        if runtime.status in ("done", "cancelled"):
            return

        while True:
            item = await runtime.stream_queue.get()
            if item is None:
                break
            yield item

    async def abort(
        self,
        agent_id: str,
        session_id: str,
        *,
        turn_id: str = "",
        user_initiated: bool = True,
    ) -> dict:
        if (turn_id or "").strip():
            current = self.coordinator.current_turn_id_for_session(
                agent_id,
                session_id,
            )
            if current and current != turn_id.strip():
                raise HTTPException(
                    status_code=409,
                    detail="turn_id does not match active session turn",
                )

        aborted = self.coordinator.abort_turn(
            agent_id,
            session_id,
            turn_id=turn_id,
            user_initiated=user_initiated,
        )
        return {"aborted": bool(aborted)}


def _create_default_user_turn_service() -> UserTurnService:
    from sessions.session_work_runtime import session_work_runtime

    return UserTurnService(runtime=session_work_runtime)


user_turn_service = _create_default_user_turn_service()
