from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import HTTPException


class UserTurnService:
    """Application-facing orchestration layer for user turns."""

    async def submit(self, message: str, agent_id: str, session_id: str) -> dict:
        from sessions.session_lock_manager import session_lock_manager
        from sessions.session_dispatcher import PRIORITY_USER, SessionWorkItem, dispatcher_manager
        from turns.coordinator import user_turn_coordinator

        cleaned = (message or "").strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="message is required")

        if user_turn_coordinator.has_active_user_turn(agent_id, session_id):
            raise HTTPException(
                status_code=409,
                detail="A user turn is already queued or running for this session",
            )

        try:
            runtime = user_turn_coordinator.create_queued(agent_id, session_id)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        session_lock = session_lock_manager.get_lock(agent_id, session_id)
        dispatcher = dispatcher_manager.get(agent_id, session_id, session_lock.lock)
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
        from turns.coordinator import user_turn_coordinator

        runtime = user_turn_coordinator.get(turn_id)
        if not runtime:
            raise HTTPException(status_code=404, detail="unknown turn_id")

        position = 0
        if runtime.status == "queued":
            from sessions.session_lock_manager import session_lock_manager
            from sessions.session_dispatcher import dispatcher_manager

            session_lock = session_lock_manager.get_lock(
                runtime.agent_id,
                runtime.session_id,
            )
            dispatcher = dispatcher_manager.get(
                runtime.agent_id,
                runtime.session_id,
                session_lock.lock,
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
        from sessions.session_lock_manager import session_lock_manager
        from sessions.session_dispatcher import dispatcher_manager
        from turns.coordinator import user_turn_coordinator

        runtime = user_turn_coordinator.get_pending_for_session(agent_id, session_id)
        if not runtime:
            return {"turn_id": None, "status": None, "session_id": session_id}

        session_lock = session_lock_manager.get_lock(agent_id, session_id)
        dispatcher = dispatcher_manager.get(agent_id, session_id, session_lock.lock)
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

    async def stream(self, turn_id: str) -> AsyncGenerator[str, None]:
        from turns.coordinator import user_turn_coordinator

        runtime = user_turn_coordinator.get(turn_id)
        if not runtime:
            err = json.dumps({"type": "error", "error": "unknown turn_id"}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"
            return

        if runtime.status == "error":
            err = json.dumps({
                "type": "error",
                "error": runtime.error or "user turn failed",
            }, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"
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
        from turns.coordinator import user_turn_coordinator

        if (turn_id or "").strip():
            current = user_turn_coordinator.current_turn_id_for_session(agent_id, session_id)
            if current and current != turn_id.strip():
                raise HTTPException(
                    status_code=409,
                    detail="turn_id does not match active session turn",
                )

        aborted = user_turn_coordinator.abort_turn(
            agent_id,
            session_id,
            turn_id=turn_id,
            user_initiated=user_initiated,
        )
        return {"aborted": bool(aborted)}


user_turn_service = UserTurnService()
