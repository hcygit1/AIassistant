"""Execution lifecycle for one queued user turn."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable

from turns.events import TurnEvent

if TYPE_CHECKING:
    from sessions.session_dispatcher import SessionWorkItem


logger = logging.getLogger(__name__)

UserStream = Callable[..., AsyncIterator[TurnEvent]]


class SessionUserTurnExecutor:
    """Run one user turn and project its terminal coordinator state."""

    def __init__(
        self,
        *,
        lock: asyncio.Lock,
        user_stream: UserStream,
        turn_coordinator: Any,
        set_current_turn: Callable[[str | None], None],
    ) -> None:
        self._lock = lock
        self._user_stream = user_stream
        self._turn_coordinator = turn_coordinator
        self._set_current_turn = set_current_turn

    async def execute(self, task: "SessionWorkItem") -> None:
        if not task.turn_id or not task.stream_queue:
            logger.error("user task missing turn_id or stream_queue")
            return

        lock_acquired = False
        try:
            await self._lock.acquire()
            lock_acquired = True
        except asyncio.CancelledError:
            self._turn_coordinator.set_cancelled(task.turn_id)
            await task.stream_queue.put(None)
            return

        self._set_current_turn(task.turn_id)
        self._turn_coordinator.set_running(task.turn_id)

        async def run_user_stream() -> tuple[str, str | None]:
            terminal_status = "done"
            terminal_error: str | None = None
            async for event in self._user_stream(
                task.content,
                task.session_id,
                task.agent_id,
                task.turn_id,
            ):
                await task.stream_queue.put(event)
                if terminal_status == "done" and event.type == "error":
                    terminal_status = "error"
                    terminal_error = event.error_message or "user turn failed"
                elif terminal_status == "done" and event.type == "aborted":
                    terminal_status = "cancelled"
            return terminal_status, terminal_error

        inner = asyncio.create_task(run_user_stream())
        self._turn_coordinator.bind_execution_task(task.turn_id, inner)

        try:
            terminal_status, terminal_error = await inner
        except asyncio.CancelledError:
            self._turn_coordinator.set_cancelled(task.turn_id)
        except Exception as exc:
            logger.error("User turn execution failed: %s", exc)
            await task.stream_queue.put(TurnEvent.error(str(exc)))
            self._turn_coordinator.set_error(task.turn_id, str(exc))
        else:
            if terminal_status == "error":
                self._turn_coordinator.set_error(
                    task.turn_id,
                    terminal_error or "user turn failed",
                )
            elif terminal_status == "cancelled":
                self._turn_coordinator.set_cancelled(task.turn_id)
            else:
                self._turn_coordinator.set_done(task.turn_id)
        finally:
            self._set_current_turn(None)
            self._turn_coordinator.bind_execution_task(task.turn_id, None)
            try:
                await task.stream_queue.put(None)
            except Exception:
                pass
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
