"""Execution lifecycle for one system session-work item."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sessions.session_dispatcher import SessionWorkItem


logger = logging.getLogger(__name__)

SystemStream = Callable[..., AsyncIterator[dict[str, Any]]]
SafeCall = Callable[[Callable[[], Any]], None]


def _default_safe_call(callback: Callable[[], Any]) -> None:
    try:
        callback()
    except Exception as exc:
        logger.warning("System work callback error: %s", exc)


class SessionSystemWorkExecutor:
    """Run and settle one Cron, Heartbeat, or Announce work item."""

    def __init__(
        self,
        *,
        lock: asyncio.Lock,
        work_store: Any,
        system_stream: SystemStream,
        is_stopping: Callable[[], bool],
        timeout_for_kind: Callable[[str], float],
        safe_call: SafeCall | None = None,
    ) -> None:
        self._lock = lock
        self._work_store = work_store
        self._system_stream = system_stream
        self._is_stopping = is_stopping
        self._timeout_for_kind = timeout_for_kind
        self._safe_call = safe_call or _default_safe_call

    def cancel_running(self, task: "SessionWorkItem") -> None:
        if task.work_id:
            try:
                self._work_store.mark_cancelled(task.work_id)
            except Exception as exc:
                logger.warning(
                    "Failed to cancel running session work %s: %s",
                    task.work_id,
                    exc,
                )
        if task.on_cancel:
            self._safe_call(task.on_cancel)

    async def execute(self, task: "SessionWorkItem") -> None:
        timeout = self._timeout_for_kind(task.kind)
        lock_acquired = False
        if task.work_id and not self._work_store.mark_running(task.work_id):
            return

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            lock_acquired = True
        except asyncio.CancelledError:
            self.cancel_running(task)
            raise
        except asyncio.TimeoutError as exc:
            logger.warning(
                "SessionSystemWorkExecutor timeout acquiring lock for %s "
                "(priority=%d, timeout=%ds)",
                task.kind,
                task.priority,
                timeout,
            )
            if task.work_id:
                self._work_store.mark_failed(task.work_id, str(exc))
            if task.on_failure_async:
                try:
                    await task.on_failure_async(exc)
                except Exception as callback_exc:
                    logger.warning(
                        "on_failure_async (lock timeout): %s",
                        callback_exc,
                    )
            elif task.on_failure:
                self._safe_call(task.on_failure)
            return

        try:
            response_parts: list[str] = []
            done_content: str | None = None
            stream = self._system_stream(
                message=task.content,
                session_id=task.session_id,
                agent_id=task.agent_id,
                prompt_mode=task.prompt_mode,
                persist_input_role=task.persist_role,
            )
            try:
                async for event in stream:
                    if self._is_stopping():
                        self.cancel_running(task)
                        return
                    event_type = event.get("type")
                    if event_type == "token":
                        response_parts.append(event.get("content", ""))
                    elif event_type == "done":
                        content = event.get("content")
                        if isinstance(content, str) and content.strip():
                            done_content = content
                    elif event_type == "error":
                        error = event.get("error")
                        message = (
                            error.strip()
                            if isinstance(error, str) and error.strip()
                            else "agent stream failed"
                        )
                        raise RuntimeError(message)
            finally:
                close_stream = getattr(stream, "aclose", None)
                if callable(close_stream):
                    try:
                        await close_stream()
                    except Exception as close_exc:
                        logger.warning("Failed to close system work stream: %s", close_exc)

            if self._is_stopping():
                self.cancel_running(task)
                return

            response = "".join(response_parts).strip()
            if done_content is not None and done_content.strip():
                response = done_content.strip()

            if task.result_handler:
                await task.result_handler(response)

            if task.work_id:
                self._work_store.mark_done(task.work_id)
            if task.on_success:
                self._safe_call(task.on_success)

        except asyncio.CancelledError:
            self.cancel_running(task)
            raise
        except Exception as exc:
            if self._is_stopping():
                self.cancel_running(task)
                return
            logger.error(
                "SessionSystemWorkExecutor failed for %s: %s",
                task.kind,
                exc,
            )
            if task.work_id:
                self._work_store.mark_failed(task.work_id, str(exc))
            if task.on_failure_async:
                try:
                    await task.on_failure_async(exc)
                except Exception as callback_exc:
                    logger.warning("on_failure_async failed: %s", callback_exc)
            elif task.on_failure:
                self._safe_call(task.on_failure)
        finally:
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
