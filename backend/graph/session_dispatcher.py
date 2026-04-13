"""统一会话调度器 — 系统消息（announce / heartbeat / cron）按优先级串行消费

所有非用户消息通过 dispatcher.submit() 入队，由单协程持锁消费。
用户消息仍走 chat.py 的 HTTP 同步路径，两者共享同一把 asyncio.Lock。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ANNOUNCE_TIMEOUT_SEC = 60
SYSTEM_TIMEOUT_SEC = 5


@dataclass(order=False)
class PendingTask:
    kind: str
    priority: int
    content: str
    agent_id: str
    session_id: str
    prompt_mode: str = "minimal"
    persist_role: str = "system"
    run_id: str | None = None
    created_at: float = field(default_factory=time.time)
    result_handler: Callable[[str], Awaitable[None]] | None = None
    on_success: Callable[[], Any] | None = None
    on_failure: Callable[[], Any] | None = None
    on_failure_async: Callable[[Exception], Awaitable[None]] | None = None


class SessionDispatcher:
    """每个 session 一个实例，后台单协程按优先级消费系统消息。"""

    def __init__(self, lock: asyncio.Lock):
        self._lock = lock
        self._queue: list[PendingTask] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def submit(self, task: PendingTask) -> int:
        self._queue.append(task)
        self._queue.sort(key=lambda t: (t.priority, t.created_at))
        self._wake.set()
        return len(self._queue)

    @property
    def pending_count(self) -> int:
        """待消费任务数（监控 / 排障时可读）。"""
        return len(self._queue)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._consume_loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    async def _consume_loop(self) -> None:
        while True:
            if not self._queue:
                self._wake.clear()
                await self._wake.wait()

            while self._queue:
                task = self._queue.pop(0)
                await self._execute(task)

    async def _execute(self, task: PendingTask) -> None:
        from graph.agent import agent_manager

        timeout = ANNOUNCE_TIMEOUT_SEC if task.kind == "announce" else SYSTEM_TIMEOUT_SEC
        lock_acquired = False

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            lock_acquired = True
        except asyncio.TimeoutError as te:
            logger.warning(
                "Dispatcher timeout acquiring lock for %s (priority=%d, timeout=%ds)",
                task.kind, task.priority, timeout,
            )
            if task.on_failure_async:
                try:
                    await task.on_failure_async(te)
                except Exception as e2:
                    logger.warning("on_failure_async (lock timeout): %s", e2)
            elif task.on_failure:
                _safe_call(task.on_failure)
            return

        try:
            response_parts: list[str] = []
            done_content: str | None = None
            async for event in agent_manager.astream(
                message=task.content,
                session_id=task.session_id,
                agent_id=task.agent_id,
                prompt_mode=task.prompt_mode,
                persist_input_role=task.persist_role,
            ):
                et = event.get("type")
                if et == "token":
                    response_parts.append(event.get("content", ""))
                elif et == "done":
                    c = event.get("content")
                    if isinstance(c, str) and c.strip():
                        done_content = c

            response = "".join(response_parts).strip()
            if done_content is not None and done_content.strip():
                response = done_content.strip()

            if task.result_handler:
                await task.result_handler(response)

            if task.on_success:
                _safe_call(task.on_success)

        except Exception as e:
            logger.error("Dispatcher execution failed for %s: %s", task.kind, e)
            if task.on_failure_async:
                try:
                    await task.on_failure_async(e)
                except Exception as e2:
                    logger.warning("on_failure_async failed: %s", e2)
            elif task.on_failure:
                _safe_call(task.on_failure)
        finally:
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass


def _safe_call(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception as e:
        logger.warning("Dispatcher callback error: %s", e)


class DispatcherManager:
    """全局管理器：每个 agent:session 一个 dispatcher。"""

    def __init__(self):
        self._dispatchers: dict[str, SessionDispatcher] = {}

    def get(self, agent_id: str, session_id: str, lock: asyncio.Lock) -> SessionDispatcher:
        key = f"{agent_id}:{session_id}"
        if key not in self._dispatchers:
            d = SessionDispatcher(lock=lock)
            d.start()
            self._dispatchers[key] = d
        return self._dispatchers[key]

    def cleanup(self, agent_id: str, session_id: str) -> None:
        key = f"{agent_id}:{session_id}"
        d = self._dispatchers.pop(key, None)
        if d:
            d.stop()


dispatcher_manager = DispatcherManager()
