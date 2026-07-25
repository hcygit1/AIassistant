"""统一会话调度器 — 用户消息与系统消息同队列、按优先级 + aging 串行消费

这个模块负责“会话正式工作项”的排队与执行。

这里的 SessionWorkItem 可以理解为 conversation work item：

- user：用户正式对话
- announce：子 Agent 向 requester 的正式汇报
- heartbeat：心跳检查形成的正式会话工作
- cron：带提醒内容的系统工作

职责：

- 维护每个 session 的正式工作队列
- 按优先级和 aging 串行执行
- 复用同一把 session 级锁，避免同一会话并发写入

非职责：

- 不保存用户 turn 的外部状态查询模型（那是 UserTurnCoordinator / user turn runtime 视图）
- 不负责前端 UI 广播（那是 event_bus）
"""

from __future__ import annotations

import asyncio
import logging
import time  # Compatibility patch path for session-work queue clocks.
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from sessions.session_work_policy import (
    PRIORITY_ANNOUNCE,
    PRIORITY_CRON,
    PRIORITY_HEARTBEAT,
)
from sessions.session_system_work_executor import SessionSystemWorkExecutor
from sessions.session_user_turn_executor import SessionUserTurnExecutor
from sessions.session_work_item import SessionWorkItem
from sessions.session_work_queue import (
    AGING_INTERVAL_SEC,
    MAX_AGING_BONUS,
    PRIORITY_MIN_SYSTEM,
    PRIORITY_USER,
    SessionWorkQueue,
)
from turns.events import TurnEvent

if TYPE_CHECKING:
    from sessions.session_work_store import SessionWorkStore

logger = logging.getLogger(__name__)

ANNOUNCE_TIMEOUT_SEC = 60
SYSTEM_TIMEOUT_SEC = 5

SystemStream = Callable[..., AsyncIterator[dict[str, Any]]]
UserStream = Callable[..., AsyncIterator[TurnEvent]]


def _default_system_stream(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
    """保持默认运行时入口，同时允许测试和宿主注入执行器。"""
    from runtime.agent import agent_manager

    return agent_manager.astream(**kwargs)


def _default_user_stream(
    message: str,
    session_id: str,
    agent_id: str,
    turn_id: str,
) -> AsyncIterator[TurnEvent]:
    from runtime.user_turn_stream import (
        UserTurnStreamDependencies,
        iter_user_turn_events,
    )

    return iter_user_turn_events(
        message,
        session_id,
        agent_id,
        turn_id,
        dependencies=UserTurnStreamDependencies.from_defaults(),
    )


def _default_turn_coordinator() -> Any:
    from turns.coordinator import user_turn_coordinator

    return user_turn_coordinator


def _default_work_store() -> "SessionWorkStore":
    from sessions.session_work_store import session_work_store

    return session_work_store


class SessionDispatcher:
    """每个 session 一个实例，后台单协程按优先级 + aging 消费队列。"""

    def __init__(
        self,
        lock: asyncio.Lock,
        work_store: "SessionWorkStore | None" = None,
        system_stream: SystemStream | None = None,
        user_stream: UserStream | None = None,
        turn_coordinator: Any | None = None,
        system_executor: SessionSystemWorkExecutor | None = None,
        user_executor: SessionUserTurnExecutor | None = None,
        work_queue: SessionWorkQueue | None = None,
    ):
        self._lock = lock
        self._work_store = (
            work_store if work_store is not None else _default_work_store()
        )
        self._system_stream = system_stream or _default_system_stream
        self._user_stream = user_stream or _default_user_stream
        self._turn_coordinator = (
            turn_coordinator
            if turn_coordinator is not None
            else _default_turn_coordinator()
        )
        self._work_queue = (
            work_queue if work_queue is not None else SessionWorkQueue()
        )
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._accepting = True
        self._stopping = False
        self._current_executing_turn_id: str | None = None
        self._system_executor = system_executor or SessionSystemWorkExecutor(
            lock=self._lock,
            work_store=self._work_store,
            system_stream=self._system_stream,
            is_stopping=lambda: self._stopping,
            timeout_for_kind=lambda kind: (
                ANNOUNCE_TIMEOUT_SEC
                if kind == "announce"
                else SYSTEM_TIMEOUT_SEC
            ),
            safe_call=_safe_call,
        )
        self._user_executor = user_executor or SessionUserTurnExecutor(
            lock=self._lock,
            user_stream=self._user_stream,
            turn_coordinator=self._turn_coordinator,
            set_current_turn=lambda turn_id: setattr(
                self,
                "_current_executing_turn_id",
                turn_id,
            ),
        )

    @property
    def current_executing_turn_id(self) -> str | None:
        return self._current_executing_turn_id

    def _effective_priority(self, task: SessionWorkItem) -> float:
        return self._work_queue.effective_priority(task)

    def _sort_key(self, t: SessionWorkItem) -> tuple[float, float]:
        return self._work_queue.sort_key(t)

    def submit(self, task: SessionWorkItem) -> int:
        if not self._accepting:
            raise RuntimeError("session dispatcher is closing")
        position = self._work_queue.submit(task)
        self._wake.set()
        return position

    def turn_queue_position(self, turn_id: str) -> int | None:
        """整队列中 1-based 位置；若不在队列中则 None。"""
        return self._work_queue.position(turn_id)

    def cancel_work(self, work_id: str) -> bool:
        removed = self._work_queue.remove(work_id)
        if removed is None:
            return False
        if removed.on_cancel:
            _safe_call(removed.on_cancel)
        return True

    @property
    def pending_count(self) -> int:
        return len(self._work_queue)

    def start(self) -> None:
        if not self._accepting:
            raise RuntimeError("session dispatcher is closing")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._consume_loop())

    def stop(self) -> asyncio.Task | None:
        self._accepting = False
        self._stopping = True
        self._cancel_queued_items()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        return task

    async def aclose(self) -> None:
        task = self.stop()
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
        finally:
            if self._task is task:
                self._task = None

    def _cancel_queued_items(self) -> None:
        pending = self._work_queue.drain()
        for task in pending:
            if task.work_id:
                try:
                    self._work_store.cancel_queued(task.work_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel queued session work %s: %s",
                        task.work_id,
                        exc,
                    )
            if task.turn_id:
                self._turn_coordinator.set_cancelled(task.turn_id)
                if task.stream_queue is not None:
                    try:
                        task.stream_queue.put_nowait(None)
                    except Exception:
                        pass
            if task.on_cancel:
                _safe_call(task.on_cancel)

    def _cancel_running_system_item(self, task: SessionWorkItem) -> None:
        self._system_executor.cancel_running(task)

    async def _consume_loop(self) -> None:
        while not self._stopping:
            if not self._work_queue:
                self._wake.clear()
                await self._wake.wait()
                if self._stopping:
                    return

            while self._work_queue and not self._stopping:
                task = self._work_queue.pop_next()
                if task is None:
                    break
                try:
                    await self._execute(task)
                finally:
                    del task

    async def _execute(self, task: SessionWorkItem) -> None:
        if task.kind == "user":
            await self._execute_user(task)
        else:
            await self._execute_system(task)

    async def _execute_user(self, task: SessionWorkItem) -> None:
        await self._user_executor.execute(task)

    async def _execute_system(self, task: SessionWorkItem) -> None:
        await self._system_executor.execute(task)


def _safe_call(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception as e:
        logger.warning("Dispatcher callback error: %s", e)


from sessions.session_dispatcher_manager import (  # noqa: E402
    DispatcherManager,
    dispatcher_manager,
)
