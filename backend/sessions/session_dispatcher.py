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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from sessions.session_work_policy import (
    PRIORITY_ANNOUNCE,
    PRIORITY_CRON,
    PRIORITY_HEARTBEAT,
)
from sessions.session_dispatcher_factory import SessionDispatcherFactory
from sessions.session_system_work_executor import SessionSystemWorkExecutor
from sessions.session_user_turn_executor import SessionUserTurnExecutor
from sessions.session_work_queue import (
    AGING_INTERVAL_SEC,
    MAX_AGING_BONUS,
    PRIORITY_MIN_SYSTEM,
    PRIORITY_USER,
    SessionWorkQueue,
)
from turns.events import TurnEvent

if TYPE_CHECKING:
    from sessions.session_lock_manager import SessionLockManager
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


def _default_lock_manager() -> "SessionLockManager":
    from sessions.session_lock_manager import session_lock_manager

    return session_lock_manager


@dataclass(order=False)
class SessionWorkItem:
    """会话正式工作项。

    这是调度器内部真正要执行的载体，而不是 API 暴露给前端的 turn 状态。
    对用户消息来说，它与 user turn runtime/status 指向同一条业务请求的两个侧面：

    - user turn runtime/status：状态与可观测性
    - SessionWorkItem：调度与执行
    """

    kind: str
    priority: int
    content: str
    agent_id: str
    session_id: str
    prompt_mode: str = "minimal"
    persist_role: str = "system"
    run_id: str | None = None
    work_id: str | None = None
    created_at: float = field(default_factory=time.time)
    result_handler: Callable[[str], Awaitable[None]] | None = None
    on_success: Callable[[], Any] | None = None
    on_failure: Callable[[], Any] | None = None
    on_failure_async: Callable[[Exception], Awaitable[None]] | None = None
    on_cancel: Callable[[], Any] | None = None
    turn_id: str | None = None
    stream_queue: asyncio.Queue[TurnEvent | None] | None = None


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


class DispatcherManager:
    """全局管理器：每个 agent:session 一个 dispatcher。"""

    def __init__(
        self,
        work_store: "SessionWorkStore | None" = None,
        system_stream: SystemStream | None = None,
        user_stream: UserStream | None = None,
        turn_coordinator: Any | None = None,
        lock_manager: "SessionLockManager | None" = None,
        dispatcher_factory: SessionDispatcherFactory | None = None,
    ):
        self._dispatchers: dict[str, SessionDispatcher] = {}
        if dispatcher_factory is not None and any(
            dependency is not None
            for dependency in (
                work_store,
                system_stream,
                user_stream,
                turn_coordinator,
            )
        ):
            raise ValueError(
                "dispatcher_factory cannot be combined with legacy "
                "dispatcher dependencies"
            )
        self._lock_manager = (
            lock_manager if lock_manager is not None else _default_lock_manager()
        )
        if dispatcher_factory is None:
            resolved_store = (
                work_store if work_store is not None else _default_work_store()
            )
            dispatcher_factory = SessionDispatcherFactory(
                work_store=resolved_store,
                system_stream=system_stream,
                user_stream=user_stream,
                turn_coordinator=turn_coordinator,
            )
        self._dispatcher_factory = dispatcher_factory
        self._closing: dict[str, asyncio.Task | None] = {}

    @property
    def work_store(self) -> "SessionWorkStore":
        """Return the store shared by all dispatchers managed here."""
        return self._dispatcher_factory.work_store

    @property
    def lock_manager(self) -> "SessionLockManager":
        """Return the session locks shared by all dispatchers managed here."""
        return self._lock_manager

    def get(
        self,
        agent_id: str,
        session_id: str,
        lock: asyncio.Lock | None = None,
    ) -> SessionDispatcher:
        key = f"{agent_id}:{session_id}"
        if key not in self._dispatchers:
            if lock is None:
                lock = self._lock_manager.get_lock(agent_id, session_id).lock
            d = self._dispatcher_factory.create(lock=lock)
            d.start()
            self._dispatchers[key] = d
        return self._dispatchers[key]

    def cleanup(
        self,
        agent_id: str,
        session_id: str,
    ) -> asyncio.Task | None:
        return self.cleanup_when_closed(agent_id, session_id)

    def cleanup_when_closed(
        self,
        agent_id: str,
        session_id: str,
        *,
        on_closed: Callable[[], Any] | None = None,
    ) -> asyncio.Task | None:
        key = f"{agent_id}:{session_id}"
        dispatcher = self._dispatchers.get(key)
        if dispatcher is None:
            if on_closed:
                _safe_call(on_closed)
            return None

        if key in self._closing:
            task = self._closing[key]
            if task is not None and on_closed:
                task.add_done_callback(lambda _task: _safe_call(on_closed))
            return task

        task = dispatcher.stop()
        self._closing[key] = task

        def finish(_task: asyncio.Task | None = None) -> None:
            if self._dispatchers.get(key) is dispatcher:
                self._dispatchers.pop(key, None)
            self._closing.pop(key, None)
            if on_closed:
                _safe_call(on_closed)

        if task is None or task.done():
            finish(task)
        else:
            task.add_done_callback(finish)
        return task

    async def aclose_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> None:
        closed = asyncio.Event()
        self.cleanup_when_closed(
            agent_id,
            session_id,
            on_closed=closed.set,
        )
        await closed.wait()

    def cancel_work(
        self,
        agent_id: str,
        session_id: str,
        work_id: str,
    ) -> bool:
        dispatcher = self._dispatchers.get(
            f"{agent_id}:{session_id}"
        )
        if dispatcher is None:
            return False
        return dispatcher.cancel_work(work_id)


dispatcher_manager = DispatcherManager()
