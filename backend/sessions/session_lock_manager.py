"""会话锁管理 — 会话级串行化

这个模块本质上是 session 级执行闸门。

它的核心职责不是维护业务 work item 顺序，而是：

- 为每个 session 提供一把共享 asyncio.Lock
- 让 chat.py 与 SessionDispatcher 通过同一把锁判断会话是否忙碌

真正的业务工作队列在 SessionDispatcher 中；
本模块更接近 execution gate / execution controller。
"""

from __future__ import annotations

import asyncio
from typing import Any


class SessionLock:
    """每个 session 一个实例，提供串行锁。

    这里管理的是“当前会话能不能执行”，
    而不是“有哪些业务任务排队等待执行”。
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def is_busy(self) -> bool:
        # 与 chat / SessionDispatcher 共享同一把锁：任一方持锁即视为忙
        return self._lock.locked()

    async def acquire(self) -> None:
        await self._lock.acquire()

    def release(self) -> None:
        try:
            self._lock.release()
        except RuntimeError:
            pass


class SessionLockManager:
    """全局 session 执行闸门管理器。"""

    def __init__(self):
        self._locks: dict[str, SessionLock] = {}

    def get_lock(self, agent_id: str, session_id: str) -> SessionLock:
        key = f"{agent_id}:{session_id}"
        if key not in self._locks:
            self._locks[key] = SessionLock()
        return self._locks[key]

    def is_session_busy(self, agent_id: str, session_id: str) -> bool:
        key = f"{agent_id}:{session_id}"
        session_lock = self._locks.get(key)
        return session_lock.is_busy if session_lock else False

    def cleanup(self, agent_id: str, session_id: str) -> None:
        key = f"{agent_id}:{session_id}"
        self._locks.pop(key, None)


session_lock_manager = SessionLockManager()


def cleanup_session_runtime(
    agent_id: str,
    session_id: str,
    *,
    dispatcher_manager: Any | None = None,
    lock_manager: SessionLockManager | None = None,
    turn_coordinator: Any | None = None,
) -> None:
    """释放会话的 SessionDispatcher 与 SessionLock（会话删除或维护 prune 时调用）。

    须先停止 dispatcher（共享同一把 asyncio.Lock），再移除 SessionLock。
    """
    if dispatcher_manager is None and lock_manager is None:
        from sessions.session_work_runtime import session_work_runtime

        dispatcher_manager = session_work_runtime.dispatcher_manager
        lock_manager = session_work_runtime.lock_manager
    elif dispatcher_manager is None:
        from sessions.session_dispatcher import dispatcher_manager as default_manager

        dispatcher_manager = default_manager
    if lock_manager is None:
        lock_manager = getattr(
            dispatcher_manager,
            "lock_manager",
            None,
        )
    if lock_manager is None:
        lock_manager = session_lock_manager
    if turn_coordinator is None:
        from turns.coordinator import user_turn_coordinator as default_coordinator

        turn_coordinator = default_coordinator

    def finalize_cleanup() -> None:
        lock_manager.cleanup(agent_id, session_id)
        turn_coordinator.clear_session(agent_id, session_id)

    cleanup_when_closed = getattr(
        dispatcher_manager,
        "cleanup_when_closed",
        None,
    )
    if callable(cleanup_when_closed):
        cleanup_when_closed(
            agent_id,
            session_id,
            on_closed=finalize_cleanup,
        )
    else:
        dispatcher_manager.cleanup(agent_id, session_id)
        finalize_cleanup()
