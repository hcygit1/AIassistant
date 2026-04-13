"""消息队列 — 会话级串行化

每个 session 持有一把 asyncio.Lock，chat.py 和 SessionDispatcher 共享，
保证同一 session 的 Agent 调用串行执行。
"""

from __future__ import annotations

import asyncio


class SessionQueue:
    """每个 session 一个实例，提供串行锁和 abort 能力。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_task: asyncio.Task | None = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def is_busy(self) -> bool:
        # 与 chat / SessionDispatcher 共享同一把锁：任一方持锁即视为忙
        return self._lock.locked()

    async def acquire(self) -> None:
        await self._lock.acquire()
        self._user_aborted = False

    def release(self) -> None:
        self._active_task = None
        try:
            self._lock.release()
        except RuntimeError:
            pass

    def set_active_task(self, task: asyncio.Task | None) -> None:
        self._active_task = task

    def abort_active_task(self, user_initiated: bool = True) -> bool:
        task = self._active_task
        if not task:
            return False
        if task.done():
            return False
        self._user_aborted = user_initiated
        task.cancel()
        return True

    def was_user_aborted(self) -> bool:
        return getattr(self, "_user_aborted", True)


class MessageQueueManager:
    """全局消息队列管理器"""

    def __init__(self):
        self._queues: dict[str, SessionQueue] = {}

    def get_queue(self, agent_id: str, session_id: str) -> SessionQueue:
        key = f"{agent_id}:{session_id}"
        if key not in self._queues:
            self._queues[key] = SessionQueue()
        return self._queues[key]

    def is_session_busy(self, agent_id: str, session_id: str) -> bool:
        key = f"{agent_id}:{session_id}"
        queue = self._queues.get(key)
        return queue.is_busy if queue else False

    def cleanup(self, agent_id: str, session_id: str) -> None:
        key = f"{agent_id}:{session_id}"
        self._queues.pop(key, None)


message_queue_manager = MessageQueueManager()


def cleanup_session_runtime(agent_id: str, session_id: str) -> None:
    """释放会话的 SessionDispatcher 与 SessionQueue（会话删除或维护 prune 时调用）。

    须先停止 dispatcher（共享同一把 asyncio.Lock），再移除 SessionQueue。
    """
    from graph.session_dispatcher import dispatcher_manager

    dispatcher_manager.cleanup(agent_id, session_id)
    message_queue_manager.cleanup(agent_id, session_id)
