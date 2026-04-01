"""消息队列 — 会话级串行化 + followup 队列"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

def _get_followup_cap() -> int:
    """从配置获取followup队列上限"""
    try:
        from config import get_config
        cfg = get_config()
        return cfg.get("app", {}).get("messageQueue", {}).get("followupCap", 20)
    except Exception:
        return 20


def _get_debounce_ms() -> int:
    """从配置获取防抖时间（毫秒）"""
    try:
        from config import get_config
        cfg = get_config()
        return cfg.get("app", {}).get("messageQueue", {}).get("debounceMs", 1000)
    except Exception:
        return 1000


@dataclass
class FollowupItem:
    message: str
    timestamp: float = field(default_factory=time.time)


class SessionQueue:
    """每个 session 一个实例，保证串行执行 + followup 收集"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._followup: list[FollowupItem] = []
        self._busy = False
        self._active_task: asyncio.Task | None = None

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def pending_count(self) -> int:
        return len(self._followup)

    def enqueue_followup(self, message: str) -> int:
        followup_cap = _get_followup_cap()
        if len(self._followup) >= followup_cap:
            self._followup.pop(0)
        self._followup.append(FollowupItem(message=message))
        return len(self._followup)

    def clear_followups(self) -> int:
        n = len(self._followup)
        self._followup.clear()
        return n

    def drain_followup(self) -> str | None:
        if not self._followup:
            return None
        items = self._followup[:]
        self._followup.clear()
        if len(items) == 1:
            return items[0].message
        # Preserve timestamps for context understanding
        lines = []
        for i, item in enumerate(items, 1):
            ts = time.strftime("%H:%M:%S", time.localtime(item.timestamp))
            lines.append(f"[{ts}] {item.message}")
        return "\n".join(lines)

    async def acquire(self) -> None:
        await self._lock.acquire()
        self._busy = True

    def release(self) -> None:
        self._busy = False
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
        # Store whether this was user-initiated for the task to check
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
