"""Ordered in-memory queue for one session's work items."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sessions.session_dispatcher import SessionWorkItem


# User work always wins; aging can improve system work only to this boundary.
PRIORITY_USER = -10
PRIORITY_MIN_SYSTEM = -9
AGING_INTERVAL_SEC = 30.0
MAX_AGING_BONUS = 3.0


class SessionWorkQueue:
    def __init__(
        self,
        *,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._now = now
        self._items: list[SessionWorkItem] = []

    def effective_priority(self, item: SessionWorkItem) -> float:
        if item.kind == "user":
            return float(PRIORITY_USER)
        now = self._now() if self._now is not None else time.time()
        age = now - item.created_at
        bonus = min(age / AGING_INTERVAL_SEC, MAX_AGING_BONUS)
        priority = float(item.priority) - bonus
        return max(priority, float(PRIORITY_MIN_SYSTEM))

    def sort_key(self, item: SessionWorkItem) -> tuple[float, float]:
        return (self.effective_priority(item), item.created_at)

    def submit(self, item: SessionWorkItem) -> int:
        self._items.append(item)
        self._sort()
        return len(self._items)

    def position(self, turn_id: str) -> int | None:
        self._sort()
        for index, item in enumerate(self._items, start=1):
            if item.turn_id == turn_id:
                return index
        return None

    def remove(self, work_id: str) -> SessionWorkItem | None:
        for index, item in enumerate(self._items):
            if item.work_id == work_id:
                return self._items.pop(index)
        return None

    def pop_next(self) -> SessionWorkItem | None:
        if not self._items:
            return None
        self._sort()
        return self._items.pop(0)

    def drain(self) -> list[SessionWorkItem]:
        pending = self._items
        self._items = []
        return pending

    def __len__(self) -> int:
        return len(self._items)

    def _sort(self) -> None:
        self._items.sort(key=self.sort_key)
