"""子 Agent 父子关系查询。"""

from __future__ import annotations

import time
from typing import Any, Callable


class SubagentRelationshipService:
    """基于 Registry 快照查询子 Agent 层级关系，不负责状态持久化。"""

    def __init__(self, list_runs: Callable[[], list[Any]]) -> None:
        self._list_runs = list_runs

    @staticmethod
    def session_key_from_child_session_key(child_session_key: str) -> str:
        """返回 child 会话可直接作为 requester 使用的 canonical key。"""
        return (child_session_key or "").strip()

    def get_requester_depth(self, requester_session_key: str) -> int:
        """Depth 0 = main, 1 = subagent, 2 = sub-subagent。"""
        key = (requester_session_key or "").strip()
        if not key:
            return 0
        for record in self._list_runs():
            if record.child_session_key == key:
                return max(0, record.spawn_depth)
        return 0

    def list_descendant_runs(
        self,
        root_session_key: str,
        include_recent_minutes: int = 60,
    ) -> list[Any]:
        """从 root 起 BFS 收集后代 runs，并按创建时间倒序返回。"""
        cutoff = time.time() - include_recent_minutes * 60
        root = (root_session_key or "").strip()
        if not root:
            return []

        pending = [root]
        visited: set[str] = {root}
        descendants: list[Any] = []
        while pending:
            requester = pending.pop(0)
            for record in self._list_runs():
                if record.requester_session_key != requester:
                    continue
                if (
                    record.ended_at is not None
                    and record.ended_at < cutoff
                ):
                    continue
                descendants.append(record)
                child_key = self.session_key_from_child_session_key(
                    record.child_session_key
                )
                if child_key and child_key not in visited:
                    visited.add(child_key)
                    pending.append(child_key)

        return sorted(
            descendants,
            key=lambda record: record.created_at,
            reverse=True,
        )

    def resolve_requester_for_child_session(
        self,
        child_session_key: str,
    ) -> tuple[str, str] | None:
        """返回 child 对应的 requester session key 和 agent id。"""
        key = (child_session_key or "").strip()
        if not key:
            return None

        best: Any | None = None
        for record in self._list_runs():
            if record.child_session_key != key:
                continue
            if best is None or record.created_at > best.created_at:
                best = record
        if best is None:
            return None
        return best.requester_session_key, best.requester_agent_id

    def count_active_descendant_runs(self, root_session_key: str) -> int:
        """统计 root 下尚未结束的后代 run 数量。"""
        root = (root_session_key or "").strip()
        if not root:
            return 0

        pending = [root]
        visited: set[str] = {root}
        count = 0
        while pending:
            requester = pending.pop(0)
            for record in self._list_runs():
                if record.requester_session_key != requester:
                    continue
                if record.ended_at is None:
                    count += 1
                child_key = self.session_key_from_child_session_key(
                    record.child_session_key
                )
                if child_key and child_key not in visited:
                    visited.add(child_key)
                    pending.append(child_key)
        return count
