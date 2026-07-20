"""子 Agent 注册表"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from subagents.subagent_relationships import SubagentRelationshipService
from subagents.subagent_run_lifecycle import SubagentRunLifecycleService
from subagents.subagent_run_state import SubagentRunStateService
from subagents.subagent_run_store import SubagentRunStore

logger = logging.getLogger(__name__)


class SubagentCapacityError(Exception):
    pass


@dataclass
class SubagentRunRecord:
    run_id: str
    child_session_key: str
    requester_session_key: str
    requester_agent_id: str
    target_agent_id: str
    task: str
    label: str | None = None
    model: str | None = None
    cleanup: Literal["delete", "keep"] = "keep"
    spawn_depth: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    outcome: str | None = None
    result_summary: str | None = None
    asyncio_task: Any = field(default=None, repr=False)
    # 创建后 N 分钟从 registry 删除并归档会话
    archive_at_ms: float | None = None
    # announce 重试
    announce_retry_count: int = 0
    last_announce_retry_at: float | None = None
    # webchat 展示/调度元数据
    state: str = "running"
    terminal_reason: str | None = None
    result_delivery_state: str = "pending"
    delivery_work_id: str | None = None


def _resolve_archive_after_ms() -> float | None:
    """从 config 读取 archive_after_minutes，返回毫秒数"""
    try:
        from config import get_config
        cfg = get_config()
        minutes = cfg.get("agents", {}).get("defaults", {}).get("subagents", {}).get("archive_after_minutes", 60)
        if not isinstance(minutes, (int, float)) or minutes <= 0:
            return None
        return max(1, int(minutes)) * 60_000
    except Exception:
        return 60 * 60_000  # 默认 60 分钟


class SubagentRegistry:
    def __init__(self, store: SubagentRunStore | None = None):
        self._store = store or SubagentRunStore()
        self._state = SubagentRunStateService(
            store=self._store,
            persist=lambda: self._persist_to_disk(),
        )
        self._relationships = SubagentRelationshipService(self.list_runs)
        self._lifecycle = SubagentRunLifecycleService(
            store=self._store,
            state=self._state,
            relationships=self._relationships,
            persist=lambda: self._persist_to_disk(),
            record_factory=SubagentRunRecord,
            snapshot=self._snapshot_record,
            resolve_archive_after_ms=_resolve_archive_after_ms,
            capacity_error=SubagentCapacityError,
        )
        self._restore_from_disk()

    @property
    def _runs(self) -> dict[str, SubagentRunRecord]:
        return self._store.records

    @property
    def _lock(self):
        return self._store.lock

    def _restore_from_disk(self) -> None:
        self._store.restore()

    def _persist_to_disk(self) -> None:
        self._store.persist()

    @staticmethod
    def _snapshot_record(
        record: SubagentRunRecord,
    ) -> SubagentRunRecord:
        return replace(record, asyncio_task=None)

    def register_run(
        self,
        run_id: str,
        child_session_key: str,
        requester_session_key: str,
        requester_agent_id: str,
        target_agent_id: str,
        task: str,
        label: str | None = None,
        model: str | None = None,
        cleanup: str = "keep",
        spawn_depth: int = 0,
        max_active_for_requester: int | None = None,
    ) -> SubagentRunRecord:
        return self._lifecycle.register_run(
            run_id=run_id,
            child_session_key=child_session_key,
            requester_session_key=requester_session_key,
            requester_agent_id=requester_agent_id,
            target_agent_id=target_agent_id,
            task=task,
            label=label,
            model=model,
            cleanup=cleanup,
            spawn_depth=spawn_depth,
            max_active_for_requester=max_active_for_requester,
        )

    def set_task(self, run_id: str, task: Any) -> bool:
        return self._lifecycle.set_task(run_id, task)

    def mark_started(self, run_id: str) -> None:
        self._state.mark_started(run_id)

    def mark_completed(
        self,
        run_id: str,
        result_summary: str = "",
        outcome: str = "completed",
        terminal_reason: str | None = None,
    ) -> None:
        self._state.mark_completed(
            run_id,
            result_summary=result_summary,
            outcome=outcome,
            terminal_reason=terminal_reason,
        )

    def mark_terminated(self, run_id: str, reason: str = "killed") -> None:
        self._state.mark_terminated(run_id, reason)

    def kill(self, run_id: str, cascade: bool = True) -> bool:
        """终止 run，cascade=True 时递归终止其子 runs"""
        return self._lifecycle.kill(run_id, cascade=cascade)

    def list_runs_for_requester(
        self, requester_key: str, include_recent_minutes: int = 30
    ) -> list[SubagentRunRecord]:
        cutoff = time.time() - include_recent_minutes * 60
        results = []
        for r in self.list_runs():
            if r.requester_session_key != requester_key:
                continue
            if r.ended_at is not None and r.ended_at < cutoff:
                continue
            results.append(r)
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results

    def count_active_for_requester(self, requester_key: str) -> int:
        return sum(
            1 for r in self.list_runs()
            if r.requester_session_key == requester_key and r.ended_at is None
        )

    def get_run(self, run_id: str) -> SubagentRunRecord | None:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            return (
                self._snapshot_record(record)
                if record is not None
                else None
            )

    def list_runs(self) -> list[SubagentRunRecord]:
        """Return a snapshot of all run records."""
        with self._store.locked_records() as runs:
            return [
                self._snapshot_record(record)
                for record in runs.values()
            ]

    def list_run_entries(
        self,
    ) -> list[tuple[str, SubagentRunRecord]]:
        """Return canonical registry keys with their records."""
        with self._store.locked_records() as runs:
            return [
                (run_id, self._snapshot_record(record))
                for run_id, record in runs.items()
            ]

    def remove_run(self, run_id: str) -> bool:
        """Remove one run and persist the registry change."""
        return self._lifecycle.remove_run(run_id)

    def mark_announce_retry(self, run_id: str) -> bool:
        """标记 announce 重试，返回是否可继续重试（未超限且未过期）"""
        return self._state.mark_announce_retry(run_id)

    def mark_result_delivery_delivered(self, run_id: str) -> None:
        self._state.mark_result_delivery_delivered(run_id)

    def mark_result_delivery_dropped(self, run_id: str) -> None:
        self._state.mark_result_delivery_dropped(run_id)

    def set_result_delivery_state(self, run_id: str, new_state: str) -> None:
        self._state.set_result_delivery_state(run_id, new_state)

    def set_delivery_work_id(self, run_id: str, work_id: str | None) -> None:
        self._state.set_delivery_work_id(run_id, work_id)

    def get_requester_depth(self, requester_session_key: str) -> int:
        return self._relationships.get_requester_depth(requester_session_key)

    @staticmethod
    def session_key_from_child_session_key(child_session_key: str) -> str:
        """返回 child 会话可直接作为 requester 使用的 canonical key。"""
        return SubagentRelationshipService.session_key_from_child_session_key(
            child_session_key
        )

    def list_descendant_runs(
        self, root_session_key: str, include_recent_minutes: int = 60
    ) -> list[SubagentRunRecord]:
        return self._relationships.list_descendant_runs(
            root_session_key,
            include_recent_minutes,
        )

    def resolve_requester_for_child_session(
        self, child_session_key: str
    ) -> tuple[str, str] | None:
        return self._relationships.resolve_requester_for_child_session(
            child_session_key
        )

    def count_active_descendant_runs(self, root_session_key: str) -> int:
        return self._relationships.count_active_descendant_runs(root_session_key)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        return self._lifecycle.cleanup_old(max_age_hours)

    def replace_active_run_for_steer(
        self,
        previous_run_id: str,
        next_run_id: str,
        task: str,
    ) -> SubagentRunRecord | None:
        """原子接管活跃 run，并在锁外取消旧任务。"""
        return self._lifecycle.replace_active_run_for_steer(
            previous_run_id,
            next_run_id,
            task,
        )

    def sweep_expired(
        self,
        on_expire: Callable[[SubagentRunRecord], None] | None = None,
    ) -> int:
        """删除 archive_at_ms 已到期的 run，并归档会话。

        每 60 秒由 subagent_archive 调用。on_expire 负责归档/删除会话文件。
        """
        now_ms = time.time() * 1000
        with self._store.locked_records() as runs:
            to_remove: list[tuple[str, SubagentRunRecord]] = []
            for rid, r in runs.items():
                if r.archive_at_ms is None or r.archive_at_ms > now_ms:
                    continue
                if r.ended_at is None:
                    continue
                self._state.mark_archived(r)
                to_remove.append((rid, r))
            for rid, _ in to_remove:
                runs.pop(rid, None)

        for rid, r in to_remove:
            # 发送事件通知前端
            try:
                from infra.event_bus import Events, event_bus
                event_bus.emit(r.requester_agent_id, Events.subagent_archived(run_id=rid, child_session_key=r.child_session_key))
            except Exception:
                pass
            if on_expire:
                try:
                    on_expire(r)
                except Exception:
                    pass
        if to_remove:
            self._persist_to_disk()
        return len(to_remove)


registry = SubagentRegistry()
