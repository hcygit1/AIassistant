"""子 Agent 注册表"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from infra.state_machine import (
    SUBAGENT_ANNOUNCE_TRANSITIONS,
    SUBAGENT_RUN_TRANSITIONS,
    transition,
)

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
    def __init__(self):
        self._runs: dict[str, SubagentRunRecord] = {}
        self._lock = threading.RLock()
        self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        from subagents.subagent_registry_state import restore_registry_from_disk
        restore_registry_from_disk(self._runs, merge_only=False)

    def _persist_to_disk(self) -> None:
        from subagents.subagent_registry_state import save_registry_to_disk

        with self._lock:
            snapshot = dict(self._runs)
            save_registry_to_disk(snapshot)

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
        now = time.time()
        archive_after_ms = _resolve_archive_after_ms()
        archive_at_ms = (now * 1000 + archive_after_ms) if archive_after_ms else None

        record = SubagentRunRecord(
            run_id=run_id,
            child_session_key=child_session_key,
            requester_session_key=requester_session_key,
            requester_agent_id=requester_agent_id,
            target_agent_id=target_agent_id,
            task=task,
            label=label,
            model=model,
            cleanup=cleanup,  # type: ignore
            spawn_depth=spawn_depth,
            archive_at_ms=archive_at_ms,
        )
        with self._lock:
            if max_active_for_requester is not None:
                active = sum(
                    1
                    for current in self._runs.values()
                    if current.requester_session_key
                    == requester_session_key
                    and current.ended_at is None
                )
                if active >= max_active_for_requester:
                    raise SubagentCapacityError(
                        "active sub-agent capacity reached"
                    )
            self._runs[run_id] = record
            self._persist_to_disk()
        return self._snapshot_record(record)

    def set_task(self, run_id: str, task: Any) -> bool:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None and record.ended_at is None:
                record.asyncio_task = task
                return True

        try:
            if hasattr(task, "cancel"):
                task.cancel()
        except Exception:
            pass
        return False

    def mark_started(self, run_id: str) -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r or r.ended_at is not None:
                return
            r.started_at = time.time()
            transition(
                r,
                "state",
                "running",
                table=SUBAGENT_RUN_TRANSITIONS,
            )
        self._persist_to_disk()

    def mark_completed(
        self,
        run_id: str,
        result_summary: str = "",
        outcome: str = "completed",
        terminal_reason: str | None = None,
    ) -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r or r.ended_at is not None:
                return
            r.ended_at = time.time()
            r.outcome = outcome
            r.result_summary = result_summary[:1000]
            transition(r, "state", "succeeded", table=SUBAGENT_RUN_TRANSITIONS)
            r.terminal_reason = terminal_reason
            transition(r, "result_delivery_state", "pending", table=SUBAGENT_ANNOUNCE_TRANSITIONS)
        self._persist_to_disk()

    def mark_terminated(self, run_id: str, reason: str = "killed") -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r or r.ended_at is not None:
                return
            self._mark_terminated_record(r, reason)
        self._persist_to_disk()

    @staticmethod
    def _mark_terminated_record(
        record: SubagentRunRecord,
        reason: str,
    ) -> None:
        record.ended_at = time.time()
        record.outcome = reason
        lowered = (reason or "").lower()
        if "timeout" in lowered:
            new_state = "timed_out"
        elif "killed" in lowered or "cancel" in lowered:
            new_state = "cancelled"
        elif "orphaned" in lowered:
            new_state = "orphaned"
        elif "restart-interrupted" in lowered:
            new_state = "interrupted"
        else:
            new_state = "failed"
        transition(
            record,
            "state",
            new_state,
            table=SUBAGENT_RUN_TRANSITIONS,
        )
        record.terminal_reason = reason

    def kill(self, run_id: str, cascade: bool = True) -> bool:
        """终止 run，cascade=True 时递归终止其子 runs"""
        tasks_to_cancel: list[Any] = []
        with self._lock:
            root = self._runs.get(run_id)
            if root is None or root.ended_at is not None:
                return False

            pending = [run_id]
            visited: set[str] = set()
            while pending:
                current_id = pending.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)
                record = self._runs.get(current_id)
                if record is None or record.ended_at is not None:
                    continue

                if cascade:
                    child_sk = self.session_key_from_child_session_key(
                        record.child_session_key
                    )
                    pending.extend(
                        child.run_id
                        for child in self._runs.values()
                        if child.requester_session_key == child_sk
                        and child.ended_at is None
                    )

                self._mark_terminated_record(record, "killed")
                if record.asyncio_task is not None:
                    tasks_to_cancel.append(record.asyncio_task)

            self._persist_to_disk()

        for task in tasks_to_cancel:
            try:
                if hasattr(task, "cancel"):
                    task.cancel()
            except Exception:
                pass
        return True

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
        with self._lock:
            record = self._runs.get(run_id)
            return (
                self._snapshot_record(record)
                if record is not None
                else None
            )

    def list_runs(self) -> list[SubagentRunRecord]:
        """Return a snapshot of all run records."""
        with self._lock:
            return [
                self._snapshot_record(record)
                for record in self._runs.values()
            ]

    def list_run_entries(
        self,
    ) -> list[tuple[str, SubagentRunRecord]]:
        """Return canonical registry keys with their records."""
        with self._lock:
            return [
                (run_id, self._snapshot_record(record))
                for run_id, record in self._runs.items()
            ]

    def remove_run(self, run_id: str) -> bool:
        """Remove one run and persist the registry change."""
        with self._lock:
            removed = self._runs.pop(run_id, None)
        if removed is None:
            return False
        self._persist_to_disk()
        return True

    def mark_announce_retry(self, run_id: str) -> bool:
        """标记 announce 重试，返回是否可继续重试（未超限且未过期）"""
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                return False
            MAX_RETRY = 3
            EXPIRE_MS = 5 * 60 * 1000
            if r.announce_retry_count >= MAX_RETRY:
                return False
            if r.ended_at and (time.time() * 1000 - r.ended_at * 1000) > EXPIRE_MS:
                return False
            r.announce_retry_count = getattr(r, "announce_retry_count", 0) + 1
            r.last_announce_retry_at = time.time()
            transition(
                r,
                "result_delivery_state",
                "retrying",
                table=SUBAGENT_ANNOUNCE_TRANSITIONS,
            )
        self._persist_to_disk()
        return True

    def mark_result_delivery_delivered(self, run_id: str) -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                return
            transition(r, "result_delivery_state", "delivered", table=SUBAGENT_ANNOUNCE_TRANSITIONS)
        self._persist_to_disk()

    def mark_result_delivery_dropped(self, run_id: str) -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                return
            transition(r, "result_delivery_state", "dropped", table=SUBAGENT_ANNOUNCE_TRANSITIONS)
        self._persist_to_disk()

    def set_result_delivery_state(self, run_id: str, new_state: str) -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                return
            transition(r, "result_delivery_state", new_state, table=SUBAGENT_ANNOUNCE_TRANSITIONS)
        self._persist_to_disk()

    def set_delivery_work_id(self, run_id: str, work_id: str | None) -> None:
        with self._lock:
            r = self._runs.get(run_id)
            if not r:
                return
            r.delivery_work_id = (work_id or "").strip() or None
        self._persist_to_disk()

    def get_requester_depth(self, requester_session_key: str) -> int:
        """Depth 0 = main, 1 = subagent, 2 = sub-subagent

        requester 作为 child 被 spawn 时，创建它的 run.spawn_depth
        就是该 requester 的深度。
        主会话无对应 run，返回 0。
        """
        key = (requester_session_key or "").strip()
        if not key:
            return 0
        for r in self.list_runs():
            if r.child_session_key == key:
                return max(0, r.spawn_depth)
        return 0

    @staticmethod
    def session_key_from_child_session_key(child_session_key: str) -> str:
        """返回 child 会话可直接作为 requester 使用的 canonical key。"""
        return (child_session_key or "").strip()

    def list_descendant_runs(
        self, root_session_key: str, include_recent_minutes: int = 60
    ) -> list[SubagentRunRecord]:
        """从 root 起 BFS 收集所有后代 runs"""
        cutoff = time.time() - include_recent_minutes * 60
        root = (root_session_key or "").strip()
        if not root:
            return []
        pending = [root]
        visited: set[str] = {root}
        descendants: list[SubagentRunRecord] = []
        while pending:
            requester = pending.pop(0)
            for r in self.list_runs():
                if r.requester_session_key != requester:
                    continue
                if r.ended_at is not None and r.ended_at < cutoff:
                    continue
                descendants.append(r)
                child_sk = self.session_key_from_child_session_key(r.child_session_key)
                if child_sk and child_sk not in visited:
                    visited.add(child_sk)
                    pending.append(child_sk)
        return sorted(descendants, key=lambda x: x.created_at, reverse=True)

    def resolve_requester_for_child_session(
        self, child_session_key: str
    ) -> tuple[str, str] | None:
        """给定 child_session_key，返回 (requester_session_key, requester_agent_id)"""
        key = (child_session_key or "").strip()
        if not key:
            return None
        best: SubagentRunRecord | None = None
        for r in self.list_runs():
            if r.child_session_key != key:
                continue
            if best is None or r.created_at > best.created_at:
                best = r
        if best is None:
            return None
        return (best.requester_session_key, best.requester_agent_id)

    def count_active_descendant_runs(self, root_session_key: str) -> int:
        """root 下尚未结束的后代 run 数量"""
        root = (root_session_key or "").strip()
        if not root:
            return 0
        pending = [root]
        visited: set[str] = {root}
        count = 0
        while pending:
            requester = pending.pop(0)
            for r in self.list_runs():
                if r.requester_session_key != requester:
                    continue
                if r.ended_at is None:
                    count += 1
                child_sk = self.session_key_from_child_session_key(r.child_session_key)
                if child_sk and child_sk not in visited:
                    visited.add(child_sk)
                    pending.append(child_sk)
        return count

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        cutoff = time.time() - max_age_hours * 3600
        with self._lock:
            to_remove = [
                rid for rid, r in self._runs.items()
                if r.ended_at is not None and r.ended_at < cutoff
            ]
            for rid in to_remove:
                self._runs.pop(rid, None)
        if to_remove:
            self._persist_to_disk()
        return len(to_remove)

    def replace_active_run_for_steer(
        self,
        previous_run_id: str,
        next_run_id: str,
        task: str,
    ) -> SubagentRunRecord | None:
        """原子接管活跃 run，并在锁外取消旧任务。"""
        old_task: Any = None
        now = time.time()
        archive_after_ms = _resolve_archive_after_ms()
        archive_at_ms = (
            now * 1000 + archive_after_ms
            if archive_after_ms
            else None
        )
        with self._lock:
            previous = self._runs.get(previous_run_id)
            if previous is None or previous.ended_at is not None:
                return None
            record = SubagentRunRecord(
                run_id=next_run_id,
                child_session_key=previous.child_session_key,
                requester_session_key=(
                    previous.requester_session_key
                ),
                requester_agent_id=previous.requester_agent_id,
                target_agent_id=previous.target_agent_id,
                task=task,
                label=previous.label,
                model=previous.model,
                cleanup=previous.cleanup,
                spawn_depth=previous.spawn_depth,
                created_at=now,
                started_at=now,
                archive_at_ms=archive_at_ms,
            )
            old_task = previous.asyncio_task
            self._runs.pop(previous_run_id, None)
            self._runs[next_run_id] = record
            self._persist_to_disk()

        try:
            if old_task is not None and hasattr(old_task, "cancel"):
                old_task.cancel()
        except Exception:
            pass
        return self._snapshot_record(record)

    def sweep_expired(
        self,
        on_expire: Callable[[SubagentRunRecord], None] | None = None,
    ) -> int:
        """删除 archive_at_ms 已到期的 run，并归档会话。

        每 60 秒由 subagent_archive 调用。on_expire 负责归档/删除会话文件。
        """
        now_ms = time.time() * 1000
        with self._lock:
            to_remove: list[tuple[str, SubagentRunRecord]] = []
            for rid, r in self._runs.items():
                if r.archive_at_ms is None or r.archive_at_ms > now_ms:
                    continue
                if r.ended_at is None:
                    continue
                transition(r, "state", "archived", table=SUBAGENT_RUN_TRANSITIONS)
                r.ended_at = time.time()
                to_remove.append((rid, r))
            for rid, _ in to_remove:
                self._runs.pop(rid, None)

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
