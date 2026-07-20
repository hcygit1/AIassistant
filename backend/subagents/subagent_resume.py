"""子 Agent 恢复

启动时对从 runs.json 恢复的 run 执行：
1. 孤儿检测：child 会话不存在则 reconcile
2. 已结束：执行 announce 交付
3. 未结束（单进程重启后 task 已丢失）：视为孤儿
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from subagents.subagent_registry import registry
from subagents.subagent_run_model import SubagentRunRecord

logger = logging.getLogger(__name__)

MAX_ANNOUNCE_RETRY = 3
ANNOUNCE_EXPIRY_SEC = 5 * 60  # 5 分钟
_INTERRUPTED_DELIVERY_STATES = {"queued", "delivering", "retrying"}
_TERMINAL_DELIVERY_STATES = {"delivered", "dropped"}


def _resolve_orphan_reason(entry: SubagentRunRecord) -> Literal["missing-session-entry", "missing-session-id", "missing-session-file"] | None:
    """判断子 Agent run 是否变成孤儿"""
    child_key = (entry.child_session_key or "").strip()
    if not child_key:
        return "missing-session-entry"
    try:
        from sessions.session_manager import session_manager
        parts = child_key.split(":")
        if len(parts) < 4:
            return "missing-session-entry"
        agent_id = parts[1]
        session_id = parts[3]
        ent = session_manager.get_session_index_entry(
            session_id,
            agent_id,
        )
        if ent is None:
            return "missing-session-entry"
        sid = ent.get("sessionId")
        if not sid or not str(sid).strip():
            return "missing-session-id"
        if not session_manager.session_file_exists(
            str(sid).strip(),
            agent_id,
        ):
            return "missing-session-file"
        return None
    except Exception:
        return None


def _reconcile_orphaned(run_id: str, entry: SubagentRunRecord, reason: str) -> bool:
    """将孤儿 run 标记为已结束并清理"""
    registry.remove_run(run_id)
    logger.warning(f"Subagent orphan pruned run={run_id} child={entry.child_session_key} reason={reason}")
    return True


async def _deliver_announce_for_run(run_id: str, entry: SubagentRunRecord) -> bool:
    """向 requester 交付 announce"""
    from subagents.subagent_delivery import subagent_announce_delivery

    return await subagent_announce_delivery.deliver_recovered_run(run_id, entry)


def _reset_interrupted_delivery(run_id: str, entry: SubagentRunRecord) -> None:
    state = getattr(entry, "result_delivery_state", "pending")
    if state in _INTERRUPTED_DELIVERY_STATES:
        registry.set_result_delivery_state(run_id, "pending")


async def resume_subagent_runs() -> None:
    """启动时调用：对恢复的 run 执行 reconcile + announce"""
    for run_id, entry in registry.list_run_entries():

        reason = _resolve_orphan_reason(entry)
        if reason:
            _reconcile_orphaned(run_id, entry, reason)
            continue

        delivery_state = getattr(
            entry,
            "result_delivery_state",
            "pending",
        )
        if delivery_state in _TERMINAL_DELIVERY_STATES:
            registry.remove_run(run_id)
            continue

        if entry.announce_retry_count >= MAX_ANNOUNCE_RETRY:
            registry.remove_run(run_id)
            continue

        if entry.ended_at and (__import__("time").time() - entry.ended_at) > ANNOUNCE_EXPIRY_SEC:
            registry.remove_run(run_id)
            continue

        if entry.ended_at:
            _reset_interrupted_delivery(run_id, entry)
            try:
                queued = await _deliver_announce_for_run(run_id, entry)
            except Exception as e:
                logger.warning(f"Resume announce error run={run_id}: {e}")
                queued = False
            if not queued and not registry.mark_announce_retry(run_id):
                registry.mark_result_delivery_dropped(run_id)
                registry.remove_run(run_id)
            continue

        registry.mark_terminated(run_id, "restart-interrupted")
        interrupted = registry.get_run(run_id)
        if interrupted is None:
            continue
        _reset_interrupted_delivery(run_id, interrupted)
        try:
            queued = await _deliver_announce_for_run(
                run_id,
                interrupted,
            )
        except Exception:
            queued = False
        if not queued and not registry.mark_announce_retry(run_id):
            registry.mark_result_delivery_dropped(run_id)
            registry.remove_run(run_id)
