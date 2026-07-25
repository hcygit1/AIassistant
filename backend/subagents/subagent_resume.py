"""子 Agent 恢复

启动时对从 runs.json 恢复的 run 执行：
1. 孤儿检测：child 会话不存在则 reconcile
2. 已结束：执行 announce 交付
3. 未结束（单进程重启后 task 已丢失）：视为孤儿
"""

from __future__ import annotations

import logging
import time

from subagents.subagent_registry import registry
from subagents.subagent_run_model import SubagentRunRecord
from subagents.subagent_run_resume import (
    ANNOUNCE_EXPIRY_SEC,
    ChildSessionValidation,
    MAX_ANNOUNCE_RETRY,
    SubagentRunResumeService,
    VALIDATION_UNAVAILABLE,
    resolve_orphan_reason,
)

logger = logging.getLogger(__name__)


def _resolve_orphan_reason(entry: SubagentRunRecord) -> ChildSessionValidation:
    """判断子 Agent run 是否变成孤儿"""
    try:
        from sessions.session_manager import session_manager
    except Exception as exc:
        logger.warning(
            "Subagent child session validation unavailable run=%s: %s",
            entry.run_id,
            exc,
        )
        return VALIDATION_UNAVAILABLE
    return resolve_orphan_reason(entry, session_manager)


async def _deliver_announce_for_run(run_id: str, entry: SubagentRunRecord) -> bool:
    """向 requester 交付 announce"""
    from subagents.subagent_delivery import subagent_announce_delivery

    return await subagent_announce_delivery.deliver_recovered_run(run_id, entry)


async def resume_subagent_runs() -> None:
    """启动时调用：对恢复的 run 执行 reconcile + announce"""
    service = SubagentRunResumeService(
        registry=registry,
        resolve_orphan_reason=_resolve_orphan_reason,
        deliver_announce=_deliver_announce_for_run,
        now=time.time,
        max_announce_retry=MAX_ANNOUNCE_RETRY,
        announce_expiry_sec=ANNOUNCE_EXPIRY_SEC,
    )
    await service.resume()
