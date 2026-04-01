"""Approval Store — 危险工具执行前用户确认"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)
Decision = Literal["approved", "denied", "timeout"]


@dataclass
class PendingApproval:
    approval_id: str
    agent_id: str
    tool: str
    input_preview: str
    created_at: float = field(default_factory=time.time)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Decision | None = None


class ApprovalStore:
    """内存态 approval 存储，支持 create/wait/resolve"""

    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}
        self._lock = asyncio.Lock()

    def _get_timeout_from_config(self) -> float:
        """从配置获取审批请求过期时间"""
        try:
            from config import get_config
            cfg = get_config()
            return cfg.get("tools", {}).get("exec", {}).get("approval", {}).get("pending_timeout_seconds", 300)
        except Exception:
            return 300

    def create(self, agent_id: str, tool: str, input_preview: str) -> str:
        """创建待确认请求，返回 approval_id"""
        # 清理已过期请求
        self._cleanup_expired()

        approval_id = str(uuid.uuid4())
        pending = PendingApproval(
            approval_id=approval_id,
            agent_id=agent_id,
            tool=tool,
            input_preview=input_preview[:500] if input_preview else "",
        )
        self._pending[approval_id] = pending
        logger.info(f"Approval created: {approval_id} for {tool}")
        return approval_id

    def _cleanup_expired(self) -> None:
        """清理已过期的审批请求"""
        timeout = self._get_timeout_from_config()
        now = time.time()
        expired = [
            aid for aid, p in self._pending.items()
            if now - p.created_at > timeout
        ]
        for aid in expired:
            logger.warning(f"Cleaning up expired approval: {aid}")
            self._pending.pop(aid, None)

    def is_expired(self, approval_id: str) -> bool:
        """检查审批请求是否已过期"""
        pending = self._pending.get(approval_id)
        if not pending:
            return True
        timeout = self._get_timeout_from_config()
        return time.time() - pending.created_at > timeout

    async def wait(
        self, approval_id: str, timeout_seconds: float = 60
    ) -> Decision:
        """等待用户确认，超时视为 denied"""
        pending = self._pending.get(approval_id)
        if not pending:
            return "denied"

        try:
            await asyncio.wait_for(pending.event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            pending.result = "timeout"
            logger.warning(f"Approval timeout: {approval_id}")
        finally:
            async with self._lock:
                self._pending.pop(approval_id, None)

        return pending.result or "timeout"

    def resolve(self, approval_id: str, decision: Literal["approved", "denied"]) -> bool:
        """用户确认/拒绝，唤醒等待的协程"""
        pending = self._pending.get(approval_id)
        if not pending:
            logger.warning(f"Attempted to resolve non-existent approval: {approval_id}")
            return False
        pending.result = decision
        pending.event.set()
        logger.info(f"Approval resolved: {approval_id} -> {decision}")
        return True


approval_store = ApprovalStore()
