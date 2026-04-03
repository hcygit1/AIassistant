"""显式状态机 — 定义合法的状态转换路径

所有实体的状态变更都必须通过 transition() 函数，
非法转换直接抛 InvalidTransitionError，防止隐式状态污染。
"""

from __future__ import annotations

from typing import Any


class InvalidTransitionError(Exception):
    pass


# ---------------------------------------------------------------------------
# 子 Agent 运行状态
# ---------------------------------------------------------------------------

SUBAGENT_RUN_TRANSITIONS: dict[str, set[str]] = {
    "running":     {"running", "succeeded", "failed", "timed_out", "cancelled", "orphaned", "interrupted"},
    "succeeded":   {"archived"},
    "failed":      {"archived"},
    "timed_out":   {"archived"},
    "cancelled":   {"archived"},
    "orphaned":    {"archived"},
    "interrupted": {"archived"},
    "archived":    set(),
}

# ---------------------------------------------------------------------------
# 子 Agent 结果投递状态
# ---------------------------------------------------------------------------

SUBAGENT_ANNOUNCE_TRANSITIONS: dict[str, set[str]] = {
    "pending":    {"pending", "queued", "dropped", "retrying"},
    "queued":     {"pending", "delivering", "dropped"},
    "delivering": {"pending", "delivered", "dropped"},
    "retrying":   {"pending", "delivering", "dropped"},
    "delivered":  {"pending"},
    "dropped":    {"pending"},
}


# ---------------------------------------------------------------------------
# 通用转换函数
# ---------------------------------------------------------------------------

def transition(
    entity: Any,
    field: str,
    new_state: str,
    *,
    table: dict[str, set[str]],
) -> None:
    """验证并执行状态转换，非法转换抛出 InvalidTransitionError。"""
    current = getattr(entity, field)
    allowed = table.get(current)
    if allowed is None:
        raise InvalidTransitionError(
            f"{type(entity).__name__}.{field}: unknown state '{current}'"
        )
    if new_state not in allowed:
        raise InvalidTransitionError(
            f"{type(entity).__name__}.{field}: "
            f"'{current}' → '{new_state}' not allowed "
            f"(valid: {sorted(allowed) if allowed else 'none'})"
        )
    setattr(entity, field, new_state)
