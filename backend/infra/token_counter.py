"""Token 计数器 — 用于自动压缩阈值检测"""

from __future__ import annotations

from typing import Any

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _encoding = "fallback"
    return _encoding


def count_tokens(text: str) -> int:
    enc = _get_encoding()
    if enc == "fallback":
        return len(text) // 3
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += count_tokens(content)
        total += 4  # role overhead
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            total += count_tokens(str(tc.get("input", "")))
            total += count_tokens(str(tc.get("output", "")))
    return total


CompactionLevel = str  # "none" | "sliding" | "forced"


def resolve_compaction_threshold(agent_id: str | None = None) -> int:
    """向后兼容：返回 sliding 阈值（用于 /context 命令等展示）。"""
    from graph.context_budget import resolve_budget
    return resolve_budget(agent_id).sliding_threshold


def resolve_compaction_thresholds(agent_id: str | None = None) -> tuple[int, int]:
    """返回 (sliding_threshold, forced_threshold)，值由 ContextBudget 统一派生。"""
    from graph.context_budget import resolve_budget
    budget = resolve_budget(agent_id)
    return (budget.sliding_threshold, budget.forced_threshold)


def should_compact(
    messages: list[dict[str, Any]],
    threshold: int | None = None,
    agent_id: str | None = None,
) -> bool:
    """向后兼容：返回 bool（是否超过 sliding 阈值）。"""
    if threshold is None:
        threshold = resolve_compaction_threshold(agent_id)
    total = count_messages_tokens(messages)
    return total >= threshold


def detect_compaction_level(
    messages: list[dict[str, Any]],
    agent_id: str | None = None,
    overhead_tokens: int = 0,
) -> CompactionLevel:
    """检测压缩级别：none / sliding / forced。
    overhead_tokens: system_prompt + session_summary 等非 messages 的固定开销。
    """
    sliding_threshold, forced_threshold = resolve_compaction_thresholds(agent_id)
    total = count_messages_tokens(messages) + overhead_tokens
    if total >= forced_threshold:
        return "forced"
    if total >= sliding_threshold:
        return "sliding"
    return "none"
