"""工具循环检测"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def _get_loop_config() -> dict[str, int]:
    """从配置获取循环检测阈值"""
    try:
        from config import get_config
        cfg = get_config()
        sandbox_cfg = cfg.get("sandbox", {})
        loop_cfg = sandbox_cfg.get("loopDetection", {})
        return {
            "warningThreshold": loop_cfg.get("warningThreshold", 10),
            "criticalThreshold": loop_cfg.get("criticalThreshold", 20),
            "circuitBreaker": loop_cfg.get("circuitBreaker", 30),
            "historySize": loop_cfg.get("historySize", 30),
        }
    except Exception:
        return {
            "warningThreshold": 10,
            "criticalThreshold": 20,
            "circuitBreaker": 30,
            "historySize": 30,
        }


@dataclass
class ToolCall:
    tool_name: str
    args_hash: str
    result_hash: str | None = None


@dataclass
class LoopDetector:
    """每个会话维护一个实例"""
    history: deque[ToolCall] = field(default_factory=lambda: deque(maxlen=_get_loop_config()["historySize"]))
    total_calls: int = 0

    @staticmethod
    def _hash(obj: Any) -> str:
        raw = str(obj).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]

    def record(self, tool_name: str, args: Any, result: Any = None) -> str | None:
        """
        记录一次工具调用，返回警告消息（如有）。
        """
        config = _get_loop_config()
        warning_threshold = config["warningThreshold"]
        critical_threshold = config["criticalThreshold"]
        circuit_breaker = config["circuitBreaker"]

        # 超过双倍阈值，强制重置
        if self.total_calls >= circuit_breaker * 2:
            self.reset()
            return "[安全警告] 循环检测器已重置"

        args_hash = self._hash(args)
        result_hash = self._hash(result) if result is not None else None

        call = ToolCall(
            tool_name=tool_name,
            args_hash=args_hash,
            result_hash=result_hash,
        )
        self.history.append(call)
        self.total_calls += 1

        if self.total_calls >= circuit_breaker:
            return (
                f"[安全警告] 工具调用已达 {self.total_calls} 次，触发全局熔断。"
                "请停止当前循环并换一种方法。"
            )

        repeat_count = sum(
            1 for c in self.history
            if c.tool_name == tool_name and c.args_hash == args_hash
        )

        if repeat_count >= critical_threshold:
            return (
                f"[严重警告] 工具 '{tool_name}' 使用相同参数已被调用 {repeat_count} 次。"
                "请立即停止重复调用。"
            )
        if repeat_count >= warning_threshold:
            return (
                f"[警告] 工具 '{tool_name}' 使用相同参数已被调用 {repeat_count} 次，"
                "可能陷入循环。请检查你的方法。"
            )

        if len(self.history) >= 4:
            recent = list(self.history)[-4:]
            if (
                recent[0].tool_name == recent[2].tool_name
                and recent[1].tool_name == recent[3].tool_name
                and recent[0].tool_name != recent[1].tool_name
            ):
                return (
                    f"[警告] 检测到乒乓循环: "
                    f"'{recent[0].tool_name}' ↔ '{recent[1].tool_name}'。"
                    "请换一种方法。"
                )

        return None

    def reset(self) -> None:
        self.history.clear()
        self.total_calls = 0
