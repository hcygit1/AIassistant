"""统一事件总线 + 标准化事件 schema

所有 event_bus.emit 调用统一通过 Events 工厂方法构建事件，
确保字段完整、命名一致、便于前端消费和后端审计。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """简易事件总线，支持 SSE 订阅"""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, agent_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(agent_id, []).append(queue)
        return queue

    def unsubscribe(self, agent_id: str, queue: asyncio.Queue) -> None:
        queues = self._queues.get(agent_id, [])
        if queue in queues:
            queues.remove(queue)

    def emit(self, agent_id: str, event: dict[str, Any]) -> None:
        if "ts" not in event:
            event["ts"] = time.time()
        for queue in self._queues.get(agent_id, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


event_bus = EventBus()


# ---------------------------------------------------------------------------
# 标准化事件工厂
# ---------------------------------------------------------------------------

class Events:
    """事件工厂 — 所有事件类型的唯一入口，防止拼写错误和字段遗漏。"""

    # ── 对话轮次 ──

    @staticmethod
    def turn_start(*, run_id: str, model: str, **extra: Any) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "turn_start", "run_id": run_id, "model": model, **extra}

    @staticmethod
    def turn_end(*, run_id: str, **extra: Any) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "turn_end", "run_id": run_id, **extra}

    @staticmethod
    def turn_error(*, error: str, **extra: Any) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "turn_error", "error": error, **extra}

    @staticmethod
    def recursion_limit_reached(*, step: int, max_steps: int) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "recursion_limit_reached", "step": step, "max_steps": max_steps}

    # ── 工具安全 ──

    @staticmethod
    def tool_dangerous_executed(*, tool: str, input_preview: str = "") -> dict[str, Any]:
        return {"type": "lifecycle", "event": "tool_dangerous_executed", "tool": tool, "input_preview": input_preview}

    @staticmethod
    def approval_required(*, approval_id: str, tool: str, input_preview: str = "") -> dict[str, Any]:
        return {"type": "lifecycle", "event": "approval_required", "approval_id": approval_id, "tool": tool, "input_preview": input_preview}

    # ── 上下文压缩 ──

    @staticmethod
    def auto_compact_start(*, session_id: str, level: str) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "auto_compact_start", "session_id": session_id, "level": level}

    @staticmethod
    def auto_compact_done(*, session_id: str) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "auto_compact_done", "session_id": session_id}

    @staticmethod
    def manual_compact_start(*, session_id: str) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "manual_compact_start", "session_id": session_id}

    @staticmethod
    def manual_compact_done(*, session_id: str, **extra: Any) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "manual_compact_done", "session_id": session_id, **extra}

    @staticmethod
    def manual_compact_skipped(*, session_id: str, reason: str = "") -> dict[str, Any]:
        return {"type": "lifecycle", "event": "manual_compact_skipped", "session_id": session_id, "reason": reason}

    @staticmethod
    def manual_compact_error(*, session_id: str, error: str) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "manual_compact_error", "session_id": session_id, "error": error}

    # ── 子 Agent ──

    @staticmethod
    def subagent_start(*, run_id: str, agent_id: str, task: str, **extra: Any) -> dict[str, Any]:
        return {"type": "subagent_start", "run_id": run_id, "agent_id": agent_id, "task": task, **extra}

    @staticmethod
    def subagent_progress(*, run_id: str, chars: int, elapsed_s: int) -> dict[str, Any]:
        return {"type": "subagent_progress", "run_id": run_id, "chars": chars, "elapsed_s": elapsed_s}

    @staticmethod
    def subagent_tool(*, run_id: str, tool: str) -> dict[str, Any]:
        return {"type": "subagent_tool", "run_id": run_id, "tool": tool}

    @staticmethod
    def subagent_tool_end(*, run_id: str, tool: str, output_preview: str = "") -> dict[str, Any]:
        return {"type": "subagent_tool_end", "run_id": run_id, "tool": tool, "output_preview": output_preview}

    @staticmethod
    def subagent_done(*, run_id: str, result: str) -> dict[str, Any]:
        return {"type": "subagent_done", "run_id": run_id, "result": result}

    @staticmethod
    def subagent_error(*, run_id: str, error: str) -> dict[str, Any]:
        return {"type": "subagent_error", "run_id": run_id, "error": error}

    @staticmethod
    def subagent_killed(*, run_id: str) -> dict[str, Any]:
        return {"type": "subagent_killed", "run_id": run_id}

    @staticmethod
    def subagent_announce(*, run_id: str, announce_state: str) -> dict[str, Any]:
        return {"type": "subagent_announce", "run_id": run_id, "announce_state": announce_state}

    @staticmethod
    def subagent_archived(*, run_id: str, child_session_key: str) -> dict[str, Any]:
        return {"type": "lifecycle", "event": "subagent_archived", "run_id": run_id, "child_session_key": child_session_key}

    # ── 热更新 ──

    @staticmethod
    def skills_updated() -> dict[str, Any]:
        return {"type": "lifecycle", "event": "skills_updated"}

    # ── 心跳 ──

    @staticmethod
    def heartbeat_message(*, session_id: str, **extra: Any) -> dict[str, Any]:
        return {"type": "heartbeat_message", "session_id": session_id, **extra}
