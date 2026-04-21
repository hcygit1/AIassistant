from __future__ import annotations

from collections.abc import Awaitable, Callable

from config import get_exec_approval_config
from infra.approval_store import approval_store
from infra.event_bus import Events, event_bus
from runtime.security_context import get_runtime_security_context
from runtime.source_sink_guard import evaluate_source_to_sink


async def run_tool_with_approval_gate(
    *,
    agent_id: str,
    tool_name: str,
    input_preview: str,
    locale: str,
    base_needs_approval: bool,
    deny_reason: str | None,
    execute_fn: Callable[[], str | Awaitable[str]],
) -> str:
    if deny_reason:
        return f"命令被拒绝: {deny_reason}" if locale == "zh-CN" else f"Command rejected: {deny_reason}"

    security_ctx = get_runtime_security_context()
    decision = evaluate_source_to_sink(
        has_recent_untrusted_content=security_ctx.recent_untrusted_content,
        tool_name=tool_name,
        tool_input={"input_preview": input_preview},
        user_message=security_ctx.user_message,
    )

    if decision.action == "block":
        reason = decision.reason or f"Blocked high-risk tool '{tool_name}'"
        return f"命令被拒绝: {reason}" if locale == "zh-CN" else f"Command rejected: {reason}"

    needs_approval = base_needs_approval or decision.action == "confirm"
    if needs_approval:
        cfg = get_exec_approval_config()
        timeout_sec = cfg.get("ask_timeout_seconds", 60)
        approval_id = approval_store.create(agent_id, tool_name, input_preview)
        event_bus.emit(
            agent_id,
            Events.approval_required(
                approval_id=approval_id,
                tool=tool_name,
                input_preview=input_preview,
            ),
        )
        decision_result = await approval_store.wait(approval_id, timeout_sec)
        if decision_result != "approved":
            if locale == "zh-CN":
                reason = "用户拒绝" if decision_result == "denied" else "确认超时，已自动拒绝"
                return f"命令被拒绝: {reason}"
            reason = "User denied" if decision_result == "denied" else "Confirmation timed out, automatically rejected"
            return f"Command rejected: {reason}"

    result = execute_fn()
    if hasattr(result, "__await__"):
        return await result  # type: ignore[return-value]
    return result  # type: ignore[return-value]
