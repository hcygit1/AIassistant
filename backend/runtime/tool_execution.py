"""统一的运行时工具异步执行入口。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from runtime.security_context import runtime_security_context

APPROVAL_SENSITIVE_TOOLS = frozenset(
    {
        "apply_patch",
        "edit",
        "exec",
        "process_kill",
        "write",
    }
)


def _execution_target(tool: Any) -> Any:
    return getattr(tool, "wrapped_tool", tool)


def _normalized_tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "")).replace("-", "_").strip().lower()


def ensure_sync_execution_allowed(tool: Any) -> None:
    """审批敏感工具不得通过同步公开接口执行。"""
    target = _execution_target(tool)
    if _normalized_tool_name(target) in APPROVAL_SENSITIVE_TOOLS:
        raise RuntimeError(
            f"Tool '{_normalized_tool_name(target)}' requires async approval execution"
        )


def ensure_async_execution_safe(tool: Any) -> None:
    """审批敏感工具必须提供自定义异步审批路径。"""
    target = _execution_target(tool)
    if _normalized_tool_name(target) not in APPROVAL_SENSITIVE_TOOLS:
        return
    if not isinstance(target, BaseTool):
        if callable(getattr(target, "ainvoke", None)):
            return
        raise RuntimeError(
            f"Tool '{_normalized_tool_name(target)}' requires async approval execution"
        )

    uses_default_ainvoke = target.__class__.ainvoke is BaseTool.ainvoke
    uses_default_arun = target.__class__._arun is BaseTool._arun
    if uses_default_ainvoke and uses_default_arun:
        raise RuntimeError(
            f"Tool '{_normalized_tool_name(target)}' requires async approval execution"
        )


async def invoke_tool_async(
    tool: Any,
    tool_input: dict[str, Any],
    *,
    user_message: str,
    recent_untrusted_content: bool,
) -> str:
    """通过工具公开异步接口执行，并显式绑定当前安全上下文。"""
    ensure_async_execution_safe(tool)
    with runtime_security_context(
        user_message,
        recent_untrusted_content=recent_untrusted_content,
    ):
        result = await tool.ainvoke(tool_input)

    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    return str(content if content is not None else result)
