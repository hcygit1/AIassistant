"""用户回合 SSE 行生成 — 供 SessionDispatcher 与 chat 路由复用"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

def _should_skip_auto_title(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return True
    if text.startswith("/"):
        return True
    return text.startswith("a new session was started via /new or /reset")


async def iter_user_turn_sse(
    message: str,
    session_id: str,
    agent_id: str,
    turn_id: str,
) -> AsyncIterator[str]:
    from runtime.agent import agent_manager
    from sessions.session_manager import session_manager
    from turns.coordinator import user_turn_coordinator

    session_data = session_manager.load_session(session_id, agent_id)
    is_first_message = session_data is None or len(session_data.get("messages", [])) == 0
    partial_text = ""
    run_id = ""

    try:
        async for event in agent_manager.astream(
            message=message,
            session_id=session_id,
            agent_id=agent_id,
        ):
            if event.get("type") == "token":
                partial_text += event.get("content", "") or ""
            elif event.get("type") == "clear_content":
                partial_text = ""
            elif event.get("type") == "content_refresh":
                refreshed = event.get("content")
                if isinstance(refreshed, str):
                    partial_text = refreshed
            elif event.get("type") == "lifecycle" and event.get("event") == "turn_start":
                run_id = str(event.get("run_id") or run_id)
            event_type = event.get("type", "")
            data = json.dumps(event, ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data}\n\n"
    except asyncio.CancelledError:
        cancel_reason = user_turn_coordinator.get_cancel_reason(turn_id) or "client_disconnected"
        was_user_initiated = cancel_reason == "stopped_by_user"
        if was_user_initiated:
            try:
                data = session_manager.load_session(session_id, agent_id) or {}
                messages = data.get("messages", []) if isinstance(data, dict) else []
                has_user = bool(
                    messages
                    and messages[-1].get("role") == "user"
                    and messages[-1].get("content") == message
                )
                if not has_user:
                    session_manager.save_message(session_id, agent_id, "user", message)
                if (partial_text or "").strip():
                    session_manager.save_message(
                        session_id,
                        agent_id,
                        "assistant",
                        partial_text,
                    )
            except Exception:
                pass
        aborted_data = json.dumps({
            "type": "aborted",
            "session_id": session_id,
            "run_id": run_id,
            "content": partial_text,
            "reason": cancel_reason,
        }, ensure_ascii=False)
        yield f"event: aborted\ndata: {aborted_data}\n\n"
        return

    except Exception as e:
        error_data = json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False)
        yield f"event: error\ndata: {error_data}\n\n"
        return

    if is_first_message and not _should_skip_auto_title(message):
        title = await _generate_title(message, agent_id)
        if title:
            session_manager.rename_session(session_id, agent_id, title)
            title_data = json.dumps(
                {"type": "title", "session_id": session_id, "title": title},
                ensure_ascii=False,
            )
            yield f"event: title\ndata: {title_data}\n\n"


async def _generate_title(message: str, agent_id: str) -> str | None:
    from runtime.agent import agent_manager

    try:
        llm = agent_manager.get_llm(agent_id)
    except Exception:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from config import get_config
        from runtime.command_parser import t

        locale = get_config().get("app", {}).get("locale", "zh-CN")

        resp = await llm.ainvoke([
            SystemMessage(content=t("title_gen_system", locale)),
            HumanMessage(content=message),
        ])
        title = resp.content.strip()[:20]
        return title if title else None
    except Exception:
        return None
