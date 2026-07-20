"""用户回合事件生成 — 供 SessionDispatcher 与 turn service 复用。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from turns.events import TurnEvent


def _should_skip_auto_title(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return True
    if text.startswith("/"):
        return True
    return text.startswith("a new session was started via /new or /reset")


@dataclass(frozen=True)
class UserTurnStreamDependencies:
    """Runtime collaborators for one user-turn event stream."""

    agent_manager: Any
    session_manager: Any
    coordinator: Any

    @classmethod
    def from_defaults(cls) -> "UserTurnStreamDependencies":
        from runtime.agent import agent_manager
        from sessions.session_manager import session_manager
        from turns.coordinator import user_turn_coordinator

        return cls(
            agent_manager=agent_manager,
            session_manager=session_manager,
            coordinator=user_turn_coordinator,
        )


async def iter_user_turn_events(
    message: str,
    session_id: str,
    agent_id: str,
    turn_id: str,
    *,
    dependencies: UserTurnStreamDependencies | None = None,
) -> AsyncIterator[TurnEvent]:
    resolved = dependencies or UserTurnStreamDependencies.from_defaults()
    agent_manager = resolved.agent_manager
    session_manager = resolved.session_manager
    user_turn_coordinator = resolved.coordinator

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
            yield TurnEvent.from_payload(event)
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
        yield TurnEvent.from_payload(
            {
                "type": "aborted",
                "session_id": session_id,
                "run_id": run_id,
                "content": partial_text,
                "reason": cancel_reason,
            }
        )
        return

    except Exception as e:
        yield TurnEvent.error(str(e))
        return

    if is_first_message and not _should_skip_auto_title(message):
        title = await _generate_title(
            message,
            agent_id,
            agent_manager=agent_manager,
        )
        if title:
            session_manager.rename_session(session_id, agent_id, title)
            yield TurnEvent.from_payload(
                {
                    "type": "title",
                    "session_id": session_id,
                    "title": title,
                }
            )


async def _generate_title(
    message: str,
    agent_id: str,
    *,
    agent_manager: Any | None = None,
) -> str | None:
    if agent_manager is None:
        from runtime.agent import agent_manager as default_agent_manager

        agent_manager = default_agent_manager

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
