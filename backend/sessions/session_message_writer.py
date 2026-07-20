"""Session transcript mutation boundary."""

from __future__ import annotations

import time
from typing import Any, Callable, ContextManager


class SessionMessageWriter:
    """Create and mutate sessions through one persistence boundary."""

    def __init__(
        self,
        *,
        transaction: Callable[[str, str], ContextManager[Any]],
        load_session: Callable[[str, str], dict[str, Any] | None],
        persist_session: Callable[[str, str, dict[str, Any]], None],
        session_key_from_id: Callable[[str, str], str],
        resolve_requester: Callable[[str], tuple[str, str] | None],
        now: Callable[[], float] | None = None,
    ) -> None:
        self._transaction = transaction
        self._load_session = load_session
        self._persist_session = persist_session
        self._session_key_from_id = session_key_from_id
        self._resolve_requester = resolve_requester
        self._now = now or time.time

    def ensure_session(
        self,
        session_id: str,
        agent_id: str,
        spawned_by: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        with self._transaction(session_id, agent_id):
            return self._ensure_session_locked(
                session_id,
                agent_id,
                spawned_by=spawned_by,
                label=label,
            )

    def _ensure_session_locked(
        self,
        session_id: str,
        agent_id: str,
        spawned_by: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        data = self._load_session(session_id, agent_id)
        if data is not None:
            return data
        if not spawned_by and session_id.startswith("subagent-"):
            child_key = self._session_key_from_id(agent_id, session_id)
            resolved = self._resolve_requester(child_key)
            if resolved:
                spawned_by = resolved[0]
        data = {
            "session_id": session_id,
            "agent_id": agent_id,
            "created_at": self._now(),
            "updated_at": self._now(),
            "messages": [],
        }
        if label and str(label).strip():
            data["label"] = str(label).strip()[:120]
        if spawned_by:
            data["spawned_by"] = spawned_by
        self._persist_session(session_id, agent_id, data)
        return data

    def rollback_last_turn(self, session_id: str, agent_id: str) -> bool:
        with self._transaction(session_id, agent_id):
            return self._rollback_last_turn_locked(session_id, agent_id)

    def _rollback_last_turn_locked(
        self,
        session_id: str,
        agent_id: str,
    ) -> bool:
        data = self._load_session(session_id, agent_id)
        if not data or not data.get("messages"):
            return False
        messages = data["messages"]
        if len(messages) < 2:
            return False
        if (
            messages[-1].get("role") != "assistant"
            or messages[-2].get("role") != "user"
        ):
            return False
        data["messages"] = messages[:-2]
        data["updated_at"] = self._now()
        self._persist_session(session_id, agent_id, data)
        return True

    def save_message(
        self,
        session_id: str,
        agent_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._transaction(session_id, agent_id):
            data = self._ensure_session_locked(session_id, agent_id)
            message: dict[str, Any] = {"role": role, "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            data["messages"].append(message)
            data["updated_at"] = self._now()
            self._persist_session(session_id, agent_id, data)
