"""Session loading, compatibility normalization, and agent-history projection."""

from __future__ import annotations

from typing import Any, Callable

from sessions.session_repository import SessionRepository


class SessionReader:
    def __init__(
        self,
        *,
        repository: SessionRepository,
        is_bootstrap_text: Callable[[str | None], bool],
    ) -> None:
        self._repository = repository
        self._is_bootstrap_text = is_bootstrap_text

    def load_session(self, session_id: str, agent_id: str) -> dict[str, Any] | None:
        data = self._repository.load_session(session_id, agent_id)
        if data is None:
            return None

        current_label = str(data.get("label", "")).strip()
        if current_label and self._is_bootstrap_text(current_label):
            data.pop("label", None)
        if not data.get("label") and data.get("title"):
            candidate = str(data.get("title", "")).strip()
            if candidate and not self._is_bootstrap_text(candidate):
                data["label"] = candidate

        return data

    def load_session_for_agent(
        self,
        session_id: str,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        data = self.load_session(session_id, agent_id)
        if data is None:
            return []

        messages: list[dict[str, Any]] = []
        for msg in data.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if (
                messages
                and messages[-1]["role"] == "assistant"
                and role == "assistant"
            ):
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": role, "content": content})
        return messages
