"""Requester session scope and recent-window resolution for subagents."""

from __future__ import annotations

from typing import Any, Callable


class SubagentScopeResolver:
    MAX_RECENT_MINUTES = 24 * 60

    def __init__(
        self,
        *,
        session_manager: Any,
        get_config: Callable[[], dict[str, Any]],
    ) -> None:
        self._session_manager = session_manager
        self._get_config = get_config

    def requester_key(
        self,
        agent_id: str,
        session_id: str | None,
    ) -> str:
        effective_session_id = (
            (session_id or "").strip()
            or self._session_manager.resolve_main_session_id(agent_id)
        )
        return self._session_manager.session_key_from_session_id(
            agent_id,
            effective_session_id,
        )

    def recent_minutes(self, value: int | None) -> int:
        if value is not None and value > 0:
            return max(
                1,
                min(self.MAX_RECENT_MINUTES, int(value)),
            )
        config = self._get_config() or {}
        default = (
            config.get("agents", {})
            .get("defaults", {})
            .get("subagents", {})
            .get("recent_minutes")
        )
        if not isinstance(default, (int, float)):
            default = 30
        return max(
            1,
            min(self.MAX_RECENT_MINUTES, int(default)),
        )
