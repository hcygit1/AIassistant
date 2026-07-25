"""FastAPI dependency providers for runtime compatibility globals."""

from __future__ import annotations

from typing import Any


def get_agent_manager() -> Any:
    from runtime.agent import agent_manager

    return agent_manager


def get_heartbeat_runner() -> Any:
    from system_messages.heartbeat import heartbeat_runner

    return heartbeat_runner


def get_session_manager() -> Any:
    from sessions.session_manager import session_manager

    return session_manager


def get_user_turn_service() -> Any:
    from turns.service import user_turn_service

    return user_turn_service
