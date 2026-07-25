"""FastAPI dependency providers for runtime compatibility globals."""

from __future__ import annotations

from typing import Any


def get_session_manager() -> Any:
    from sessions.session_manager import session_manager

    return session_manager


def get_user_turn_service() -> Any:
    from turns.service import user_turn_service

    return user_turn_service
