"""FastAPI dependency providers for runtime compatibility globals."""

from __future__ import annotations

from typing import Any


def get_session_manager() -> Any:
    from sessions.session_manager import session_manager

    return session_manager
