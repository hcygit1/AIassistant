"""Canonical session identifiers and keys."""

from __future__ import annotations


def resolve_main_session_id(agent_id: str) -> str:
    """Return the fixed main-session ID for one agent."""
    return f"{agent_id}-main"


def main_session_key(agent_id: str) -> str:
    """Return the canonical main-session key for one agent."""
    return f"agent:{agent_id}:main"


def session_key_from_session_id(agent_id: str, session_id: str) -> str:
    """Convert a session ID to its canonical session key."""
    if session_id == resolve_main_session_id(agent_id):
        return main_session_key(agent_id)
    return f"agent:{agent_id}:subagent:{session_id}"


def session_id_from_session_key(
    session_key: str,
) -> tuple[str, str] | None:
    """Parse a canonical or legacy session key into agent and session IDs."""
    parts = (session_key or "").strip().split(":")
    if len(parts) < 3:
        return None
    if parts[0].lower() != "agent":
        return None
    agent_id = parts[1]
    rest = ":".join(parts[2:])
    if rest == "main":
        return agent_id, resolve_main_session_id(agent_id)
    if len(parts) >= 4 and parts[2].lower() == "subagent":
        return agent_id, parts[3]
    return agent_id, rest
