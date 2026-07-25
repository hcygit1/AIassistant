"""Persistent record model for session work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionWorkRecord:
    id: str
    kind: str
    agent_id: str
    session_id: str
    content: str
    priority: int
    prompt_mode: str = "minimal"
    persist_role: str = "system"
    run_id: str | None = None
    status: str = "queued"
    recover_on_restart: bool = False
    created_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    last_error: str | None = None
