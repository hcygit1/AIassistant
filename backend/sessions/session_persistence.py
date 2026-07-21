"""Combined persistence boundary for session transcripts and catalog entries."""

from __future__ import annotations

import time
from typing import Any, Callable

from sessions.session_repository import SessionRepository


class SessionPersistenceService:
    """Persist one transcript and its catalog projection under shared locks."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        update_index_entry: Callable[..., None],
        session_key_from_id: Callable[[str, str], str],
    ) -> None:
        self._repository = repository
        self._update_index_entry = update_index_entry
        self._session_key_from_id = session_key_from_id

    def persist_session(
        self,
        session_id: str,
        agent_id: str,
        data: dict[str, Any],
    ) -> None:
        with self._repository.get_agent_lock(agent_id):
            with self._repository.get_session_lock(session_id, agent_id):
                self._repository.save_session(session_id, agent_id, data)
                session_key = self._session_key_from_id(agent_id, session_id)
                self._update_index_entry(
                    agent_id,
                    session_key,
                    session_id,
                    data.get("updated_at", time.time()),
                    label=data.get("label", ""),
                    spawned_by=data.get("spawned_by"),
                )
