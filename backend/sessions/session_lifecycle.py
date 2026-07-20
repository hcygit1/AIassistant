"""Session creation, rename, deletion, and reset orchestration."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, ContextManager

from sessions.session_repository import SessionRepository

logger = logging.getLogger(__name__)


def _default_cleanup_runtime(agent_id: str, session_id: str) -> None:
    from sessions.session_lock_manager import cleanup_session_runtime

    cleanup_session_runtime(agent_id, session_id)


class SessionLifecycleService:
    """Coordinate session lifecycle operations across storage and runtime state."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        transaction: Callable[[str, str], ContextManager[Any]],
        load_session: Callable[[str, str], dict[str, Any] | None],
        save_session_data: Callable[[str, str, dict[str, Any]], None],
        remove_index: Callable[[str, str], None],
        session_key_from_id: Callable[[str, str], str],
        ensure_session: Callable[[str, str], dict[str, Any]],
        cleanup_runtime: Callable[[str, str], None] | None = None,
        now: Callable[[], float] | None = None,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._transaction = transaction
        self._load_session = load_session
        self._save_session_data = save_session_data
        self._remove_index = remove_index
        self._session_key_from_id = session_key_from_id
        self._ensure_session = ensure_session
        self._cleanup_runtime = (
            cleanup_runtime
            if cleanup_runtime is not None
            else _default_cleanup_runtime
        )
        self._now = now if now is not None else time.time
        self._session_id_factory = (
            session_id_factory
            if session_id_factory is not None
            else lambda: uuid.uuid4().hex[:12]
        )

    def create_session(self, agent_id: str, title: str = "新会话") -> str:
        session_id = self._session_id_factory()
        data = {
            "session_id": session_id,
            "agent_id": agent_id,
            "created_at": self._now(),
            "updated_at": self._now(),
            "messages": [],
        }
        if title and title.strip():
            data["label"] = title.strip()
        self._save_session_data(session_id, agent_id, data)
        return session_id

    def rename_session(
        self,
        session_id: str,
        agent_id: str,
        title: str,
    ) -> bool:
        with self._transaction(session_id, agent_id):
            data = self._load_session(session_id, agent_id)
            if data is None:
                return False
            data["label"] = title
            data["updated_at"] = self._now()
            self._save_session_data(session_id, agent_id, data)
            return True

    def delete_session(self, session_id: str, agent_id: str) -> bool:
        with self._repository.get_agent_lock(agent_id):
            if not self._repository.delete_session_file(session_id, agent_id):
                return False
            session_key = self._session_key_from_id(agent_id, session_id)
            self._remove_index(agent_id, session_key)
            try:
                self._cleanup_runtime(agent_id, session_id)
            except Exception as exc:
                logger.warning(
                    "cleanup_session_runtime after delete_session: %s",
                    exc,
                )
            return True

    def reset_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"archived": False}

        with self._repository.get_agent_lock(agent_id):
            path = self._repository.session_path(session_id, agent_id)
            if path.exists():
                archive_dir = self._repository.sessions_dir(agent_id) / "archive"
                try:
                    timestamp = int(self._now())
                    archive_name = f"{session_id}.reset.{timestamp}.json"
                    archive_path = archive_dir / archive_name
                    self._repository.archive_session_file(
                        session_id,
                        agent_id,
                        archive_path,
                    )
                    result["archived"] = True
                    result["archive_file"] = f"archive/{archive_name}"
                except OSError as exc:
                    logger.warning(
                        "Failed to archive session %s: %s",
                        session_id,
                        exc,
                    )
                    try:
                        self._repository.delete_session_file(
                            session_id,
                            agent_id,
                        )
                    except OSError:
                        pass

            self._repository.invalidate_session(session_id, agent_id)
            session_key = self._session_key_from_id(agent_id, session_id)
            self._remove_index(agent_id, session_key)
            self._ensure_session(session_id, agent_id)

        return result
