from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_manager import SessionManager
from sessions.session_persistence import SessionPersistenceService


class RecordingLock:
    def __init__(self, name: str, events: list[tuple[str, object]]) -> None:
        self._name = name
        self._events = events
        self._lock = threading.RLock()

    def __enter__(self):
        self._lock.acquire()
        self._events.append((f"{self._name}-enter", None))
        return self

    def __exit__(self, *_args) -> None:
        self._events.append((f"{self._name}-exit", None))
        self._lock.release()


class RecordingRepository:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.agent_lock = RecordingLock("agent", self.events)
        self.session_lock = RecordingLock("session", self.events)

    def get_agent_lock(self, _agent_id: str):
        return self.agent_lock

    def get_session_lock(self, _session_id: str, _agent_id: str):
        return self.session_lock

    def save_session(self, session_id: str, agent_id: str, data: dict) -> None:
        self.events.append(("save", (session_id, agent_id, data)))


class SessionPersistenceServiceTests(unittest.TestCase):
    def test_persists_transcript_then_updates_catalog_entry(self) -> None:
        repository = RecordingRepository()
        update_index = Mock(
            side_effect=lambda *args, **kwargs: repository.events.append(
                ("index", (args, kwargs))
            )
        )
        service = SessionPersistenceService(
            repository=repository,
            update_index_entry=update_index,
            session_key_from_id=lambda agent_id, session_id: (
                f"{agent_id}:{session_id}"
            ),
        )
        data = {
            "updated_at": 12.5,
            "label": "A session",
            "spawned_by": "agent:main:main",
        }

        service.persist_session("session-1", "agent-1", data)

        self.assertEqual(
            [name for name, _ in repository.events],
            [
                "agent-enter",
                "session-enter",
                "save",
                "index",
                "session-exit",
                "agent-exit",
            ],
        )
        self.assertEqual(
            repository.events[2][1],
            ("session-1", "agent-1", data),
        )
        update_index.assert_called_once_with(
            "agent-1",
            "agent-1:session-1",
            "session-1",
            12.5,
            label="A session",
            spawned_by="agent:main:main",
        )

    def test_manager_save_session_data_remains_compatibility_facade(self) -> None:
        manager = SessionManager(repository=Mock())
        manager._persistence.persist_session = Mock()

        manager._save_session_data("session-1", "agent-1", {"messages": []})

        manager._persistence.persist_session.assert_called_once_with(
            "session-1",
            "agent-1",
            {"messages": []},
        )

    def test_manager_uses_injected_persistence_service(self) -> None:
        persistence = Mock()

        manager = SessionManager(
            repository=Mock(),
            persistence=persistence,
        )

        self.assertIs(manager._persistence, persistence)
        manager._save_session_data("session-1", "agent-1", {"messages": []})
        persistence.persist_session.assert_called_once_with(
            "session-1",
            "agent-1",
            {"messages": []},
        )


if __name__ == "__main__":
    unittest.main()
