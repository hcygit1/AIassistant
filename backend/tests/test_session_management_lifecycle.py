from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_lifecycle import SessionLifecycleService
from sessions.session_repository import SessionRepository


class SessionLifecycleServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.root = Path(tempdir.name)
        self.repository = SessionRepository(
            resolve_sessions_dir=lambda agent_id: self.root,
        )
        self.index: dict[str, dict] = {}
        self.cleaned: list[tuple[str, str]] = []

        @contextmanager
        def transaction(session_id: str, agent_id: str):
            with self.repository.get_agent_lock(agent_id):
                with self.repository.get_session_lock(session_id, agent_id):
                    yield

        def save_session_data(
            session_id: str,
            agent_id: str,
            data: dict,
        ) -> None:
            self.repository.save_session(session_id, agent_id, data)
            self.index[self._session_key(agent_id, session_id)] = {
                "sessionId": session_id,
            }

        def ensure_session(session_id: str, agent_id: str) -> dict:
            data = self.repository.load_session(session_id, agent_id)
            if data is None:
                data = {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "created_at": 100.0,
                    "updated_at": 100.0,
                    "messages": [],
                }
                save_session_data(session_id, agent_id, data)
            return data

        self.service = SessionLifecycleService(
            repository=self.repository,
            transaction=transaction,
            load_session=self.repository.load_session,
            save_session_data=save_session_data,
            remove_index=lambda agent_id, key: self.index.pop(key, None),
            session_key_from_id=self._session_key,
            ensure_session=ensure_session,
            cleanup_runtime=lambda agent_id, session_id: self.cleaned.append(
                (agent_id, session_id)
            ),
            now=lambda: 42.0,
            session_id_factory=lambda: "session-created",
        )

    @staticmethod
    def _session_key(agent_id: str, session_id: str) -> str:
        return f"agent:{agent_id}:subagent:{session_id}"

    def test_create_and_rename_session_update_transcript_and_index(self) -> None:
        session_id = self.service.create_session("agent-1", "  Initial  ")

        self.assertEqual(session_id, "session-created")
        created = self.repository.load_session(session_id, "agent-1")
        self.assertEqual(created["label"], "Initial")
        self.assertIn(self._session_key("agent-1", session_id), self.index)

        renamed = self.service.rename_session(
            session_id,
            "agent-1",
            "Renamed",
        )

        self.assertTrue(renamed)
        current = self.repository.load_session(session_id, "agent-1")
        self.assertEqual(current["label"], "Renamed")
        self.assertEqual(current["updated_at"], 42.0)

    def test_rename_missing_session_returns_false(self) -> None:
        self.assertFalse(
            self.service.rename_session("missing", "agent-1", "Renamed")
        )

    def test_delete_session_removes_index_and_runtime_state(self) -> None:
        session_id = self.service.create_session("agent-1")

        deleted = self.service.delete_session(session_id, "agent-1")

        self.assertTrue(deleted)
        self.assertFalse(
            self.repository.session_file_exists(session_id, "agent-1")
        )
        self.assertNotIn(self._session_key("agent-1", session_id), self.index)
        self.assertEqual(self.cleaned, [("agent-1", session_id)])

    def test_delete_missing_session_does_not_cleanup_runtime(self) -> None:
        self.assertFalse(self.service.delete_session("missing", "agent-1"))
        self.assertEqual(self.cleaned, [])

    def test_reset_archives_existing_session_and_recreates_empty_session(self) -> None:
        session_id = self.service.create_session("agent-1", "Before reset")

        result = self.service.reset_session(session_id, "agent-1")

        self.assertEqual(
            result,
            {
                "archived": True,
                "archive_file": "archive/session-created.reset.42.json",
            },
        )
        archive_path = self.root / result["archive_file"]
        self.assertTrue(archive_path.is_file())
        current = self.repository.load_session(session_id, "agent-1")
        self.assertEqual(current["messages"], [])
        self.assertNotIn("label", current)

    def test_reset_recreates_session_when_archive_fails(self) -> None:
        session_id = self.service.create_session("agent-1", "Before reset")

        with patch.object(
            self.repository,
            "archive_session_file",
            side_effect=OSError("archive failed"),
        ):
            result = self.service.reset_session(session_id, "agent-1")

        self.assertEqual(result, {"archived": False})
        current = self.repository.load_session(session_id, "agent-1")
        self.assertEqual(current["messages"], [])
        self.assertNotIn("label", current)


if __name__ == "__main__":
    unittest.main()
