from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_manager import SessionManager
from sessions.session_reader import SessionReader
from sessions.session_title import SessionTitleService


class SessionReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Mock()
        self.title_service = SessionTitleService()
        self.reader = SessionReader(
            repository=self.repository,
            is_bootstrap_text=self.title_service.is_bootstrap_text,
        )

    def test_load_session_normalizes_bootstrap_and_legacy_labels(self) -> None:
        bootstrap = {
            "label": "[System Message] initialize workspace",
            "title": "Legacy title",
            "messages": [],
        }
        self.repository.load_session.return_value = bootstrap

        loaded = self.reader.load_session("session-1", "agent-1")

        self.assertIs(loaded, bootstrap)
        self.assertEqual(loaded["label"], "Legacy title")
        self.repository.load_session.assert_called_once_with(
            "session-1",
            "agent-1",
        )

    def test_load_session_ignores_bootstrap_legacy_title(self) -> None:
        data = {
            "title": "A new session was started via /new or /reset",
            "messages": [],
        }
        self.repository.load_session.return_value = data

        loaded = self.reader.load_session("session-1", "agent-1")

        self.assertNotIn("label", loaded)

    def test_load_session_for_agent_merges_consecutive_assistant_messages(self) -> None:
        self.repository.load_session.return_value = {
            "messages": [
                {"role": "user", "content": "question", "id": "u1"},
                {"role": "assistant", "content": "first", "id": "a1"},
                {"role": "assistant", "content": "second", "id": "a2"},
                {"content": "follow-up", "id": "u2"},
            ]
        }

        messages = self.reader.load_session_for_agent("session-1", "agent-1")

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first\n\nsecond"},
                {"role": "user", "content": "follow-up"},
            ],
        )

    def test_missing_session_returns_empty_agent_history(self) -> None:
        self.repository.load_session.return_value = None

        self.assertIsNone(self.reader.load_session("session-1", "agent-1"))
        self.assertEqual(
            self.reader.load_session_for_agent("session-1", "agent-1"),
            [],
        )


class SessionManagerReaderFacadeTests(unittest.TestCase):
    def test_manager_preserves_reader_facade(self) -> None:
        reader = Mock()
        reader.load_session.return_value = {"messages": []}
        reader.load_session_for_agent.return_value = [
            {"role": "user", "content": "hello"}
        ]
        manager = SessionManager(repository=Mock(), reader=reader)

        self.assertEqual(
            manager.load_session("session-1", "agent-1"),
            {"messages": []},
        )
        self.assertEqual(
            manager.load_session_for_agent("session-1", "agent-1"),
            [{"role": "user", "content": "hello"}],
        )
        reader.load_session.assert_called_once_with("session-1", "agent-1")
        reader.load_session_for_agent.assert_called_once_with(
            "session-1",
            "agent-1",
        )


if __name__ == "__main__":
    unittest.main()
