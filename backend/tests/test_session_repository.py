from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_repository import (
    SessionDataCorruptionError,
    SessionRepository,
)


class SessionRepositoryTests(unittest.TestCase):
    def test_cache_is_isolated_between_repository_roots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_tmp,
            tempfile.TemporaryDirectory() as second_tmp,
        ):
            first_root = Path(first_tmp)
            second_root = Path(second_tmp)
            (first_root / "session-1.json").write_text(
                '{"messages": [], "label": "first"}',
                encoding="utf-8",
            )
            (second_root / "session-1.json").write_text(
                '{"messages": [], "label": "second"}',
                encoding="utf-8",
            )
            first = SessionRepository(
                resolve_sessions_dir=lambda agent_id: first_root
            )
            second = SessionRepository(
                resolve_sessions_dir=lambda agent_id: second_root
            )

            first_data = first.load_session("session-1", "agent-1")
            second_data = second.load_session("session-1", "agent-1")

        self.assertEqual(first_data["label"], "first")
        self.assertEqual(second_data["label"], "second")

    def test_repositories_for_same_root_share_cache_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session-1.json").write_text(
                '{"messages": [], "label": "old"}',
                encoding="utf-8",
            )
            first = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            second = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            first.load_session("session-1", "agent-1")

            second.save_session(
                "session-1",
                "agent-1",
                {"messages": [], "label": "new"},
            )

            current = first.load_session("session-1", "agent-1")

        self.assertEqual(current["label"], "new")

    def test_load_holds_shared_lock_against_concurrent_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session-1.json").write_text(
                '{"messages": [], "label": "old"}',
                encoding="utf-8",
            )
            reader = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            deleter = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            original_load = json.load
            file_read = threading.Event()
            release_read = threading.Event()
            delete_finished = threading.Event()

            def delayed_load(file):
                data = original_load(file)
                file_read.set()
                release_read.wait(timeout=2)
                return data

            def delete() -> None:
                deleter.delete_session_file("session-1", "agent-1")
                delete_finished.set()

            with patch(
                "sessions.session_repository.json.load",
                side_effect=delayed_load,
            ):
                load_thread = threading.Thread(
                    target=reader.load_session,
                    args=("session-1", "agent-1"),
                )
                delete_thread = threading.Thread(target=delete)
                load_thread.start()
                self.assertTrue(file_read.wait(timeout=2))
                delete_thread.start()
                deleted_before_release = delete_finished.wait(timeout=0.1)
                release_read.set()
                load_thread.join(timeout=2)
                delete_thread.join(timeout=2)

            current = reader.load_session("session-1", "agent-1")

        self.assertFalse(deleted_before_release)
        self.assertIsNone(current)

    def test_load_session_normalizes_legacy_message_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session-1.json").write_text(
                json.dumps([{"role": "user", "content": "hello"}]),
                encoding="utf-8",
            )
            repository = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )

            data = repository.load_session("session-1", "agent-1")

        self.assertEqual(data["messages"][0]["content"], "hello")
        self.assertEqual(data["agent_id"], "agent-1")

    def test_load_index_rejects_corrupted_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sessions.json").write_text("{broken", encoding="utf-8")
            repository = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )

            with self.assertRaises(SessionDataCorruptionError):
                repository.load_index("agent-1")


if __name__ == "__main__":
    unittest.main()
