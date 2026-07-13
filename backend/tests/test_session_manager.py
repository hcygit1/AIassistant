from __future__ import annotations

import json
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import sessions.session_manager as session_manager_module
from sessions.session_manager import SessionManager


class SessionManagerPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.temp_dir.name)
        self.path_patch = patch(
            "sessions.session_manager.resolve_agent_sessions_dir",
            return_value=self.sessions_dir,
        )
        self.path_patch.start()
        SessionManager._cache.clear()
        self.manager = SessionManager()

    def tearDown(self) -> None:
        SessionManager._cache.clear()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_load_session_raises_when_json_is_corrupted(self) -> None:
        path = self.sessions_dir / "session-1.json"
        path.write_text('{"messages": [', encoding="utf-8")
        error_type = getattr(
            session_manager_module,
            "SessionDataCorruptionError",
            None,
        )

        self.assertIsNotNone(error_type)
        with self.assertRaises(error_type):
            self.manager.load_session("session-1", "agent-1")

    def test_ensure_session_does_not_overwrite_corrupted_file(self) -> None:
        path = self.sessions_dir / "session-1.json"
        corrupted_content = '{"messages": ['
        path.write_text(corrupted_content, encoding="utf-8")
        error_type = getattr(
            session_manager_module,
            "SessionDataCorruptionError",
            None,
        )

        self.assertIsNotNone(error_type)
        with self.assertRaises(error_type):
            self.manager.ensure_session("session-1", "agent-1")

        self.assertEqual(path.read_text(encoding="utf-8"), corrupted_content)

    def test_load_session_rejects_non_object_json(self) -> None:
        error_type = session_manager_module.SessionDataCorruptionError
        path = self.sessions_dir / "session-1.json"

        for content in ("null", '"text"', "1", "true"):
            with self.subTest(content=content):
                SessionManager._cache.clear()
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(error_type):
                    self.manager.load_session("session-1", "agent-1")

    def test_failed_serialization_preserves_existing_session(self) -> None:
        path = self.sessions_dir / "session-1.json"
        original_data = {"messages": [{"role": "user", "content": "original"}]}
        original_content = json.dumps(original_data, ensure_ascii=False)
        path.write_text(original_content, encoding="utf-8")

        with self.assertRaises(TypeError):
            self.manager._save_session_data(
                "session-1",
                "agent-1",
                {"messages": [], "invalid": object()},
            )

        self.assertEqual(path.read_text(encoding="utf-8"), original_content)
        self.assertEqual(list(self.sessions_dir.glob("*.tmp")), [])

    def test_failed_replace_preserves_existing_session_store(self) -> None:
        path = self.sessions_dir / "sessions.json"
        original_store = {
            "agent:agent-1:main": {
                "sessionId": "agent-1-main",
                "updatedAt": 1,
            }
        }
        path.write_text(
            json.dumps(original_store, ensure_ascii=False),
            encoding="utf-8",
        )

        caught_error = None
        try:
            with patch("os.replace", side_effect=OSError("replace failed")):
                self.manager._save_session_store(
                    "agent-1",
                    {
                        "agent:agent-1:main": {
                            "sessionId": "agent-1-main",
                            "updatedAt": 2,
                        }
                    },
                )
        except OSError as exc:
            caught_error = exc

        self.assertIsNotNone(caught_error)
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            original_store,
        )
        self.assertEqual(list(self.sessions_dir.glob("*.tmp")), [])

    def test_update_does_not_overwrite_corrupted_session_store(self) -> None:
        path = self.sessions_dir / "sessions.json"
        corrupted_content = '{"agent:agent-1:main":'
        path.write_text(corrupted_content, encoding="utf-8")

        with self.assertRaises(
            session_manager_module.SessionDataCorruptionError
        ):
            self.manager._update_session_store_entry(
                "agent-1",
                "agent:agent-1:main",
                "agent-1-main",
                1,
            )

        self.assertEqual(path.read_text(encoding="utf-8"), corrupted_content)

    def test_concurrent_session_updates_do_not_lose_store_entries(self) -> None:
        original_load = self.manager._load_session_store
        first_loaded = threading.Event()
        second_loaded = threading.Event()
        release_first = threading.Event()
        count_lock = threading.Lock()
        load_count = 0
        errors: list[Exception] = []

        def controlled_load(agent_id: str) -> dict:
            nonlocal load_count
            store = original_load(agent_id)
            with count_lock:
                load_count += 1
                current_call = load_count
            if current_call == 1:
                first_loaded.set()
                release_first.wait(timeout=2)
            elif current_call == 2:
                second_loaded.set()
            return store

        def update_entry(session_id: str) -> None:
            try:
                self.manager._update_session_store_entry(
                    "agent-1",
                    f"agent:agent-1:subagent:{session_id}",
                    session_id,
                    1,
                )
            except Exception as exc:
                errors.append(exc)

        with patch.object(
            self.manager,
            "_load_session_store",
            side_effect=controlled_load,
        ):
            first = threading.Thread(target=update_entry, args=("session-1",))
            second = threading.Thread(target=update_entry, args=("session-2",))
            first.start()
            self.assertTrue(first_loaded.wait(timeout=2))
            second.start()
            second_loaded_before_release = second_loaded.wait(timeout=0.1)
            release_first.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(second_loaded_before_release)
        self.assertEqual(errors, [])
        store = json.loads(
            (self.sessions_dir / "sessions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(store),
            {
                "agent:agent-1:subagent:session-1",
                "agent:agent-1:subagent:session-2",
            },
        )

    def test_successful_save_writes_valid_json_without_temp_files(self) -> None:
        session_data = {
            "session_id": "session-1",
            "agent_id": "agent-1",
            "updated_at": 1,
            "messages": [{"role": "user", "content": "hello"}],
        }

        self.manager._save_session_data("session-1", "agent-1", session_data)

        session_path = self.sessions_dir / "session-1.json"
        store_path = self.sessions_dir / "sessions.json"
        self.assertEqual(
            json.loads(session_path.read_text(encoding="utf-8")),
            session_data,
        )
        self.assertEqual(
            json.loads(store_path.read_text(encoding="utf-8"))[
                "agent:agent-1:subagent:session-1"
            ]["sessionId"],
            "session-1",
        )
        self.assertEqual(list(self.sessions_dir.glob("*.tmp")), [])

    def test_atomic_save_preserves_existing_file_mode(self) -> None:
        path = self.sessions_dir / "sessions.json"
        path.write_text("{}", encoding="utf-8")
        path.chmod(0o640)

        self.manager._save_session_store("agent-1", {})

        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)

    def test_atomic_save_preserves_mode_without_fchmod(self) -> None:
        path = self.sessions_dir / "sessions.json"
        path.write_text("{}", encoding="utf-8")
        path.chmod(0o640)

        caught_error = None
        with patch.object(session_manager_module.os, "fchmod", None):
            try:
                self.manager._save_session_store("agent-1", {})
            except Exception as exc:
                caught_error = exc

        self.assertIsNone(caught_error)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
