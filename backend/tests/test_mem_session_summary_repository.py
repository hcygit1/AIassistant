from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class SessionSummaryRepositoryTests(unittest.TestCase):
    def test_session_summary_persistence_has_a_repository_owner(self) -> None:
        repository_path = BACKEND_DIR / "mem" / "session_summary_repository.py"

        self.assertTrue(
            repository_path.is_file(),
            "mem.session_summary_repository should own summary persistence",
        )
        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SessionSummaryRepository(", store_source)
        self.assertNotIn("INSERT INTO session_summaries", store_source)
        self.assertNotIn("UPDATE session_summaries SET", store_source)
        self.assertNotIn("DELETE FROM session_summaries", store_source)

    def test_repository_preserves_fields_and_scoped_delete_semantics(self) -> None:
        from mem.session_summary_repository import SessionSummaryRepository

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "summaries.db"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.execute(
                """CREATE TABLE session_summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    goal TEXT,
                    decisions TEXT,
                    progress TEXT,
                    open_items TEXT,
                    entities TEXT,
                    user_preferences TEXT,
                    raw_summary TEXT,
                    token_count INTEGER,
                    created_at REAL,
                    updated_at REAL,
                    UNIQUE(session_id, agent_id)
                )"""
            )
            repository = SessionSummaryRepository(
                connection,
                now=lambda: 100.0,
                new_id=lambda: "summary-id",
            )

            created = repository.upsert(
                "session-1",
                "agent-1",
                {
                    "goal": "goal",
                    "decisions": ["decision"],
                    "progress": "progress",
                    "open_items": ["item"],
                    "entities": ["entity"],
                    "user_preferences": ["preference"],
                    "raw_summary": "raw",
                    "token_count": 42,
                },
            )
            stored = repository.get("session-1", "agent-1")
            with sqlite3.connect(path) as observer:
                count_after_upsert = observer.execute(
                    "SELECT COUNT(*) FROM session_summaries"
                ).fetchone()[0]

            wrong_scope_deleted = repository.delete("session-1", "agent-2")
            deleted = repository.delete("session-1", "agent-1")
            deleted_again = repository.delete("session-1", "agent-1")
            missing = repository.get("session-1", "agent-1")
            with sqlite3.connect(path) as observer:
                count_after_delete = observer.execute(
                    "SELECT COUNT(*) FROM session_summaries"
                ).fetchone()[0]
            connection.close()

        self.assertEqual(stored, created)
        self.assertEqual(created.decisions, '["decision"]')
        self.assertEqual(created.open_items, '["item"]')
        self.assertEqual(created.entities, '["entity"]')
        self.assertEqual(created.user_preferences, '["preference"]')
        self.assertEqual(created.raw_summary, "raw")
        self.assertEqual(created.token_count, 42)
        self.assertEqual(count_after_upsert, 1)
        self.assertFalse(wrong_scope_deleted)
        self.assertTrue(deleted)
        self.assertFalse(deleted_again)
        self.assertIsNone(missing)
        self.assertEqual(count_after_delete, 0)

    def test_update_preserves_created_at_in_returned_and_stored_summary(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    import tempfile
                    from dataclasses import asdict
                    from pathlib import Path
                    from unittest.mock import patch
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        with (
                            patch(
                                "mem.store.time.time",
                                side_effect=[100.0, 200.0],
                            ),
                            patch(
                                "mem.store.uuid.uuid4",
                                return_value="summary-id",
                            ),
                        ):
                            created = store.upsert_session_summary(
                                "session-1",
                                "agent-1",
                                {"goal": "first", "decisions": ["one"]},
                            )
                            updated = store.upsert_session_summary(
                                "session-1",
                                "agent-1",
                                {"goal": "second", "decisions": ["two"]},
                            )

                        stored = store.get_session_summary(
                            "session-1", "agent-1"
                        )
                        store.close()
                        print(json.dumps({
                            "created": asdict(created),
                            "updated": asdict(updated),
                            "stored": asdict(stored),
                        }, sort_keys=True))
                    """
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.splitlines()[-1])

        created = payload["created"]
        updated = payload["updated"]
        stored = payload["stored"]
        self.assertEqual(created["id"], "summary-id")
        self.assertEqual(updated["id"], created["id"])
        self.assertEqual(
            (created["version"], updated["version"], stored["version"]),
            (1, 2, 2),
        )
        self.assertEqual(created["created_at"], 100.0)
        self.assertEqual(updated["created_at"], 100.0)
        self.assertEqual(stored["created_at"], 100.0)
        self.assertEqual(updated["updated_at"], 200.0)
        self.assertEqual(stored["updated_at"], 200.0)
        self.assertEqual(stored["goal"], "second")
        self.assertEqual(stored["decisions"], '["two"]')


if __name__ == "__main__":
    unittest.main()
