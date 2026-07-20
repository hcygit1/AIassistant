from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_history_archive import SessionHistoryArchive
from sessions.session_repository import SessionRepository


class SessionHistoryArchiveTests(unittest.TestCase):
    def test_compress_archives_messages_and_updates_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repository = SessionRepository(
                resolve_sessions_dir=lambda _agent_id: root,
            )
            data = {
                "session_id": "s1",
                "agent_id": "main",
                "created_at": 1,
                "updated_at": 2,
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ],
            }
            original_messages = list(data["messages"])
            repository.save_session("s1", "main", data)
            archive = SessionHistoryArchive(
                repository=repository,
                load_session=repository.load_session,
                save_session=repository.save_session,
                now=lambda: 100.0,
            )

            result = archive.compress_history("s1", "main", 2)

            saved = repository.load_session("s1", "main")
            archives = list((root / "archive").glob("s1_100.json"))
            archive_content = json.loads(
                archives[0].read_text(encoding="utf-8")
            )
            compaction_log = (root / "compactions.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result, {"archived_count": 2, "remaining_count": 2})
        self.assertEqual(saved["messages"], original_messages[2:])
        self.assertEqual(len(archives), 1)
        self.assertEqual(archive_content, original_messages[:2])
        self.assertIn('"archived_count": 2', compaction_log)


if __name__ == "__main__":
    unittest.main()
