from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_record import SessionWorkRecord
from sessions.session_work_store import (
    SessionWorkRecord as CompatibilityWorkRecord,
)


class SessionWorkRecordBoundaryTests(unittest.TestCase):
    def test_store_keeps_the_same_record_compatibility_export(self) -> None:
        self.assertIs(CompatibilityWorkRecord, SessionWorkRecord)

    def test_record_keeps_persistence_defaults(self) -> None:
        record = SessionWorkRecord(
            id="work-1",
            kind="cron",
            agent_id="main",
            session_id="main-main",
            content="report",
            priority=1,
        )

        self.assertEqual(record.prompt_mode, "minimal")
        self.assertEqual(record.persist_role, "system")
        self.assertEqual(record.status, "queued")
        self.assertFalse(record.recover_on_restart)
        self.assertEqual(record.created_at_ms, 0)
        self.assertIsNone(record.started_at_ms)
        self.assertIsNone(record.finished_at_ms)
        self.assertIsNone(record.last_error)

    def test_services_depend_on_the_record_module_not_sqlite_store(self) -> None:
        paths = [
            BACKEND_DIR / "scheduler" / "task_history_service.py",
            BACKEND_DIR / "sessions" / "session_work_delivery.py",
            BACKEND_DIR / "sessions" / "session_work_recovery_resolver.py",
        ]

        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "from sessions.session_work_store import SessionWorkRecord",
                source,
                path.name,
            )
            self.assertIn(
                "from sessions.session_work_record import SessionWorkRecord",
                source,
                path.name,
            )


if __name__ == "__main__":
    unittest.main()
