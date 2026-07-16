from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.task_store import (
    TaskKind,
    TaskRecord,
    TaskStatus,
    TaskStore,
)


class TaskStoreTests(unittest.TestCase):
    def test_count_uses_same_filters_as_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.db")
            store.insert(
                TaskRecord(
                    id="matching",
                    kind=TaskKind.HEARTBEAT,
                    agent_id="main",
                    status=TaskStatus.SUCCESS,
                    created_at_ms=1,
                )
            )
            store.insert(
                TaskRecord(
                    id="other-status",
                    kind=TaskKind.HEARTBEAT,
                    agent_id="main",
                    status=TaskStatus.FAILED,
                    created_at_ms=2,
                )
            )
            store.insert(
                TaskRecord(
                    id="other-kind",
                    kind=TaskKind.CRON,
                    agent_id="main",
                    status=TaskStatus.SUCCESS,
                    created_at_ms=3,
                )
            )

            count = store.count(
                agent_id="main",
                kind=TaskKind.HEARTBEAT,
                status=TaskStatus.SUCCESS,
            )

        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
