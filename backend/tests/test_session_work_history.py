from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_history import SessionWorkHistoryService


class SessionWorkHistoryServiceTests(unittest.TestCase):
    def test_query_uses_injected_runtime_and_maps_records(self) -> None:
        record = SimpleNamespace(
            id="work-1",
            kind="heartbeat",
            agent_id="main",
            session_id="main-main",
            run_id=None,
            status="done",
            recover_on_restart=False,
            created_at_ms=1,
            started_at_ms=2,
            finished_at_ms=3,
            last_error=None,
            content="x" * 250,
        )
        work_store = Mock()
        work_store.query.return_value = [record]
        work_store.count.return_value = 1
        runtime = SimpleNamespace(work_store=work_store)
        service = SessionWorkHistoryService(runtime=runtime)

        page = service.query(
            kind="heartbeat",
            status="done",
            agent_id="main",
            session_id="main-main",
            run_id=None,
            limit=10,
            offset=0,
        )

        expected_filters = {
            "kind": "heartbeat",
            "status": "done",
            "agent_id": "main",
            "session_id": "main-main",
            "run_id": None,
        }
        work_store.query.assert_called_once_with(
            **expected_filters,
            limit=10,
            offset=0,
        )
        work_store.count.assert_called_once_with(**expected_filters)
        self.assertEqual(page.total, 1)
        self.assertEqual(page.limit, 10)
        self.assertEqual(page.offset, 0)
        self.assertEqual(
            page.items,
            [{
                "id": "work-1",
                "kind": "heartbeat",
                "agent_id": "main",
                "session_id": "main-main",
                "run_id": None,
                "status": "done",
                "recover_on_restart": False,
                "created_at_ms": 1,
                "started_at_ms": 2,
                "finished_at_ms": 3,
                "last_error": None,
                "content_preview": "x" * 200,
            }],
        )


if __name__ == "__main__":
    unittest.main()
