from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRunRecord
from subagents.subagent_run_queries import SubagentRunQueryService
from subagents.subagent_run_store import SubagentRunStore


class SubagentRunQueryServiceTests(unittest.TestCase):
    def test_returns_snapshots_and_filters_requester_runs(self) -> None:
        record = SubagentRunRecord(
            run_id="run-1",
            child_session_key="agent:worker:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            created_at=1.0,
        )
        store = SubagentRunStore(
            load_runs=lambda: {record.run_id: record},
            save_runs=lambda _runs: None,
        )
        store.restore()
        service = SubagentRunQueryService(
            store=store,
            snapshot=lambda item: SubagentRunRecord(
                run_id=item.run_id,
                child_session_key=item.child_session_key,
                requester_session_key=item.requester_session_key,
                requester_agent_id=item.requester_agent_id,
                target_agent_id=item.target_agent_id,
                task=item.task,
                created_at=item.created_at,
            ),
        )

        snapshot = service.get_run("run-1")
        self.assertIsNot(snapshot, record)
        self.assertEqual(
            service.list_runs_for_requester("agent:main:main"),
            [record],
        )
        self.assertEqual(
            service.count_active_for_requester("agent:main:main"),
            1,
        )
        self.assertEqual(
            service.list_run_entries()[0][0],
            "run-1",
        )


if __name__ == "__main__":
    unittest.main()
