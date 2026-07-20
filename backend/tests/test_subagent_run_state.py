from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRunRecord
from subagents.subagent_run_state import SubagentRunStateService
from subagents.subagent_run_store import SubagentRunStore


class SubagentRunStateServiceTests(unittest.TestCase):
    def test_mark_completed_updates_run_and_persists_state(self) -> None:
        saved: list[int] = []
        store = SubagentRunStore(load_runs=lambda: {}, save_runs=lambda _: saved.append(1))
        record = SubagentRunRecord(
            run_id="run-1",
            child_session_key="agent:worker:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
        )
        with store.locked_records() as records:
            records[record.run_id] = record

        service = SubagentRunStateService(
            store=store,
            persist=store.persist,
            now=lambda: 100.0,
        )
        service.mark_completed(
            "run-1",
            result_summary="done",
            terminal_reason="completed normally",
        )

        self.assertEqual(record.state, "succeeded")
        self.assertEqual(record.result_summary, "done")
        self.assertEqual(record.ended_at, 100.0)
        self.assertEqual(record.result_delivery_state, "pending")
        self.assertEqual(saved, [1])

    def test_termination_maps_timeout_to_timed_out_without_persisting_record_twice(self) -> None:
        saved: list[int] = []
        store = SubagentRunStore(load_runs=lambda: {}, save_runs=lambda _: saved.append(1))
        record = SubagentRunRecord(
            run_id="run-1",
            child_session_key="agent:worker:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
        )
        service = SubagentRunStateService(
            store=store,
            persist=store.persist,
            now=lambda: 100.0,
        )

        service.terminate_record(record, "timeout")

        self.assertEqual(record.state, "timed_out")
        self.assertEqual(record.outcome, "timeout")
        self.assertEqual(record.terminal_reason, "timeout")
        self.assertEqual(record.ended_at, 100.0)
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
