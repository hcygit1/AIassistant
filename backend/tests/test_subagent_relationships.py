from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRunRecord
from subagents.subagent_relationships import SubagentRelationshipService


class SubagentRelationshipServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            SubagentRunRecord(
                run_id="run-root-child",
                child_session_key="agent:worker:subagent:child-1",
                requester_session_key="agent:main:main",
                requester_agent_id="main",
                target_agent_id="worker",
                task="root child",
                spawn_depth=1,
                created_at=1.0,
            ),
            SubagentRunRecord(
                run_id="run-nested-child",
                child_session_key="agent:worker:subagent:child-2",
                requester_session_key="agent:worker:subagent:child-1",
                requester_agent_id="worker",
                target_agent_id="worker",
                task="nested child",
                spawn_depth=2,
                created_at=2.0,
            ),
        ]
        self.service = SubagentRelationshipService(lambda: list(self.records))

    def test_resolves_depth_and_requester_from_canonical_child_keys(self) -> None:
        self.assertEqual(
            self.service.get_requester_depth(
                "agent:worker:subagent:child-1"
            ),
            1,
        )
        self.assertEqual(
            self.service.resolve_requester_for_child_session(
                "agent:worker:subagent:child-2"
            ),
            ("agent:worker:subagent:child-1", "worker"),
        )

    def test_lists_active_descendants_once_in_newest_first_order(self) -> None:
        self.assertEqual(
            [
                record.run_id
                for record in self.service.list_descendant_runs(
                    "agent:main:main"
                )
            ],
            ["run-nested-child", "run-root-child"],
        )
        self.assertEqual(
            self.service.count_active_descendant_runs("agent:main:main"),
            2,
        )

    def test_cycle_does_not_repeat_descendant_nodes(self) -> None:
        self.records.append(
            SubagentRunRecord(
                run_id="run-cycle",
                child_session_key="agent:main:main",
                requester_session_key="agent:worker:subagent:child-2",
                requester_agent_id="worker",
                target_agent_id="main",
                task="cycle",
                created_at=3.0,
            )
        )

        descendants = self.service.list_descendant_runs("agent:main:main")

        self.assertEqual(
            [record.run_id for record in descendants],
            ["run-cycle", "run-nested-child", "run-root-child"],
        )
        self.assertEqual(
            self.service.count_active_descendant_runs("agent:main:main"),
            3,
        )

    def test_each_descendant_query_uses_one_registry_snapshot(self) -> None:
        calls = 0

        def list_runs() -> list[SubagentRunRecord]:
            nonlocal calls
            calls += 1
            return list(self.records)

        service = SubagentRelationshipService(list_runs)

        service.list_descendant_runs("agent:main:main")
        self.assertEqual(calls, 1)

        calls = 0
        service.count_active_descendant_runs("agent:main:main")
        self.assertEqual(calls, 1)

    def test_preserves_recent_list_and_active_count_traversal_rules(self) -> None:
        self.records[0].ended_at = 1.0

        self.assertEqual(
            self.service.list_descendant_runs(
                "agent:main:main",
                include_recent_minutes=1,
            ),
            [],
        )
        self.assertEqual(
            self.service.count_active_descendant_runs("agent:main:main"),
            1,
        )

    def test_duplicate_child_key_keeps_runs_and_expands_subtree_once(self) -> None:
        self.records.append(
            SubagentRunRecord(
                run_id="run-root-child-retry",
                child_session_key="agent:worker:subagent:child-1",
                requester_session_key="agent:main:main",
                requester_agent_id="main",
                target_agent_id="worker",
                task="root child retry",
                spawn_depth=1,
                created_at=1.5,
            )
        )

        descendants = self.service.list_descendant_runs("agent:main:main")

        self.assertEqual(
            [record.run_id for record in descendants],
            [
                "run-nested-child",
                "run-root-child-retry",
                "run-root-child",
            ],
        )
        self.assertEqual(
            self.service.count_active_descendant_runs("agent:main:main"),
            3,
        )


if __name__ == "__main__":
    unittest.main()
