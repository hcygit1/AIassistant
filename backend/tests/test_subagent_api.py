from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.events import (
    SubagentKillRequest,
    SubagentSteerRequest,
    kill_subagents,
    steer_subagent,
)
from subagents.subagent_service import KillResult, SteerResult
from subagents.subagent_service import SubagentListResult
from subagents.subagent_registry import SubagentRunRecord


class SubagentApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_builds_tree_and_counts_all_active_descendants(self) -> None:
        parent = self._record(
            "run-parent",
            "agent:main:main",
            "agent:worker:subagent:child-1",
        )
        child = self._record(
            "run-child",
            parent.child_session_key,
            "agent:worker:subagent:child-2",
        )
        grandchild = self._record(
            "run-grandchild",
            child.child_session_key,
            "agent:worker:subagent:child-3",
        )
        service = Mock()
        service.list_runs.return_value = SubagentListResult(
            records=[parent, child, grandchild],
            recent_minutes=30,
            requester_key="agent:main:main",
        )
        service.child_requester_key.side_effect = (
            lambda record: record.child_session_key
        )

        with (
            patch(
                "runtime.agent.agent_manager.subagent_service",
                service,
                create=True,
            ),
            patch(
                "sessions.session_manager.session_manager.resolve_main_session_id",
                return_value="main-main",
            ),
            patch(
                "sessions.session_manager.session_manager.load_session",
                return_value=None,
            ),
        ):
            from api.events import list_subagents

            result = await list_subagents("main")

        service.list_runs.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id="main-main",
            recent_minutes=None,
            recursive=True,
        )
        self.assertEqual(len(result["tree"]), 1)
        self.assertEqual(
            result["tree"][0]["descendants_active_count"],
            2,
        )
        self.assertEqual(
            result["tree"][0]["children"][0][
                "descendants_active_count"
            ],
            1,
        )

    async def test_kill_delegates_to_agent_manager_service(self) -> None:
        service = Mock()
        service.kill.return_value = KillResult(
            killed=1,
            scope="agent:main:main",
            run_id="run-1",
        )

        with patch(
            "runtime.agent.agent_manager.subagent_service",
            service,
            create=True,
        ):
            result = await kill_subagents(
                "main",
                SubagentKillRequest(
                    target="run-1",
                    session_id="main-main",
                ),
            )

        service.kill.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id="main-main",
            target="run-1",
        )
        self.assertEqual(
            result,
            {"ok": True, "run_id": "run-1"},
        )

    async def test_steer_delegates_to_agent_manager_service(self) -> None:
        service = Mock()
        service.steer.return_value = SteerResult(
            record=Mock(run_id="run-2"),
            replaced_run_id="run-1",
            label="inspection",
        )

        with patch(
            "runtime.agent.agent_manager.subagent_service",
            service,
            create=True,
        ):
            result = await steer_subagent(
                "main",
                SubagentSteerRequest(
                    run_id="run-1",
                    message="inspect tests",
                ),
            )

        service.steer.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id=None,
            run_id="run-1",
            message="inspect tests",
        )
        self.assertEqual(
            result,
            {
                "ok": True,
                "run_id": "run-2",
                "replaced_run_id": "run-1",
            },
        )

    @staticmethod
    def _record(
        run_id: str,
        requester_key: str,
        child_key: str,
    ) -> SubagentRunRecord:
        return SubagentRunRecord(
            run_id=run_id,
            child_session_key=child_key,
            requester_session_key=requester_key,
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            started_at=1.0,
        )


if __name__ == "__main__":
    unittest.main()
