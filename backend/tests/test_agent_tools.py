from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_service import (
    SpawnResult,
    SubagentListResult,
)
from tools.agent_tools import get_agent_tools


class AgentToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = Mock()
        self.tools = {
            tool.name: tool
            for tool in get_agent_tools(
                "main",
                subagent_service=self.service,
                session_id="main-main",
            )
        }

    async def test_sessions_spawn_delegates_to_service(self) -> None:
        record = SimpleNamespace(
            run_id="run-1",
            child_session_key=(
                "agent:worker:subagent:subagent-child-1"
            ),
        )
        self.service.spawn.return_value = SpawnResult(
            record=record,
            session_id="subagent-child-1",
        )

        response = await self.tools["sessions_spawn"]._arun(
            task="inspect files",
            agent_id="worker",
            label="inspection",
            model="provider/model",
        )

        self.service.spawn.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id="main-main",
            task="inspect files",
            target_agent_id="worker",
            label="inspection",
            model="provider/model",
        )
        self.assertIn("run_id: run-1", response)
        self.assertIn(record.child_session_key, response)

    async def test_subagents_list_delegates_to_service(self) -> None:
        record = SimpleNamespace(
            run_id="run-1",
            label="inspection",
            target_agent_id="worker",
            ended_at=None,
            outcome=None,
            started_at=None,
            task="inspect files",
        )
        self.service.list_runs.return_value = SubagentListResult(
            records=[record],
            recent_minutes=15,
            requester_key="agent:main:main",
        )

        response = await self.tools["subagents"]._arun(
            action="list",
            recent_minutes=15,
        )

        self.service.list_runs.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id="main-main",
            recent_minutes=15,
        )
        self.assertIn("[run-1] inspection", response)


if __name__ == "__main__":
    unittest.main()
