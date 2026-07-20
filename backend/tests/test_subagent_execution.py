from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_execution import SubagentExecutionService
from subagents.subagent_run_model import SubagentRunRecord
from subagents.subagent_service import SubagentService


class SubagentExecutionServiceTests(unittest.TestCase):
    def test_spawn_registers_before_session_creation_and_runner_start(self) -> None:
        registry = Mock()
        registry.get_requester_depth.return_value = 0
        record = SubagentRunRecord(
            run_id="run-1",
            child_session_key="agent:worker:subagent:subagent-child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            label="inspection",
            spawn_depth=1,
        )
        registry.register_run.return_value = record
        registry.get_run.return_value = record
        session_manager = Mock()
        runner = Mock()
        runner_factory = Mock(return_value=runner)
        scope = Mock()
        scope.requester_key.return_value = "agent:main:main"
        ids = iter(["child-1", "run-1"])
        execution = SubagentExecutionService(
            registry=registry,
            session_manager=session_manager,
            resolve_agent_config=lambda agent_id: {
                "subagents": {
                    "allow_agents": ["worker"],
                    "max_spawn_depth": 2,
                    "max_children_per_agent": 2,
                    "run_timeout_seconds": 15,
                }
            },
            runner_factory=runner_factory,
            id_factory=lambda: next(ids),
            scope=scope,
        )
        calls = Mock()
        calls.attach_mock(registry.register_run, "register")
        calls.attach_mock(session_manager.ensure_session, "ensure")
        calls.attach_mock(runner.start, "start")

        result = execution.spawn(
            requester_agent_id="main",
            requester_session_id="main-main",
            task="inspect files",
            target_agent_id="worker",
            label="inspection",
        )

        self.assertEqual(result.record, record)
        self.assertEqual(result.session_id, "subagent-child-1")
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["register", "ensure", "start"],
        )

    def test_service_delegates_spawn_and_steer_to_execution(self) -> None:
        execution = Mock()
        spawn_result = object()
        steer_result = object()
        execution.spawn.return_value = spawn_result
        execution.steer.return_value = steer_result
        service = SubagentService(
            registry=Mock(),
            session_manager=Mock(),
            resolve_agent_config=Mock(),
            get_config=Mock(return_value={}),
            runner_factory=Mock(),
            id_factory=Mock(),
            scope=Mock(),
            run_operations=Mock(),
            execution=execution,
        )

        spawned = service.spawn(
            requester_agent_id="main",
            requester_session_id="main-main",
            task="inspect files",
            target_agent_id="worker",
            label="inspection",
            model="provider/model",
        )
        steered = service.steer(
            requester_agent_id="main",
            requester_session_id="main-main",
            run_id="run-1",
            message="inspect tests",
        )

        self.assertIs(spawned, spawn_result)
        self.assertIs(steered, steer_result)
        execution.spawn.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id="main-main",
            task="inspect files",
            target_agent_id="worker",
            label="inspection",
            model="provider/model",
        )
        execution.steer.assert_called_once_with(
            requester_agent_id="main",
            requester_session_id="main-main",
            run_id="run-1",
            message="inspect tests",
        )


if __name__ == "__main__":
    unittest.main()
