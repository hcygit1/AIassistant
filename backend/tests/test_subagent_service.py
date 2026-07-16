from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRegistry
from subagents.subagent_service import (
    SubagentService,
    SubagentServiceError,
)


class SubagentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(SubagentRegistry, "_restore_from_disk"):
            self.registry = SubagentRegistry()
        self.registry._persist_to_disk = Mock()
        self.session_manager = Mock()
        self.session_manager.resolve_main_session_id.side_effect = (
            lambda agent_id: f"{agent_id}-main"
        )
        self.session_manager.session_key_from_session_id.side_effect = (
            lambda agent_id, session_id: (
                f"agent:{agent_id}:main"
                if session_id == f"{agent_id}-main"
                else f"agent:{agent_id}:subagent:{session_id}"
            )
        )
        self.session_manager.session_id_from_session_key.side_effect = (
            self._parse_session_key
        )
        self.runner = Mock()
        self.runner_factory = Mock(return_value=self.runner)
        self.ids = iter([
            "child0000001",
            "run000000001",
            "run000000002",
        ])
        self.configs = {
            "main": {
                "subagents": {
                    "allow_agents": ["worker"],
                    "max_spawn_depth": 2,
                    "max_children_per_agent": 2,
                    "run_timeout_seconds": 15,
                }
            },
            "worker": {
                "subagents": {
                    "allow_agents": ["*"],
                    "max_spawn_depth": 2,
                    "max_children_per_agent": 2,
                    "run_timeout_seconds": 15,
                }
            },
        }
        self.service = SubagentService(
            registry=self.registry,
            session_manager=self.session_manager,
            resolve_agent_config=lambda agent_id: self.configs[agent_id],
            get_config=lambda: {
                "agents": {
                    "defaults": {
                        "subagents": {"recent_minutes": 45}
                    }
                }
            },
            runner_factory=self.runner_factory,
            id_factory=lambda: next(self.ids),
        )

    @staticmethod
    def _parse_session_key(key: str):
        parts = key.split(":")
        if len(parts) < 3 or parts[0] != "agent":
            return None
        if parts[2] == "main":
            return parts[1], f"{parts[1]}-main"
        if len(parts) >= 4 and parts[2] == "subagent":
            return parts[1], parts[3]
        return None

    def test_spawn_registers_session_and_starts_runner(self) -> None:
        result = self.service.spawn(
            requester_agent_id="main",
            requester_session_id="main-main",
            task="inspect files",
            target_agent_id="worker",
            label="inspection",
            model="provider/model",
        )

        self.assertEqual(result.record.run_id, "run000000001")
        self.assertEqual(result.session_id, "subagent-child0000001")
        self.assertEqual(result.record.spawn_depth, 1)
        self.session_manager.ensure_session.assert_called_once_with(
            "subagent-child0000001",
            "worker",
            spawned_by="agent:main:main",
            label="inspection",
        )
        self.runner_factory.assert_called_once_with("main")
        self.runner.start.assert_called_once_with(
            run_id="run000000001",
            session_id="subagent-child0000001",
            agent_id="worker",
            task="inspect files",
            requester_key="agent:main:main",
            run_timeout_seconds=15,
        )

    def test_agent_manager_owns_subagent_service(self) -> None:
        from runtime.agent import AgentManager

        manager = AgentManager()

        self.assertIsInstance(
            manager.subagent_service,
            SubagentService,
        )

    def test_spawn_failure_marks_registered_run_terminal(self) -> None:
        self.runner.start.side_effect = RuntimeError("task factory failed")

        with self.assertRaises(SubagentServiceError) as raised:
            self.service.spawn(
                requester_agent_id="main",
                requester_session_id="main-main",
                task="inspect files",
                target_agent_id="worker",
            )

        self.assertEqual(raised.exception.code, "start_failed")
        record = self.registry.get_run("run000000001")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "failed")  # type: ignore[union-attr]
        self.assertIn("task factory failed", record.outcome)  # type: ignore[union-attr]

    def test_zero_depth_and_children_limits_disable_spawn(self) -> None:
        self.configs["worker"]["subagents"][
            "max_spawn_depth"
        ] = 0

        with self.assertRaises(SubagentServiceError) as depth_error:
            self.service.spawn(
                requester_agent_id="main",
                requester_session_id="main-main",
                task="inspect files",
                target_agent_id="worker",
            )
        self.assertEqual(depth_error.exception.code, "depth_limit")

        self.configs["worker"]["subagents"][
            "max_spawn_depth"
        ] = 2
        self.configs["worker"]["subagents"][
            "max_children_per_agent"
        ] = 0
        with self.assertRaises(SubagentServiceError) as child_error:
            self.service.spawn(
                requester_agent_id="main",
                requester_session_id="main-main",
                task="inspect files",
                target_agent_id="worker",
            )
        self.assertEqual(
            child_error.exception.code,
            "children_limit",
        )

    def test_nested_spawn_uses_parent_run_depth(self) -> None:
        self.registry.register_run(
            run_id="run-parent",
            child_session_key=(
                "agent:worker:subagent:child-existing"
            ),
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            spawn_depth=1,
        )
        self.configs["worker"]["subagents"][
            "max_spawn_depth"
        ] = 1

        with self.assertRaises(SubagentServiceError) as raised:
            self.service.spawn(
                requester_agent_id="worker",
                requester_session_id="child-existing",
                task="inspect nested files",
                target_agent_id="worker",
            )

        self.assertEqual(raised.exception.code, "depth_limit")

    def test_list_runs_uses_default_recent_window(self) -> None:
        self._register_run(
            "run-direct",
            requester_key="agent:main:main",
        )

        result = self.service.list_runs(
            requester_agent_id="main",
            requester_session_id="main-main",
        )

        self.assertEqual(result.recent_minutes, 45)
        self.assertEqual(
            [record.run_id for record in result.records],
            ["run-direct"],
        )

    def test_kill_rejects_run_outside_requester_scope(self) -> None:
        self._register_run(
            "run-foreign",
            requester_key="agent:main:subagent:other-session",
        )

        with self.assertRaises(SubagentServiceError) as raised:
            self.service.kill(
                requester_agent_id="main",
                requester_session_id="main-main",
                target="run-foreign",
            )

        self.assertEqual(raised.exception.code, "out_of_scope")
        self.assertIsNone(
            self.registry.get_run("run-foreign").ended_at  # type: ignore[union-attr]
        )

    def test_steer_replaces_scoped_run_and_starts_runner(self) -> None:
        self._register_run(
            "run-old",
            requester_key="agent:main:main",
        )

        result = self.service.steer(
            requester_agent_id="main",
            requester_session_id="main-main",
            run_id="run-old",
            message="inspect tests instead",
        )

        self.assertEqual(result.record.run_id, "child0000001")
        self.assertEqual(result.replaced_run_id, "run-old")
        self.assertIsNone(self.registry.get_run("run-old"))
        self.session_manager.save_message.assert_called_once_with(
            "child-existing",
            "worker",
            "user",
            "inspect tests instead",
        )
        self.runner.start.assert_called_once_with(
            run_id="child0000001",
            session_id="child-existing",
            agent_id="worker",
            task="inspect tests instead",
            requester_key="agent:main:main",
            run_timeout_seconds=15,
        )

    def test_steer_claims_run_before_writing_message(self) -> None:
        entry = self.registry.register_run(
            run_id="run-old",
            child_session_key=(
                "agent:worker:subagent:child-existing"
            ),
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            label="inspection",
            spawn_depth=1,
        )
        next_record = self.registry.register_run(
            run_id="run-new",
            child_session_key=entry.child_session_key,
            requester_session_key=entry.requester_session_key,
            requester_agent_id=entry.requester_agent_id,
            target_agent_id=entry.target_agent_id,
            task="inspect tests",
            spawn_depth=entry.spawn_depth,
        )
        registry = Mock()
        registry.list_descendant_runs.return_value = [entry]
        registry.get_run.side_effect = [entry, next_record]
        registry.replace_active_run_for_steer.return_value = (
            next_record
        )
        calls = Mock()
        calls.attach_mock(
            registry.replace_active_run_for_steer,
            "claim",
        )
        calls.attach_mock(
            self.session_manager.save_message,
            "save",
        )
        service = SubagentService(
            registry=registry,
            session_manager=self.session_manager,
            resolve_agent_config=lambda agent_id: self.configs[agent_id],
            get_config=lambda: {},
            runner_factory=self.runner_factory,
            id_factory=lambda: "run-new",
        )

        service.steer(
            requester_agent_id="main",
            requester_session_id="main-main",
            run_id="run-old",
            message="inspect tests",
        )

        self.assertEqual(
            [call[0] for call in calls.mock_calls[:2]],
            ["claim", "save"],
        )
        registry.replace_active_run_for_steer.assert_called_once_with(
            previous_run_id="run-old",
            next_run_id="run-new",
            task="inspect tests",
        )

    def _register_run(
        self,
        run_id: str,
        *,
        requester_key: str,
    ) -> None:
        self.registry.register_run(
            run_id=run_id,
            child_session_key=(
                "agent:worker:subagent:child-existing"
            ),
            requester_session_key=requester_key,
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            label="inspection",
            spawn_depth=1,
        )


if __name__ == "__main__":
    unittest.main()
