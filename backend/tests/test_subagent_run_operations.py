from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_run_operations import SubagentRunOperations
from subagents.subagent_scope import SubagentScopeResolver
from subagents.subagent_service_models import SubagentServiceError


class SubagentScopeResolverTests(unittest.TestCase):
    def test_resolves_main_session_and_recent_window(self) -> None:
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        session_manager.session_key_from_session_id.return_value = (
            "agent:main:main"
        )
        scope = SubagentScopeResolver(
            session_manager=session_manager,
            get_config=lambda: {
                "agents": {
                    "defaults": {
                        "subagents": {"recent_minutes": 45}
                    }
                }
            },
        )

        requester_key = scope.requester_key("main", None)

        self.assertEqual(requester_key, "agent:main:main")
        self.assertEqual(scope.recent_minutes(None), 45)
        self.assertEqual(scope.recent_minutes(10_000), 24 * 60)
        session_manager.resolve_main_session_id.assert_called_once_with("main")


class SubagentRunOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Mock()
        self.scope = Mock()
        self.scope.requester_key.return_value = "agent:main:main"
        self.scope.recent_minutes.return_value = 45
        self.operations = SubagentRunOperations(
            registry=self.registry,
            scope=self.scope,
        )

    def test_list_runs_selects_direct_or_recursive_query(self) -> None:
        direct = [SimpleNamespace(run_id="direct")]
        recursive = [SimpleNamespace(run_id="recursive")]
        self.registry.list_runs_for_requester.return_value = direct
        self.registry.list_descendant_runs.return_value = recursive

        direct_result = self.operations.list_runs(
            requester_agent_id="main",
            requester_session_id="main-main",
            recent_minutes=None,
            recursive=False,
        )
        recursive_result = self.operations.list_runs(
            requester_agent_id="main",
            requester_session_id="main-main",
            recent_minutes=60,
            recursive=True,
        )

        self.assertEqual(direct_result.records, direct)
        self.assertEqual(recursive_result.records, recursive)
        self.registry.list_runs_for_requester.assert_called_once_with(
            "agent:main:main",
            include_recent_minutes=45,
        )
        self.registry.list_descendant_runs.assert_called_once_with(
            "agent:main:main",
            include_recent_minutes=45,
        )

    def test_kill_all_only_terminates_active_descendants(self) -> None:
        active = SimpleNamespace(run_id="active", ended_at=None)
        ended = SimpleNamespace(run_id="ended", ended_at=1.0)
        self.registry.list_descendant_runs.return_value = [active, ended]
        self.registry.kill.return_value = True

        result = self.operations.kill(
            requester_agent_id="main",
            requester_session_id="main-main",
            target="all",
        )

        self.assertEqual(result.killed, 1)
        self.assertEqual(result.scope, "agent:main:main")
        self.registry.kill.assert_called_once_with("active")

    def test_kill_rejects_target_outside_scope(self) -> None:
        self.registry.list_descendant_runs.return_value = []

        with self.assertRaises(SubagentServiceError) as raised:
            self.operations.kill(
                requester_agent_id="main",
                requester_session_id="main-main",
                target="foreign",
            )

        self.assertEqual(raised.exception.code, "out_of_scope")

    def test_kill_active_scoped_target(self) -> None:
        active = SimpleNamespace(run_id="active", ended_at=None)
        self.registry.list_descendant_runs.return_value = [active]
        self.registry.get_run.return_value = active
        self.registry.kill.return_value = True

        result = self.operations.kill(
            requester_agent_id="main",
            requester_session_id="main-main",
            target="active",
        )

        self.assertEqual(result.killed, 1)
        self.assertEqual(result.run_id, "active")
        self.registry.kill.assert_called_once_with("active")


if __name__ == "__main__":
    unittest.main()
