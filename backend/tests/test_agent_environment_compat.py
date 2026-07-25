from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent import AgentManager
from runtime.agent_environment_compat import (
    AgentManagerEnvironmentCompatibilityMixin,
)


class AgentEnvironmentCompatibilityTests(unittest.TestCase):
    def test_environment_facade_is_outside_agent_manager(self) -> None:
        method_names = {
            "_log_compress",
            "_emit_runtime_event",
            "_audit_runtime_event",
            "_write_skills_snapshot",
            "_has_bootstrap",
            "_get_locale",
            "_resolve_context_budget",
            "_get_state_persist_config",
            "_get_state_path",
            "_resolve_think_level",
            "_ensure_agent_workspace",
        }

        self.assertTrue(
            issubclass(
                AgentManager,
                AgentManagerEnvironmentCompatibilityMixin,
            )
        )
        for method_name in method_names:
            self.assertIn(
                method_name,
                AgentManagerEnvironmentCompatibilityMixin.__dict__,
            )
            self.assertNotIn(method_name, AgentManager.__dict__)

    def test_audit_and_path_methods_keep_agent_module_patch_points(self) -> None:
        audit = Mock()
        resolve_agent_dir = Mock(return_value=Path("/tmp/worker"))
        manager = object.__new__(AgentManager)

        with (
            patch("runtime.agent.audit_logger", audit),
            patch("runtime.agent.resolve_agent_dir", resolve_agent_dir),
        ):
            manager._log_compress("worker", "session-1", 2, 3)
            state_path = manager._get_state_path("worker")

        audit.log_compress.assert_called_once_with(
            "worker",
            "session-1",
            2,
            3,
        )
        resolve_agent_dir.assert_called_once_with("worker")
        self.assertEqual(state_path, Path("/tmp/worker/agent_state.json"))


if __name__ == "__main__":
    unittest.main()
