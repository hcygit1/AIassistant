import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent_runtime_assembly import AgentRuntimeComponents


class AgentRuntimeComponentsTests(unittest.TestCase):
    def test_install_on_exposes_all_owned_runtime_components(self):
        values = {
            name: object()
            for name in (
                "state_runtime",
                "memory_runtime",
                "model_runtime",
                "subagent_service",
                "tool_registry",
                "turn_context",
                "turn_preparation",
                "turn_preparation_adapter",
                "session_compactor",
                "session_commands",
                "turn_recovery",
                "turn_executor",
                "turn_service",
                "session_lifecycle",
                "tool_name_cache",
                "lifecycle",
            )
        }
        pending_tasks = set()
        components = AgentRuntimeComponents(
            **values,
            pending_tasks=pending_tasks,
        )
        manager = SimpleNamespace()

        components.install_on(manager)

        attribute_names = {
            name: f"_{name}"
            for name in values
        }
        attribute_names["subagent_service"] = "subagent_service"
        for name, value in values.items():
            self.assertIs(getattr(manager, attribute_names[name]), value)
        self.assertIs(manager._pending_tasks, pending_tasks)


class AgentManagerFacadeStructureTests(unittest.TestCase):
    def test_turn_preparation_facade_is_outside_agent_manager(self):
        from runtime.agent import AgentManager
        from runtime.agent_turn_compat import (
            AgentManagerTurnPreparationCompatibilityMixin,
        )

        facade_methods = {
            "_collect_tools",
            "_wrap_tools_for_session",
            "_build_tools",
            "_resolve_tool_policy",
            "_filter_tools_by_policy",
            "_build_messages",
            "_safe_mtime",
            "_project_context_signature",
            "_prompt_runtime_signature",
            "_pruning_signature",
            "_tool_policy_signature",
            "_get_or_build_tool_names",
            "_get_or_build_prompt",
            "_session_summary_fingerprint",
            "_get_or_build_session_context",
        }

        self.assertTrue(
            issubclass(
                AgentManager,
                AgentManagerTurnPreparationCompatibilityMixin,
            )
        )
        for method_name in facade_methods:
            self.assertIn(
                method_name,
                AgentManagerTurnPreparationCompatibilityMixin.__dict__,
            )
            self.assertNotIn(method_name, AgentManager.__dict__)


if __name__ == "__main__":
    unittest.main()
