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


if __name__ == "__main__":
    unittest.main()
