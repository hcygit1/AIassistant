from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent_runtime_bindings import AgentRuntimeBindings


class AgentRuntimeBindingsTests(unittest.TestCase):
    def test_default_bindings_resolve_module_symbols_dynamically(self) -> None:
        module_symbols = {
            "count_tokens": lambda _text: 1,
            "prompt_builder": SimpleNamespace(
                format_session_summary=lambda _summary: "first"
            ),
            "run_tracker": "tracker-1",
            "session_manager": "sessions-1",
            "detect_compaction_level": Mock(return_value="initial"),
        }
        bindings = AgentRuntimeBindings.from_module_symbols(module_symbols)
        module_symbols["count_tokens"] = lambda _text: 2
        module_symbols["prompt_builder"] = SimpleNamespace(
            format_session_summary=lambda _summary: "second"
        )
        module_symbols["run_tracker"] = "tracker-2"
        module_symbols["session_manager"] = "sessions-2"

        self.assertEqual(bindings.count_tokens("hello"), 2)
        self.assertEqual(bindings.format_session_summary("summary"), "second")
        self.assertEqual(bindings.get_run_tracker(), "tracker-2")
        self.assertEqual(bindings.get_session_manager(), "sessions-2")

    def test_compaction_detector_keeps_construction_time_binding(self) -> None:
        initial = Mock(return_value="initial")
        replacement = Mock(return_value="replacement")
        module_symbols = {"detect_compaction_level": initial}
        bindings = AgentRuntimeBindings.from_module_symbols(module_symbols)
        module_symbols["detect_compaction_level"] = replacement

        result = bindings.detect_compaction_level("sliding")

        self.assertEqual(result, "initial")
        initial.assert_called_once_with("sliding")
        replacement.assert_not_called()

    def test_assembler_has_no_module_globals_lookup(self) -> None:
        source = (
            BACKEND_DIR / "runtime" / "agent_runtime_assembly.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("module_globals", source)
        self.assertNotIn("self._global(", source)
        self.assertIn("self._bindings", source)

    def test_agent_manager_forwards_explicit_runtime_bindings(self) -> None:
        from runtime.agent import AgentManager

        bindings = Mock(spec=AgentRuntimeBindings)
        components = Mock()
        assembler = Mock()
        assembler.build.return_value = components

        with patch(
            "runtime.agent.AgentRuntimeAssembler",
            return_value=assembler,
        ) as assembler_type:
            manager = AgentManager(runtime_bindings=bindings)

        args, kwargs = assembler_type.call_args
        self.assertIs(args[0], manager)
        self.assertIs(args[1], bindings)
        self.assertEqual(set(kwargs), {"get_session_manager"})
        assembler.build.assert_called_once_with()
        components.install_on.assert_called_once_with(manager)


if __name__ == "__main__":
    unittest.main()
