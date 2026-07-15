from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class AgentStateTests(unittest.TestCase):
    def test_agent_module_reexports_extracted_state_type(self) -> None:
        spec = importlib.util.find_spec("runtime.agent_state")
        self.assertIsNotNone(spec, "runtime.agent_state should be defined")

        state_module = importlib.import_module("runtime.agent_state")
        agent_module = importlib.import_module("runtime.agent")

        self.assertIs(agent_module.AgentState, state_module.AgentState)

    def test_extracted_state_keeps_legacy_logger_name(self) -> None:
        state_module = importlib.import_module("runtime.agent_state")

        self.assertEqual(state_module.logger.name, "runtime.agent")

    def test_state_round_trips_through_disk_with_requested_agent_id(self) -> None:
        state_module = importlib.import_module("runtime.agent_state")
        state = state_module.AgentState(
            agent_id="source",
            compaction_count=2,
            total_input_tokens=11,
            total_output_tokens=7,
            total_turns=3,
            think_level=1,
            verbose=True,
            reasoning=True,
            last_active=123.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state.save_to_disk(path)
            loaded = state_module.AgentState.load_from_disk(path, "target")

        self.assertEqual(loaded.agent_id, "target")
        self.assertEqual(loaded.compaction_count, 2)
        self.assertEqual(loaded.total_input_tokens, 11)
        self.assertEqual(loaded.total_output_tokens, 7)
        self.assertEqual(loaded.total_turns, 3)
        self.assertEqual(loaded.think_level, 1)
        self.assertTrue(loaded.verbose)
        self.assertTrue(loaded.reasoning)
        self.assertEqual(loaded.last_active, 123.0)

    def test_invalid_state_file_falls_back_to_requested_agent(self) -> None:
        state_module = importlib.import_module("runtime.agent_state")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{invalid", encoding="utf-8")
            loaded = state_module.AgentState.load_from_disk(path, "main")

        self.assertEqual(loaded, state_module.AgentState(agent_id="main"))

    def test_state_save_failure_is_non_fatal(self) -> None:
        state_module = importlib.import_module("runtime.agent_state")
        state = state_module.AgentState(agent_id="main")

        with tempfile.TemporaryDirectory() as tmp:
            directory_path = Path(tmp)
            state.save_to_disk(directory_path)

    def test_record_turn_updates_counters_and_activity(self) -> None:
        state_module = importlib.import_module("runtime.agent_state")
        state = state_module.AgentState(agent_id="main")

        state.record_turn(input_tokens=5, output_tokens=8)

        self.assertEqual(state.total_input_tokens, 5)
        self.assertEqual(state.total_output_tokens, 8)
        self.assertEqual(state.total_turns, 1)
        self.assertGreater(state.last_active, 0)


if __name__ == "__main__":
    unittest.main()
