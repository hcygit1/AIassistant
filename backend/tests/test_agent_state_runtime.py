from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent_state import AgentState
from runtime.agent_state_runtime import AgentStateRuntime


class AgentStateRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_creates_default_state_when_persistence_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "agent_state.json"
            AgentState(
                agent_id="source",
                total_turns=4,
                think_level=2,
            ).save_to_disk(state_path)
            runtime = AgentStateRuntime(
                resolve_persist_config=lambda _agent_id: (False, 5),
                resolve_state_path=lambda _agent_id: state_path,
                resolve_think_level=lambda _agent_id: 1,
                is_initialized=lambda: True,
            )

            runtime.initialize_agent("main")

            self.assertEqual(runtime.states["main"].total_turns, 0)
            self.assertEqual(runtime.states["main"].agent_id, "main")
            self.assertEqual(runtime.states["main"].think_level, 1)
            self.assertEqual(runtime.save_tasks, {})

    async def test_loads_persisted_state_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "agent_state.json"
            AgentState(
                agent_id="source",
                total_turns=4,
                think_level=2,
            ).save_to_disk(state_path)
            runtime = AgentStateRuntime(
                resolve_persist_config=lambda _agent_id: (True, 5),
                resolve_state_path=lambda _agent_id: state_path,
                resolve_think_level=lambda _agent_id: 1,
                is_initialized=lambda: False,
            )

            runtime.initialize_agent("main")
            await runtime.stop_periodic_saves()

            self.assertEqual(runtime.states["main"].total_turns, 4)
            self.assertEqual(runtime.states["main"].agent_id, "main")

    async def test_save_all_states_respects_persistence_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "enabled": Path(tmpdir) / "enabled.json",
                "disabled": Path(tmpdir) / "disabled.json",
            }
            runtime = AgentStateRuntime(
                resolve_persist_config=lambda agent_id: (
                    agent_id == "enabled",
                    5,
                ),
                resolve_state_path=lambda agent_id: paths[agent_id],
                resolve_think_level=lambda _agent_id: 0,
                is_initialized=lambda: False,
            )
            runtime.states["enabled"] = AgentState(
                agent_id="enabled",
                total_turns=2,
            )
            runtime.states["disabled"] = AgentState(
                agent_id="disabled",
                total_turns=3,
            )

            await runtime.save_all_states()

            self.assertTrue(paths["enabled"].exists())
            self.assertFalse(paths["disabled"].exists())


if __name__ == "__main__":
    unittest.main()
