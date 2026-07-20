from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent_lifecycle import AgentLifecycle
from runtime.agent import AgentManager


class _FakeRuntime:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name
        self.states: dict[str, object] = {}
        self.save_tasks: dict[str, asyncio.Task] = {}

    def initialize_agent(self, agent_id: str) -> None:
        self.events.append(f"{self.name}.initialize:{agent_id}")

    async def save_all_states(self) -> None:
        self.events.append("state.save")

    async def stop_periodic_saves(self) -> None:
        self.events.append("state.stop")

    def clear(self) -> None:
        self.events.append(f"{self.name}.clear")

    def close(self) -> None:
        self.clear()


class AgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_prepares_each_agent_before_marking_ready(self) -> None:
        events: list[str] = []
        state_runtime = _FakeRuntime(events, "state")
        memory_runtime = _FakeRuntime(events, "memory")
        model_runtime = _FakeRuntime(events, "model")
        lifecycle = AgentLifecycle(
            state_runtime=state_runtime,
            memory_runtime=memory_runtime,
            model_runtime=model_runtime,
            list_agents=lambda: [{"id": "main"}, {"id": "writer"}],
            ensure_workspace=lambda agent_id: events.append(
                f"workspace:{agent_id}"
            ),
            prompt_cache={},
            session_context_cache={},
            tool_name_cache={},
        )

        await lifecycle.initialize("/tmp/pipixia")

        self.assertEqual(lifecycle.data_dir, "/tmp/pipixia")
        self.assertTrue(lifecycle.initialized)
        self.assertEqual(
            events,
            [
                "workspace:main",
                "memory.initialize:main",
                "state.initialize:main",
                "workspace:writer",
                "memory.initialize:writer",
                "state.initialize:writer",
            ],
        )

    async def test_register_agent_uses_full_runtime_initialization(self) -> None:
        events: list[str] = []
        lifecycle = AgentLifecycle(
            state_runtime=_FakeRuntime(events, "state"),
            memory_runtime=_FakeRuntime(events, "memory"),
            model_runtime=_FakeRuntime(events, "model"),
            list_agents=lambda: [],
            ensure_workspace=lambda agent_id: events.append(
                f"workspace:{agent_id}"
            ),
            prompt_cache={},
            session_context_cache={},
            tool_name_cache={},
        )

        await lifecycle.register_agent("writer")

        self.assertEqual(
            events,
            [
                "workspace:writer",
                "memory.initialize:writer",
                "state.initialize:writer",
            ],
        )

    async def test_manager_register_agent_delegates_to_lifecycle(self) -> None:
        manager = AgentManager()
        register_agent = AsyncMock()
        manager._lifecycle.register_agent = register_agent

        with (
            patch("runtime.workspace.ensure_agent_workspace"),
            patch.object(manager, "_init_mem_system"),
        ):
            await manager.register_agent("writer")

        register_agent.assert_awaited_once_with("writer")

    async def test_close_waits_and_clears_runtime_resources(self) -> None:
        events: list[str] = []
        state_runtime = _FakeRuntime(events, "state")
        memory_runtime = _FakeRuntime(events, "memory")
        model_runtime = _FakeRuntime(events, "model")
        prompt_cache = {"prompt": object()}
        session_context_cache = {"session": object()}
        tool_name_cache = {"tool": object()}
        lifecycle = AgentLifecycle(
            state_runtime=state_runtime,
            memory_runtime=memory_runtime,
            model_runtime=model_runtime,
            list_agents=lambda: [],
            ensure_workspace=Mock(),
            prompt_cache=prompt_cache,
            session_context_cache=session_context_cache,
            tool_name_cache=tool_name_cache,
        )
        lifecycle.initialized = True
        pending_finished = asyncio.Event()

        async def finish_pending() -> None:
            pending_finished.set()

        pending_tasks = {asyncio.create_task(finish_pending())}

        await lifecycle.close(timeout=1, pending_tasks=pending_tasks)

        self.assertFalse(lifecycle.initialized)
        self.assertEqual(lifecycle.data_dir, "")
        self.assertTrue(pending_finished.is_set())
        self.assertEqual(pending_tasks, set())
        self.assertEqual(prompt_cache, {})
        self.assertEqual(session_context_cache, {})
        self.assertEqual(tool_name_cache, {})
        self.assertEqual(
            events,
            [
                "state.save",
                "state.stop",
                "memory.clear",
                "model.clear",
                "state.clear",
            ],
        )


if __name__ == "__main__":
    unittest.main()
