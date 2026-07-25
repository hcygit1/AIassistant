from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tools.agent_tools import SessionsListTool, get_agent_tools
from tools.memory_tools import MemSearchTool, get_memory_tools
from tools.runtime_dependencies import (
    ToolRuntimeDependencies,
    default_tool_runtime_dependencies,
)
from tools.status_tool import SessionStatusTool, get_status_tools


class ToolRuntimeDependencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session_manager = Mock()
        self.session_manager.resolve_main_session_id.return_value = "main-session"
        self.session_manager.session_key_from_session_id.return_value = (
            "agent:main:main-session"
        )
        self.session_manager.list_sessions.return_value = [
            {
                "session_id": "main-session",
                "title": "Main",
                "message_count": 2,
            }
        ]
        self.session_manager.load_session.return_value = {
            "messages": [{"role": "user", "content": "hello"}]
        }
        self.recall = Mock()
        self.recall.search = AsyncMock(
            return_value=SimpleNamespace(hits=[])
        )
        self.store = Mock()
        self.store.get_chunk.return_value = None
        self.get_session_manager = Mock(
            return_value=self.session_manager
        )
        self.get_memory_recall = Mock(return_value=self.recall)
        self.get_memory_store = Mock(return_value=self.store)
        self.count_active = Mock(return_value=2)
        self.dependencies = ToolRuntimeDependencies(
            get_session_manager=self.get_session_manager,
            get_memory_recall=self.get_memory_recall,
            get_memory_store=self.get_memory_store,
            count_active_for_requester=self.count_active,
        )

    async def test_session_and_status_tools_use_injected_dependencies(
        self,
    ) -> None:
        agent_tools = {
            tool.name: tool
            for tool in get_agent_tools(
                "main",
                subagent_service=Mock(),
                runtime_dependencies=self.dependencies,
            )
        }
        status_tool = get_status_tools(
            "main",
            runtime_dependencies=self.dependencies,
        )[0]

        sessions = agent_tools["sessions_list"]._run()
        history = agent_tools["sessions_history"]._run()
        sent = agent_tools["sessions_send"]._run(message="next")
        status = status_tool._run()

        self.assertIn("main-session", sessions)
        self.assertIn("[user] hello", history)
        self.assertIn("agent:main:main-session", sent)
        self.assertIn("活跃子 Agent: 2", status)
        self.session_manager.list_sessions.assert_called_once_with(
            "main",
            spawned_by_session_key=None,
        )
        self.session_manager.load_session.assert_called_once_with(
            "main-session",
            "main",
        )
        self.session_manager.save_message.assert_called_once_with(
            "main-session",
            "main",
            "user",
            "next",
        )
        self.count_active.assert_called_once_with(
            "agent:main:main-session"
        )

    async def test_memory_tools_use_injected_runtime_stores(self) -> None:
        tools = {
            tool.name: tool
            for tool in get_memory_tools(
                agent_id="main",
                runtime_dependencies=self.dependencies,
            )
        }

        search_result = await tools["memory_search"]._arun("decision")
        get_result = tools["memory_get"]._run("chunk-1")

        self.assertIn("未找到", search_result)
        self.assertEqual(get_result, "未找到 chunk: chunk-1")
        self.get_memory_recall.assert_called_once_with("main")
        self.get_memory_store.assert_called_once_with("main")

    async def test_default_session_dependency_is_resolved_lazily(
        self,
    ) -> None:
        dependencies = default_tool_runtime_dependencies()
        replacement = object()

        with patch(
            "sessions.session_manager.session_manager",
            replacement,
        ):
            resolved = dependencies.get_session_manager()

        self.assertIs(resolved, replacement)

    async def test_direct_session_tool_keeps_global_fallback(self) -> None:
        with patch(
            "sessions.session_manager.session_manager",
            self.session_manager,
        ):
            result = SessionsListTool(current_agent_id="main")._run()

        self.assertIn("main-session", result)

    async def test_direct_memory_tool_keeps_global_fallback(self) -> None:
        manager = SimpleNamespace(mem_recalls={"main": self.recall})

        with patch("runtime.agent.agent_manager", manager):
            result = await MemSearchTool(agent_id="main")._arun(
                "decision"
            )

        self.assertIn("未找到", result)

    async def test_direct_status_tool_keeps_global_fallback(self) -> None:
        registry = Mock()
        registry.count_active_for_requester.return_value = 2

        with (
            patch(
                "sessions.session_manager.session_manager",
                self.session_manager,
            ),
            patch("subagents.subagent_registry.registry", registry),
        ):
            result = SessionStatusTool(agent_id="main")._run()

        self.assertIn("活跃子 Agent: 2", result)


if __name__ == "__main__":
    unittest.main()
