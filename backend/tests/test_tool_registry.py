from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import is_tool_allowed_by_policy, resolve_tool_policy
from runtime.agent import AgentManager
from runtime.tool_registry import ToolRegistry
from tools.runtime_dependencies import ToolRuntimeDependencies


class ToolPolicyTests(unittest.TestCase):
    def test_resolve_policy_uses_agent_override_before_defaults(self) -> None:
        config = {
            "agents": {
                "defaults": {
                    "tools": {
                        "allow": ["read"],
                        "deny": ["exec"],
                    }
                },
                "list": [
                    {
                        "id": "main",
                        "tools": {
                            "allow": ["read", "grep"],
                            "deny": ["write"],
                        },
                    }
                ],
            }
        }

        with patch("config.get_config", return_value=config):
            allow, deny = resolve_tool_policy("main")

        self.assertEqual(allow, ["read", "grep"])
        self.assertEqual(deny, ["write"])

    def test_policy_preserves_apply_patch_exec_alias(self) -> None:
        config = {
            "agents": {
                "defaults": {"tools": {"allow": ["exec"]}},
                "list": [{"id": "main"}],
            }
        }

        with patch("config.get_config", return_value=config):
            self.assertTrue(is_tool_allowed_by_policy("main", "apply-patch"))

    def test_exec_deny_also_denies_apply_patch_alias(self) -> None:
        config = {
            "agents": {
                "defaults": {"tools": {"allow": ["exec"]}},
                "list": [
                    {
                        "id": "main",
                        "tools": {"deny": ["exec"]},
                    }
                ],
            }
        }

        with patch("config.get_config", return_value=config):
            self.assertFalse(is_tool_allowed_by_policy("main", "apply_patch"))


class ToolRegistryTests(unittest.TestCase):
    def test_registry_delegates_to_shared_tool_collection(self) -> None:
        service = object()
        dependencies = ToolRuntimeDependencies(
            get_session_manager=Mock(),
            get_memory_recall=Mock(),
            get_memory_store=Mock(),
            count_active_for_requester=Mock(),
        )
        registry = ToolRegistry(
            subagent_service=service,
            runtime_dependencies=dependencies,
        )
        expected = [SimpleNamespace(name="read")]

        with patch("tools.get_all_tools", return_value=expected) as collect:
            result = registry.collect_tools("main", "session-1")

        self.assertIs(result, expected)
        collect.assert_called_once_with(
            "main",
            subagent_service=service,
            session_id="session-1",
            runtime_dependencies=dependencies,
        )

    def test_registry_filters_tools_with_shared_policy(self) -> None:
        registry = ToolRegistry(subagent_service=object())
        tools = [
            SimpleNamespace(name="read"),
            SimpleNamespace(name="write"),
            SimpleNamespace(name="exec"),
            SimpleNamespace(name="apply_patch"),
        ]
        config = {
            "agents": {
                "defaults": {"tools": {"allow": ["read", "exec"]}},
                "list": [{"id": "main", "tools": {"deny": ["write"]}}],
            }
        }

        with patch("config.get_config", return_value=config):
            filtered = registry.filter_tools("main", tools)

        self.assertEqual(
            [tool.name for tool in filtered],
            ["read", "exec", "apply_patch"],
        )

    def test_tool_name_cache_uses_supplied_compatibility_callbacks(self) -> None:
        registry = ToolRegistry(subagent_service=object())
        collect = Mock(
            return_value=[
                SimpleNamespace(name="read"),
                SimpleNamespace(name="exec"),
            ]
        )
        filter_tools = Mock(side_effect=lambda _agent_id, tools: tools[:1])

        with patch(
            "config.get_config",
            return_value={
                "agents": {
                    "defaults": {"tools": {"allow": ["read"]}},
                    "list": [{"id": "main"}],
                }
            },
        ):
            first = registry.get_or_build_tool_names(
                "main",
                collect_tools=collect,
                filter_tools=filter_tools,
            )
            second = registry.get_or_build_tool_names(
                "main",
                collect_tools=collect,
                filter_tools=filter_tools,
            )

        self.assertEqual(first, ("read",))
        self.assertEqual(second, ("read",))
        collect.assert_called_once_with("main", "")
        filter_tools.assert_called_once()

    def test_agent_manager_exposes_registry_cache_for_compatibility(self) -> None:
        manager = AgentManager()

        self.assertIs(manager._tool_name_cache, manager._tool_registry.name_cache)

    def test_agent_manager_collection_uses_injected_session_manager(
        self,
    ) -> None:
        session_manager = Mock()
        session_manager.list_sessions.return_value = []
        session_manager.session_key_from_session_id.return_value = (
            "agent:main:session-1"
        )
        manager = AgentManager(session_manager=session_manager)

        tools = {
            tool.name: tool
            for tool in manager.collect_tools("main", "session-1")
        }
        tools["sessions_list"]._run()
        tools["session_status"]._run()

        session_manager.list_sessions.assert_called_once_with(
            "main",
            spawned_by_session_key=None,
        )
        session_manager.session_key_from_session_id.assert_called_once_with(
            "main",
            "session-1",
        )

    def test_agent_manager_policy_override_remains_effective(self) -> None:
        manager = AgentManager()
        tools = [
            SimpleNamespace(name="read"),
            SimpleNamespace(name="exec"),
        ]

        with patch.object(
            manager,
            "_resolve_tool_policy",
            return_value=(["read"], []),
        ):
            filtered = manager._filter_tools_by_policy("main", tools)

        self.assertEqual([tool.name for tool in filtered], ["read"])

    def test_agent_manager_wrap_override_remains_effective(self) -> None:
        manager = AgentManager()
        wrapped = [SimpleNamespace(name="wrapped")]

        with (
            patch.object(manager, "_collect_tools", return_value=[]),
            patch.object(manager, "_filter_tools_by_policy", return_value=[]),
            patch.object(
                manager,
                "_wrap_tools_for_session",
                return_value=wrapped,
            ) as wrap,
        ):
            result = manager._build_tools("main", "s1")

        self.assertIs(result, wrapped)
        wrap.assert_called_once_with("main", "s1", [])

    def test_tool_name_cache_invalidates_when_tool_config_changes(self) -> None:
        registry = ToolRegistry(subagent_service=object())
        read = SimpleNamespace(name="read")
        apply_patch_tool = SimpleNamespace(name="apply_patch")
        collect = Mock(side_effect=[[read], [read, apply_patch_tool]])
        filter_tools = Mock(side_effect=lambda _agent_id, tools: tools)
        config = {
            "tools": {
                "exec": {
                    "apply_patch": {
                        "enabled": False,
                    }
                }
            },
            "agents": {
                "defaults": {"tools": {}},
                "list": [{"id": "main"}],
            },
        }

        with patch("config.get_config", return_value=config):
            first = registry.get_or_build_tool_names(
                "main",
                collect_tools=collect,
                filter_tools=filter_tools,
            )
            config["tools"]["exec"]["apply_patch"]["enabled"] = True
            second = registry.get_or_build_tool_names(
                "main",
                collect_tools=collect,
                filter_tools=filter_tools,
            )

        self.assertEqual(first, ("read",))
        self.assertEqual(second, ("apply_patch", "read"))
        self.assertEqual(collect.call_count, 2)


if __name__ == "__main__":
    unittest.main()
