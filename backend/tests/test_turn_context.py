from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent import (
    AgentManager,
    PromptCacheEntry,
    SessionContextCacheEntry,
)
from runtime.turn_context import (
    PromptCacheEntry as ContextPromptCacheEntry,
    SessionContextCacheEntry as ContextSessionContextCacheEntry,
    TurnContext,
)


class TurnContextTests(unittest.TestCase):
    def test_agent_manager_exposes_context_owned_caches(self) -> None:
        manager = AgentManager()

        self.assertIs(manager._prompt_cache, manager._turn_context.prompt_cache)
        self.assertIs(
            manager._session_context_cache,
            manager._turn_context.session_context_cache,
        )
        self.assertIs(
            PromptCacheEntry,
            ContextPromptCacheEntry,
        )
        self.assertIs(
            SessionContextCacheEntry,
            ContextSessionContextCacheEntry,
        )

    def test_replacing_compatibility_cache_replaces_authoritative_cache(
        self,
    ) -> None:
        manager = AgentManager()
        manager._turn_context = TurnContext(
            max_prompt_entries=2,
            max_session_entries=2,
        )
        prompt_cache = {
            ("prompt-1",): SimpleNamespace(),
            ("prompt-2",): SimpleNamespace(),
            ("prompt-3",): SimpleNamespace(),
        }
        session_cache = {
            ("main", "s1"): SimpleNamespace(),
            ("main", "s2"): SimpleNamespace(),
            ("main", "s3"): SimpleNamespace(),
        }

        manager._prompt_cache = prompt_cache
        manager._session_context_cache = session_cache

        self.assertIs(
            manager._prompt_cache,
            manager._turn_context.prompt_cache,
        )
        self.assertIs(
            manager._session_context_cache,
            manager._turn_context.session_context_cache,
        )
        self.assertEqual(
            list(manager._prompt_cache),
            [("prompt-2",), ("prompt-3",)],
        )
        self.assertEqual(
            list(manager._session_context_cache),
            [("main", "s2"), ("main", "s3")],
        )

    def test_prompt_cache_invalidates_when_runtime_signature_changes(self) -> None:
        manager = AgentManager()
        report = SimpleNamespace(summary=lambda: "")
        build_prompt = Mock(return_value=("prompt", report))

        with (
            patch.object(
                manager,
                "_project_context_signature",
                return_value=(("AGENTS.md", 1.0),),
            ),
            patch.object(
                manager,
                "_prompt_runtime_signature",
                side_effect=[
                    ("model-a", "2026-07-16 10:00"),
                    ("model-b", "2026-07-16 10:00"),
                ],
            ),
            patch(
                "runtime.agent.prompt_builder.build_system_prompt_with_report",
                build_prompt,
            ),
            patch("runtime.agent.count_tokens", return_value=10),
        ):
            manager._get_or_build_prompt(
                agent_id="main",
                prompt_mode="full",
                available_tool_names=["read"],
                extra_system_prompt=None,
                locale="zh-CN",
            )
            manager._get_or_build_prompt(
                agent_id="main",
                prompt_mode="full",
                available_tool_names=["read"],
                extra_system_prompt=None,
                locale="zh-CN",
            )

        self.assertEqual(build_prompt.call_count, 2)

    def test_message_builder_uses_legacy_agent_constructor_bindings(self) -> None:
        manager = AgentManager()
        human = Mock(side_effect=lambda content: ("human", content))
        ai = Mock(side_effect=lambda content: ("ai", content))
        system = Mock(side_effect=lambda content: ("system", content))

        with (
            patch("runtime.agent.HumanMessage", human),
            patch("runtime.agent.AIMessage", ai),
            patch("runtime.agent.SystemMessage", system),
        ):
            messages = manager._build_messages(
                [
                    {"role": "system", "content": "rules"},
                    {"role": "assistant", "content": "answer"},
                ],
                "question",
            )

        self.assertEqual(
            messages,
            [
                ("system", "rules"),
                ("ai", "answer"),
                ("human", "question"),
            ],
        )

    def test_session_cache_invalidates_when_pruning_signature_changes(
        self,
    ) -> None:
        manager = AgentManager()
        manager.mem_stores["main"] = SimpleNamespace(
            get_session_summary=lambda _session_id, _agent_id: None
        )
        load_history = Mock(
            return_value=[{"role": "user", "content": "hello"}]
        )

        with (
            patch.object(manager, "_safe_mtime", return_value=1.0),
            patch.object(
                manager,
                "_pruning_signature",
                side_effect=["budget-a", "budget-a", "budget-b"],
            ),
            patch(
                "runtime.agent.session_manager.load_session_for_agent",
                load_history,
            ),
            patch(
                "runtime.agent.prune_messages",
                side_effect=lambda history, agent_id=None: history,
            ),
            patch("runtime.agent.count_tokens", return_value=0),
            patch("runtime.agent.count_messages_tokens", return_value=1),
        ):
            first = manager._get_or_build_session_context(
                agent_id="main",
                session_id="s1",
            )
            second = manager._get_or_build_session_context(
                agent_id="main",
                session_id="s1",
            )
            third = manager._get_or_build_session_context(
                agent_id="main",
                session_id="s1",
            )

        self.assertIs(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(load_history.call_count, 2)

    def test_prompt_runtime_signature_ignores_unrelated_secret_fields(
        self,
    ) -> None:
        with (
            patch(
                "runtime.turn_context.resolve_agent_config",
                side_effect=[
                    {
                        "model": "provider/model",
                        "thinkingDefault": "off",
                        "user_timezone": "Asia/Shanghai",
                        "custom_api_key": "secret-one",
                    },
                    {
                        "model": "provider/model",
                        "thinkingDefault": "off",
                        "user_timezone": "Asia/Shanghai",
                        "custom_api_key": "secret-two",
                    },
                ],
            ),
            patch(
                "runtime.turn_context.get_heartbeat_config",
                return_value={"prompt": "heartbeat"},
            ),
        ):
            first_signature, _first_minute = TurnContext.prompt_runtime_signature(
                "main"
            )
            second_signature, _second_minute = TurnContext.prompt_runtime_signature(
                "main"
            )

        self.assertEqual(first_signature, second_signature)

    def test_prompt_cache_direct_writes_remain_bounded(self) -> None:
        context = TurnContext(max_prompt_entries=2)

        context.prompt_cache[("prompt-1",)] = SimpleNamespace()
        context.prompt_cache[("prompt-2",)] = SimpleNamespace()
        context.prompt_cache[("prompt-3",)] = SimpleNamespace()

        self.assertEqual(
            list(context.prompt_cache),
            [("prompt-2",), ("prompt-3",)],
        )

    def test_session_cache_direct_writes_remain_bounded(self) -> None:
        context = TurnContext(max_session_entries=2)

        context.session_context_cache[("main", "s1")] = SimpleNamespace()
        context.session_context_cache[("main", "s2")] = SimpleNamespace()
        context.session_context_cache[("main", "s3")] = SimpleNamespace()

        self.assertEqual(
            list(context.session_context_cache),
            [("main", "s2"), ("main", "s3")],
        )

    def test_prompt_cache_evicts_least_recently_used_entry(self) -> None:
        context = TurnContext(max_prompt_entries=2)
        report = SimpleNamespace()

        for index in range(2):
            context.get_or_build_prompt(
                agent_id="main",
                prompt_mode="full",
                available_tool_names=["read"],
                extra_system_prompt=None,
                locale="zh-CN",
                static_signature=(index,),
                runtime_signature=("runtime",),
                build_prompt=lambda _params: (f"prompt-{index}", report),
                count_tokens=lambda _text: 1,
            )

        first_key = next(iter(context.prompt_cache))
        context.prompt_cache.get(first_key)
        context.get_or_build_prompt(
            agent_id="main",
            prompt_mode="full",
            available_tool_names=["read"],
            extra_system_prompt=None,
            locale="zh-CN",
            static_signature=(2,),
            runtime_signature=("runtime",),
            build_prompt=lambda _params: ("prompt-2", report),
            count_tokens=lambda _text: 1,
        )

        self.assertEqual(len(context.prompt_cache), 2)
        cached_prompts = {
            entry.system_prompt
            for entry in context.prompt_cache.values()
        }
        self.assertEqual(cached_prompts, {"prompt-0", "prompt-2"})


if __name__ == "__main__":
    unittest.main()
