from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.turn_service import TurnService, TurnServicePorts
from runtime.agent import AgentManager
from llm.models_config import ModelRef


async def _empty_stream(*_args, **_kwargs):
    if False:
        yield {}


class TurnServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_manager_ports_resolve_overrides_lazily(
        self,
    ) -> None:
        with (
            unittest.mock.patch.object(
                AgentManager,
                "_write_skills_snapshot",
            ),
            unittest.mock.patch.object(
                AgentManager,
                "_has_bootstrap",
                return_value=False,
            ),
            unittest.mock.patch.object(
                AgentManager,
                "_get_locale",
                return_value="zh-CN",
            ),
            unittest.mock.patch.object(
                AgentManager,
                "_resolve_context_budget",
                return_value=SimpleNamespace(
                    active_tokens=1000
                ),
            ),
        ):
            manager = AgentManager()

        write_snapshot = Mock()
        has_bootstrap = Mock(return_value=True)
        get_locale = Mock(return_value="en-US")
        resolve_budget = Mock(
            return_value=SimpleNamespace(
                active_tokens=2000
            )
        )
        manager._write_skills_snapshot = write_snapshot
        manager._has_bootstrap = has_bootstrap
        manager._get_locale = get_locale
        manager._resolve_context_budget = resolve_budget
        get_model_override = Mock(
            return_value=ModelRef(
                provider="fake",
                model="runtime",
            )
        )
        manager.get_model_override = get_model_override
        ports = manager._turn_service._ports

        ports.write_skills_snapshot("main")
        self.assertTrue(ports.has_bootstrap("main"))
        self.assertEqual(ports.get_locale(), "en-US")
        self.assertEqual(
            ports.resolve_budget("main").active_tokens,
            2000,
        )
        self.assertEqual(
            ports.get_model_override("main"),
            ModelRef(
                provider="fake",
                model="runtime",
            ),
        )

        write_snapshot.assert_called_once_with("main")
        has_bootstrap.assert_called_once_with("main")
        get_locale.assert_called_once_with()
        resolve_budget.assert_called_once_with("main")
        get_model_override.assert_called_once_with("main")

    async def test_agent_manager_command_port_forwards_keywords(
        self,
    ) -> None:
        manager = AgentManager()
        command_result = {
            "handled": True,
            "action": "setting",
            "response": "ok",
        }
        execute = AsyncMock(return_value=command_result)
        switch_model = Mock()

        with patch(
            "runtime.agent.execute_command",
            new=execute,
        ):
            result = await (
                manager._turn_service._ports.execute_command(
                    "parsed",
                    "main",
                    "s1",
                    "state",
                    switch_model=switch_model,
                )
            )

        self.assertEqual(result, command_result)
        execute.assert_awaited_once_with(
            "parsed",
            "main",
            "s1",
            "state",
            switch_model=switch_model,
        )

    async def test_stream_prepares_and_executes_normal_turn(
        self,
    ) -> None:
        state = SimpleNamespace()
        captured: dict[str, object] = {}
        prompt_report = SimpleNamespace(
            summary=lambda: "prompt report"
        )
        context = SimpleNamespace(
            pruned_history=[
                {"role": "user", "content": "history"}
            ],
            summary_tokens=7,
            history_tokens=11,
        )

        async def _execute_turn(request):
            captured["request"] = request
            yield {
                "type": "done",
                "content": "answer",
                "session_id": request.session_id,
            }

        async def _run_fallback(
            _candidates,
            run_model,
            _agent_id,
        ):
            async for event in run_model("fake", "model"):
                yield event

        async def _recover_turn(**kwargs):
            captured["recovery"] = kwargs
            async for event in kwargs["stream"]():
                yield event

        ports = TurnServicePorts(
            get_state=Mock(return_value=state),
            parse_command=Mock(return_value=None),
            execute_command=AsyncMock(),
            switch_model=Mock(),
            get_current_model=Mock(
                return_value=ModelRef(
                    provider="fake",
                    model="runtime",
                )
            ),
            get_model_override=Mock(
                return_value=ModelRef(
                    provider="fake",
                    model="runtime",
                )
            ),
            handle_reset=_empty_stream,
            handle_reset_noflush=_empty_stream,
            handle_compact=_empty_stream,
            write_skills_snapshot=Mock(),
            has_bootstrap=Mock(return_value=False),
            resolve_workspace=Mock(),
            get_locale=Mock(return_value="zh-CN"),
            get_tool_names=Mock(
                return_value=("read", "exec")
            ),
            build_prompt=Mock(
                return_value=(
                    "system prompt",
                    prompt_report,
                    13,
                )
            ),
            get_session_context=Mock(
                return_value=context
            ),
            build_tools=Mock(return_value=["tool"]),
            resolve_budget=Mock(
                return_value=SimpleNamespace(
                    active_tokens=1000
                )
            ),
            resolve_agent_config=Mock(
                return_value={"recursion_limit": 25}
            ),
            resolve_candidates=Mock(
                return_value=["candidate"]
            ),
            execute_turn=_execute_turn,
            run_fallback_stream=_run_fallback,
            recover_turn=_recover_turn,
        )
        service = TurnService(ports)

        events = [
            event
            async for event in service.stream(
                "question",
                "s1",
                agent_id="main",
                prompt_mode="minimal",
                persist_input_role="system",
            )
        ]

        self.assertEqual(events[-1]["content"], "answer")
        request = captured["request"]
        self.assertEqual(request.state, state)
        self.assertEqual(request.message, "question")
        self.assertEqual(request.persist_input_role, "system")
        self.assertEqual(request.system_prompt, "system prompt")
        self.assertEqual(request.history, context.pruned_history)
        self.assertEqual(request.tools, ["tool"])
        self.assertEqual(request.recursion_limit, 25)
        self.assertEqual(request.prompt_tokens, 13)
        self.assertEqual(request.summary_tokens, 7)
        self.assertEqual(request.history_tokens, 11)
        self.assertEqual(request.active_tokens, 1000)
        self.assertEqual(
            captured["recovery"]["state"],
            state,
        )
        ports.write_skills_snapshot.assert_called_once_with(
            "main"
        )
        self.assertIn(
            "fake/runtime",
            ports.build_prompt.call_args.kwargs[
                "extra_system_prompt"
            ],
        )

    async def test_stream_returns_handled_command_without_turn(
        self,
    ) -> None:
        state = SimpleNamespace()
        execute_turn = Mock()
        ports = TurnServicePorts(
            get_state=Mock(return_value=state),
            parse_command=Mock(return_value={"name": "stop"}),
            execute_command=AsyncMock(
                return_value={
                    "handled": True,
                    "action": "stop",
                    "response": "已停止",
                }
            ),
            switch_model=Mock(),
            get_current_model=Mock(),
            get_model_override=Mock(return_value=None),
            handle_reset=_empty_stream,
            handle_reset_noflush=_empty_stream,
            handle_compact=_empty_stream,
            write_skills_snapshot=Mock(),
            has_bootstrap=Mock(return_value=False),
            resolve_workspace=Mock(),
            get_locale=Mock(return_value="zh-CN"),
            get_tool_names=Mock(),
            build_prompt=Mock(),
            get_session_context=Mock(),
            build_tools=Mock(),
            resolve_budget=Mock(),
            resolve_agent_config=Mock(),
            resolve_candidates=Mock(),
            execute_turn=execute_turn,
            run_fallback_stream=_empty_stream,
            recover_turn=_empty_stream,
        )
        service = TurnService(ports)

        events = [
            event
            async for event in service.stream(
                "/stop",
                "s1",
                agent_id="main",
            )
        ]

        self.assertEqual(
            events,
            [
                {
                    "type": "command_response",
                    "response": "已停止",
                },
                {
                    "type": "done",
                    "content": "已停止",
                    "session_id": "s1",
                },
            ],
        )
        execute_turn.assert_not_called()
        ports.write_skills_snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
