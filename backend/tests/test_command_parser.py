from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent import _should_persist_input_message
from runtime.command_parser import execute_command, format_help, parse_command
from runtime.user_turn_stream import _should_skip_auto_title
from llm.models_config import ModelRef


class CommandParserTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_known_command(self) -> None:
        parsed = parse_command("  /model openai/gpt-4o  ")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.command, "/model")
        self.assertEqual(parsed.args, ["openai/gpt-4o"])

    async def test_unknown_slash_command_returns_explicit_error(self) -> None:
        parsed = parse_command("/stop")

        self.assertIsNotNone(parsed)
        assert parsed is not None

        result = await execute_command(parsed, "main", "main-main")
        self.assertTrue(result["handled"])
        self.assertEqual(result["action"], "info")
        self.assertIn("未知命令", result["response"])

    async def test_model_switch_uses_injected_callback(
        self,
    ) -> None:
        parsed = parse_command("/model fake/new-model")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        switch_model = Mock(return_value="New Model")

        with (
            patch(
                "llm.model_selection.resolve_agent_model",
                return_value=ModelRef(
                    provider="fake",
                    model="old-model",
                ),
            ),
            patch(
                "llm.model_selection.get_model_display_name",
                return_value="Old Model",
            ),
        ):
            result = await execute_command(
                parsed,
                "main",
                "main-main",
                switch_model=switch_model,
            )

        switch_model.assert_called_once_with(
            "main",
            "fake/new-model",
        )
        self.assertEqual(result["action"], "setting")
        self.assertIn("New Model", result["response"])

    async def test_model_list_uses_injected_current_model(
        self,
    ) -> None:
        parsed = parse_command("/model")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        current = ModelRef(
            provider="fake",
            model="runtime-model",
        )

        with (
            patch(
                "llm.model_selection.resolve_agent_model",
                return_value=ModelRef(
                    provider="fake",
                    model="configured-model",
                ),
            ),
            patch(
                "llm.model_selection.get_model_display_name",
                side_effect=lambda ref: ref.model,
            ),
            patch(
                "llm.models_config.models_config.list_all_models",
                return_value=[
                    SimpleNamespace(
                        provider="fake",
                        id="runtime-model",
                        name="Runtime Model",
                        reasoning=False,
                        input=["text"],
                    )
                ],
            ),
        ):
            result = await execute_command(
                parsed,
                "main",
                "main-main",
                get_current_model=lambda _agent_id: current,
            )

        self.assertIn(
            "`fake/runtime-model`",
            result["response"],
        )
        self.assertIn("**<-**", result["response"])

    async def test_usage_uses_injected_current_model(
        self,
    ) -> None:
        parsed = parse_command("/usage")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        current = ModelRef(
            provider="fake",
            model="runtime",
        )

        with (
            patch(
                "infra.run_tracker.run_tracker.get_cumulative_usage",
                return_value={
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "cache_read_tokens": 3,
                    "total_tokens": 6,
                    "turns": 1,
                },
            ),
            patch(
                "llm.model_selection.get_model_display_name",
                return_value="Runtime Model",
            ),
        ):
            result = await execute_command(
                parsed,
                "main",
                "main-main",
                get_current_model=lambda _agent_id: current,
            )

        self.assertIn("`fake/runtime`", result["response"])

    async def test_whoami_uses_injected_current_model(
        self,
    ) -> None:
        parsed = parse_command("/whoami")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        current = ModelRef(
            provider="fake",
            model="runtime",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with (
                patch(
                    "config.resolve_agent_config",
                    return_value={"name": "Main"},
                ),
                patch(
                    "config.resolve_agent_workspace",
                    return_value=workspace,
                ),
                patch(
                    "llm.model_selection.get_model_display_name",
                    return_value="Runtime Model",
                ),
                patch(
                    "llm.models_config.models_config.resolve_api_protocol",
                    return_value="openai-completions",
                ),
            ):
                result = await execute_command(
                    parsed,
                    "main",
                    "main-main",
                    get_current_model=(
                        lambda _agent_id: current
                    ),
                )

        self.assertIn("`fake/runtime`", result["response"])

    def test_absolute_path_is_not_treated_as_command(self) -> None:
        self.assertIsNone(parse_command("/etc/hosts"))

    def test_removed_pseudo_commands_are_not_listed_in_help(self) -> None:
        help_text = format_help("zh-CN")

        self.assertNotIn("`/stop`", help_text)
        self.assertNotIn("`/think`", help_text)
        self.assertNotIn("`/verbose`", help_text)
        self.assertNotIn("`/reasoning`", help_text)

    def test_bootstrap_inputs_are_not_persisted(self) -> None:
        self.assertFalse(_should_persist_input_message(""))
        self.assertFalse(_should_persist_input_message("   "))
        self.assertTrue(_should_persist_input_message("user"))

    def test_auto_title_skips_slash_commands(self) -> None:
        self.assertTrue(_should_skip_auto_title("/new"))
        self.assertTrue(_should_skip_auto_title("/model openai/gpt-4o"))
        self.assertFalse(_should_skip_auto_title("hello"))


if __name__ == "__main__":
    unittest.main()
