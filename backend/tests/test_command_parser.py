from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent import _should_persist_input_message
from runtime.command_parser import execute_command, format_help, parse_command
from runtime.user_turn_stream import _should_skip_auto_title


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
