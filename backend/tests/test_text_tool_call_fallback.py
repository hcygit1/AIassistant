from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.text_tool_call_fallback import TextToolCallFallbackProcessor
from runtime.turn_event_stream import TurnEventStreamState
from runtime.turn_models import TurnExecutionRequest


class TextToolCallFallbackProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_text_tool_call_and_updates_shared_state(self) -> None:
        tool = SimpleNamespace(name="read")
        request = TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=SimpleNamespace(),
            provider="fake",
            model="model",
            message="read identity",
            persist_input_role="user",
            system_prompt="system",
            tools=[tool],
            history=[],
            recursion_limit=10,
            prompt_tokens=0,
            summary_tokens=0,
            history_tokens=0,
            active_tokens=1000,
        )
        state = TurnEventStreamState(
            full_response='functions.read:0{"path":""}',
        )
        tracker = SimpleNamespace(
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        invoke_tool = AsyncMock(return_value="identity contents")
        processor = TextToolCallFallbackProcessor(
            request=request,
            turn=SimpleNamespace(run_id="turn-1"),
            state=state,
            run_tracker=tracker,
            audit_logger=audit,
            emit_event=Mock(),
            loop_detector=SimpleNamespace(record=Mock(return_value=None)),
            parse_text_tool_calls=lambda _content: [("read", {})],
            strip_tool_call_patterns=lambda _content: "clean answer",
            invoke_tool_async=invoke_tool,
            format_tool_error=lambda _name, error: str(error),
            is_untrusted_source_tool=lambda _name: True,
            new_tool_call_id=lambda: "tc-fixed",
            infer_tool_result_status=lambda _output: ("success", None),
            loop_warning_is_breaker=lambda _warning: False,
        )

        events = [event async for event in processor.stream()]

        self.assertEqual(
            [event["type"] for event in events],
            ["content_refresh", "tool_start", "tool_end"],
        )
        self.assertEqual(events[0]["content"], "clean answer")
        self.assertEqual(events[1]["input"], {"path": "IDENTITY.md"})
        self.assertEqual(events[2]["output"], "identity contents")
        self.assertEqual(state.full_response, "clean answer")
        self.assertTrue(state.content_refresh_sent)
        self.assertTrue(state.recent_untrusted_content)
        self.assertEqual(state.step_count, 1)
        self.assertEqual(
            state.tool_calls_log,
            [
                {
                    "tool_call_id": "tc-fixed",
                    "tool": "read",
                    "status": "success",
                    "input": {"path": "IDENTITY.md"},
                    "output": "identity contents",
                    "error": None,
                }
            ],
        )
        invoke_tool.assert_awaited_once_with(
            tool,
            {"path": "IDENTITY.md"},
            user_message="read identity",
            recent_untrusted_content=False,
        )
        tracker.record_tool_start.assert_called_once_with(
            "turn-1",
            "read",
            {"path": "IDENTITY.md"},
            tool_call_id="tc-fixed",
        )
        audit.log_tool_call.assert_called_once_with(
            "main",
            "turn-1",
            "read",
            {"path": "IDENTITY.md"},
            "identity contents",
            tool_call_id="tc-fixed",
            status="success",
            error=None,
        )


if __name__ == "__main__":
    unittest.main()
