from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.turn_event_stream import TurnEventStreamProcessor
from runtime.turn_models import TurnExecutionRequest


class TurnEventStreamProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_input_token_details_records_zero_cache_reads(
        self,
    ) -> None:
        class _Agent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=SimpleNamespace(
                                input_tokens=3,
                                output_tokens=2,
                                input_token_details=None,
                            ),
                        )
                    },
                }

        request = TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=SimpleNamespace(),
            provider="fake",
            model="model",
            message="question",
            persist_input_role="user",
            system_prompt="system",
            tools=[],
            history=[],
            recursion_limit=10,
            prompt_tokens=0,
            summary_tokens=0,
            history_tokens=0,
            active_tokens=1000,
        )
        tracker = SimpleNamespace(
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        processor = TurnEventStreamProcessor(
            agent=_Agent(),
            request=request,
            messages=[],
            turn=SimpleNamespace(run_id="turn-1"),
            turn_start_event={"type": "lifecycle", "event": "turn_start"},
            run_tracker=tracker,
            audit_logger=SimpleNamespace(),
            get_lifecycle_hooks=lambda: None,
            emit_event=Mock(),
            loop_detector=SimpleNamespace(record=Mock(return_value=None)),
            new_tool_call_id=lambda: "tc-fixed",
            infer_tool_result_status=lambda _output: ("success", None),
            loop_warning_is_breaker=lambda _warning: False,
        )

        events = [event async for event in processor.stream()]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "token"],
        )
        tracker.record_tokens.assert_called_once_with(
            "turn-1",
            input_tokens=3,
            output_tokens=2,
            cache_read=0,
        )

    async def test_translates_native_model_and_tool_events_in_order(self) -> None:
        class _Agent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=SimpleNamespace(
                                input_tokens=3,
                                output_tokens=2,
                                input_token_details={"cache_read": 1},
                            ),
                        )
                    },
                }
                yield {
                    "event": "on_tool_start",
                    "run_id": "tool-run",
                    "name": "read",
                    "data": {"input": {"path": "note.txt"}},
                }
                yield {
                    "event": "on_tool_end",
                    "run_id": "tool-run",
                    "name": "read",
                    "data": {"output": "contents"},
                }

        request = TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=SimpleNamespace(),
            provider="fake",
            model="model",
            message="question",
            persist_input_role="user",
            system_prompt="system",
            tools=[],
            history=[],
            recursion_limit=10,
            prompt_tokens=0,
            summary_tokens=0,
            history_tokens=0,
            active_tokens=1000,
        )
        tracker = SimpleNamespace(
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        hooks = SimpleNamespace(
            on_before_tool_call=AsyncMock(),
            on_after_tool_call=AsyncMock(),
        )
        processor = TurnEventStreamProcessor(
            agent=_Agent(),
            request=request,
            messages=[{"role": "user", "content": "question"}],
            turn=SimpleNamespace(run_id="turn-1"),
            turn_start_event={"type": "lifecycle", "phase": "turn_start"},
            run_tracker=tracker,
            audit_logger=audit,
            get_lifecycle_hooks=lambda: hooks,
            emit_event=Mock(),
            loop_detector=SimpleNamespace(record=Mock(return_value=None)),
            new_tool_call_id=lambda: "tc-fixed",
            infer_tool_result_status=lambda _output: ("success", None),
            loop_warning_is_breaker=lambda _warning: False,
        )

        events = [event async for event in processor.stream()]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "token", "tool_start", "tool_end"],
        )
        self.assertEqual(events[1]["content"], "answer")
        self.assertEqual(events[2]["tool_call_id"], "tc-fixed")
        self.assertEqual(events[3]["output"], "contents")
        self.assertEqual(processor.state.full_response, "answer")
        self.assertEqual(processor.state.step_count, 1)
        self.assertEqual(
            processor.state.tool_calls_log,
            [
                {
                    "tool_call_id": "tc-fixed",
                    "tool": "read",
                    "status": "success",
                    "input": {"path": "note.txt"},
                    "output": "contents",
                    "error": None,
                }
            ],
        )
        tracker.record_tokens.assert_called_once_with(
            "turn-1",
            input_tokens=3,
            output_tokens=2,
            cache_read=1,
        )
        hooks.on_before_tool_call.assert_awaited_once()
        hooks.on_after_tool_call.assert_awaited_once()

    async def test_emits_loop_and_dangerous_tool_events(self) -> None:
        class _Agent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_tool_start",
                    "run_id": "tool-run",
                    "name": "exec",
                    "data": {"input": {"command": "pwd"}},
                }
                yield {
                    "event": "on_tool_end",
                    "run_id": "tool-run",
                    "name": "exec",
                    "data": {"output": "/tmp"},
                }

        request = TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=SimpleNamespace(),
            provider="fake",
            model="model",
            message="run pwd",
            persist_input_role="user",
            system_prompt="system",
            tools=[],
            history=[],
            recursion_limit=10,
            prompt_tokens=0,
            summary_tokens=0,
            history_tokens=0,
            active_tokens=1000,
        )
        tracker = SimpleNamespace(
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        emit_event = Mock()
        processor = TurnEventStreamProcessor(
            agent=_Agent(),
            request=request,
            messages=[],
            turn=SimpleNamespace(run_id="turn-1"),
            turn_start_event={"type": "lifecycle", "event": "turn_start"},
            run_tracker=tracker,
            audit_logger=audit,
            get_lifecycle_hooks=lambda: None,
            emit_event=emit_event,
            loop_detector=SimpleNamespace(
                record=Mock(return_value="repeated tool call")
            ),
            new_tool_call_id=lambda: "tc-fixed",
            infer_tool_result_status=lambda _output: ("success", None),
            loop_warning_is_breaker=lambda _warning: False,
        )

        events = [event async for event in processor.stream()]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "tool_start", "tool_end", "lifecycle"],
        )
        self.assertEqual(events[-1]["event"], "tool_loop_warning")
        self.assertEqual(
            [call.args[1]["event"] for call in emit_event.call_args_list],
            ["tool_loop_warning", "tool_dangerous_executed"],
        )
        audit.log_tool_loop_warning.assert_called_once_with(
            "main",
            "turn-1",
            "exec",
            "repeated tool call",
            tool_call_id="tc-fixed",
        )


if __name__ == "__main__":
    unittest.main()
