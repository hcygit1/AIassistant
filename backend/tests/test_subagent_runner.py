from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRegistry
from subagents.subagent_runner import SubagentRunner


class _StreamingManager:
    def __init__(self, events: list[dict]) -> None:
        self.events = events

    async def astream(self, **kwargs):
        for event in self.events:
            yield event


class _BlockingManager:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def astream(self, **kwargs):
        self.started.set()
        await self.release.wait()
        yield {"type": "token", "content": "late"}


class SubagentRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with patch.object(SubagentRegistry, "_restore_from_disk"):
            self.registry = SubagentRegistry()
        self.registry._persist_to_disk = Mock()
        self.registry.register_run(
            run_id="run-1",
            child_session_key="agent:worker:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
            label="inspection",
        )
        self.event_bus = Mock()
        self.session_manager = Mock()
        self.delivery = Mock()
        self.delivery.deliver_to_requester = AsyncMock()

    def _runner(self, manager) -> SubagentRunner:
        return SubagentRunner(
            agent_manager=manager,
            requester_agent_id="main",
            registry=self.registry,
            session_manager=self.session_manager,
            event_bus=self.event_bus,
            delivery=self.delivery,
        )

    async def test_run_streams_events_and_delivers_success(self) -> None:
        manager = _StreamingManager([
            {"type": "token", "content": "done"},
            {"type": "tool_start", "tool": "read"},
            {
                "type": "tool_end",
                "tool": "read",
                "output": "file contents",
            },
        ])

        await self._runner(manager).run(
            run_id="run-1",
            session_id="child-1",
            agent_id="worker",
            task="inspect files",
            requester_key="agent:main:main",
        )

        record = self.registry.get_run("run-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "succeeded")  # type: ignore[union-attr]
        self.assertEqual(record.result_summary, "done")  # type: ignore[union-attr]
        event_types = [
            call.args[1]["type"]
            for call in self.event_bus.emit.call_args_list
        ]
        self.assertIn("subagent_start", event_types)
        self.assertIn("subagent_tool", event_types)
        self.assertIn("subagent_tool_end", event_types)
        self.assertIn("subagent_done", event_types)
        self.delivery.deliver_to_requester.assert_awaited_once()
        kwargs = self.delivery.deliver_to_requester.await_args.kwargs
        self.assertEqual(kwargs["result"], "done")
        self.assertEqual(kwargs["outcome"], "completed successfully")
        self.assertNotIn("debug_log", kwargs)

    async def test_run_timeout_marks_terminal_and_delivers_fallback(self) -> None:
        manager = _BlockingManager()
        self.session_manager.load_session.return_value = None

        await self._runner(manager).run(
            run_id="run-1",
            session_id="child-1",
            agent_id="worker",
            task="inspect files",
            requester_key="agent:main:main",
            run_timeout_seconds=0.01,
        )

        record = self.registry.get_run("run-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "timed_out")  # type: ignore[union-attr]
        kwargs = self.delivery.deliver_to_requester.await_args.kwargs
        self.assertEqual(kwargs["outcome"], "timed out")
        self.assertIn("timed out", kwargs["result"])
        self.assertIn(
            "subagent_error",
            [
                call.args[1]["type"]
                for call in self.event_bus.emit.call_args_list
            ],
        )

    async def test_registry_kill_cancels_running_task(self) -> None:
        manager = _BlockingManager()
        runner = self._runner(manager)

        task = runner.start(
            run_id="run-1",
            session_id="child-1",
            agent_id="worker",
            task="inspect files",
            requester_key="agent:main:main",
        )
        await manager.started.wait()
        self.assertTrue(self.registry.kill("run-1"))
        await task

        record = self.registry.get_run("run-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "cancelled")  # type: ignore[union-attr]
        event_types = [
            call.args[1]["type"]
            for call in self.event_bus.emit.call_args_list
        ]
        self.assertIn("subagent_killed", event_types)
        self.assertNotIn("subagent_done", event_types)
        self.delivery.deliver_to_requester.assert_not_awaited()

    def test_collect_latest_output_reads_session_when_stream_is_empty(self) -> None:
        self.session_manager.load_session.return_value = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "summary",
                    "tool_calls": [
                        {"tool": "exec", "output": "failed"}
                    ],
                }
            ]
        }

        result, all_failed = self._runner(
            _StreamingManager([])
        ).collect_latest_output(
            session_id="child-1",
            agent_id="worker",
            streamed_text="",
            tool_calls=[],
        )

        self.assertIn("summary", result)
        self.assertIn("[exec] failed", result)
        self.assertTrue(all_failed)


if __name__ == "__main__":
    unittest.main()
