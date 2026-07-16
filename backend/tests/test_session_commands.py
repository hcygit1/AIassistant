from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.session_commands import SessionCommands


class SessionCommandsTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_reset_flushes_memory_and_resets_session_state(
        self,
    ) -> None:
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        reset_session = Mock()
        audit_log = Mock()
        commands = SessionCommands(
            load_session=lambda _session_id, _agent_id: {
                "messages": messages,
            },
            reset_session=reset_session,
            resolve_agent_config=lambda _agent_id: {},
            emit_event=lambda _agent_id, _event: None,
            audit_log=audit_log,
        )
        store = SimpleNamespace(delete_session_summary=Mock())
        state = SimpleNamespace(compaction_count=3)
        batch_ingest = AsyncMock()
        switch_model = Mock(return_value="provider/model")

        events = [
            event
            async for event in commands.handle_reset(
                "s1",
                "main",
                model_override="provider/model",
                get_store=lambda _agent_id: store,
                get_state=lambda _agent_id: state,
                batch_ingest_messages=batch_ingest,
                switch_model=switch_model,
            )
        ]

        batch_ingest.assert_awaited_once_with(
            "main",
            "s1",
            messages,
            session_end=True,
        )
        reset_session.assert_called_once_with("s1", "main")
        store.delete_session_summary.assert_called_once_with(
            "s1",
            "main",
        )
        self.assertEqual(state.compaction_count, 0)
        switch_model.assert_called_once_with("main", "provider/model")
        audit_log.assert_called_once()
        self.assertEqual(
            [event["type"] for event in events],
            ["command_response", "command_response", "session_reset"],
        )
        self.assertIn("provider/model", events[1]["response"])

    async def test_handle_reset_noflush_skips_memory_ingest(
        self,
    ) -> None:
        reset_session = Mock()
        audit_log = Mock()
        commands = SessionCommands(
            load_session=lambda _session_id, _agent_id: {
                "messages": [{"role": "user", "content": "discard"}],
            },
            reset_session=reset_session,
            resolve_agent_config=lambda _agent_id: {},
            emit_event=lambda _agent_id, _event: None,
            audit_log=audit_log,
        )
        store = SimpleNamespace(delete_session_summary=Mock())
        state = SimpleNamespace(compaction_count=2)

        events = [
            event
            async for event in commands.handle_reset_noflush(
                "s1",
                "main",
                get_store=lambda _agent_id: store,
                get_state=lambda _agent_id: state,
            )
        ]

        reset_session.assert_called_once_with("s1", "main")
        store.delete_session_summary.assert_called_once_with(
            "s1",
            "main",
        )
        self.assertEqual(state.compaction_count, 0)
        self.assertEqual(events[-1]["memory"]["saved"], False)
        self.assertEqual(
            audit_log.call_args.args[2]["memory_saved"],
            False,
        )

    async def test_handle_compact_emits_result_and_terminal_event(
        self,
    ) -> None:
        emitted = []
        commands = SessionCommands(
            load_session=lambda _session_id, _agent_id: None,
            reset_session=lambda _session_id, _agent_id: None,
            resolve_agent_config=lambda _agent_id: {},
            emit_event=lambda agent_id, event: emitted.append(
                (agent_id, event)
            ),
            audit_log=lambda *_args: None,
        )
        compress_session = AsyncMock(
            return_value={
                "compress": {
                    "archived_count": 4,
                    "remaining_count": 6,
                }
            }
        )

        events = [
            event
            async for event in commands.handle_compact(
                "s1",
                "main",
                compress_session=compress_session,
                calc_compress_count_by_turns=lambda _messages, _turns: 0,
            )
        ]

        compress_session.assert_awaited_once_with("s1", "main")
        self.assertEqual(
            [event["type"] for event in events],
            [
                "command_response",
                "command_response",
                "session_compacted",
                "done",
            ],
        )
        self.assertEqual(events[2]["result"]["compress"]["archived_count"], 4)
        self.assertEqual(emitted[0][0], "main")
        self.assertGreaterEqual(len(emitted), 2)

    async def test_handle_compact_reports_why_compaction_was_skipped(
        self,
    ) -> None:
        emitted = []
        commands = SessionCommands(
            load_session=lambda _session_id, _agent_id: {
                "messages": [
                    {"role": "user", "content": "question"},
                ]
            },
            reset_session=lambda _session_id, _agent_id: None,
            resolve_agent_config=lambda _agent_id: {
                "compaction": {"keepRecentTurns": 12}
            },
            emit_event=lambda _agent_id, event: emitted.append(event),
            audit_log=lambda *_args: None,
        )

        events = [
            event
            async for event in commands.handle_compact(
                "s1",
                "main",
                compress_session=AsyncMock(
                    return_value={"error": "消息过少，无需压缩"}
                ),
                calc_compress_count_by_turns=lambda _messages, _turns: 0,
            )
        ]

        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("至少累积到 4 条", events[-1]["content"])
        self.assertEqual(emitted[-1]["event"], "manual_compact_skipped")


if __name__ == "__main__":
    unittest.main()
