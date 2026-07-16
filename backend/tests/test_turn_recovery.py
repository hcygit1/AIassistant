from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.turn_recovery import TurnRecovery


class TurnRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_http_error_once(
        self,
    ) -> None:
        attempts = 0
        sleep = AsyncMock()

        async def _stream():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("503 upstream unavailable")
            yield {
                "type": "done",
                "content": "ok",
                "session_id": "s1",
            }

        recovery = TurnRecovery(
            reset_session=Mock(),
            compress_session=AsyncMock(),
            audit_log=Mock(),
            sleep=sleep,
        )
        events = [
            event
            async for event in recovery.run(
                agent_id="main",
                session_id="s1",
                state=SimpleNamespace(compaction_count=1),
                stream=_stream,
            )
        ]

        self.assertEqual(attempts, 2)
        sleep.assert_awaited_once_with(2.5)
        self.assertEqual(events[-1]["type"], "done")

    async def test_does_not_retry_committed_stream(
        self,
    ) -> None:
        attempts = 0

        async def _stream():
            nonlocal attempts
            attempts += 1
            yield {"type": "token", "content": "partial"}
            error = RuntimeError("503 upstream unavailable")
            error.committed = True
            raise error

        recovery = TurnRecovery(
            reset_session=Mock(),
            compress_session=AsyncMock(),
            audit_log=Mock(),
            sleep=AsyncMock(),
        )
        events = [
            event
            async for event in recovery.run(
                agent_id="main",
                session_id="s1",
                state=SimpleNamespace(compaction_count=1),
                stream=_stream,
            )
        ]

        self.assertEqual(attempts, 1)
        self.assertEqual(
            [event["type"] for event in events],
            ["token", "lifecycle", "error"],
        )

    async def test_resets_session_after_compaction_failure(
        self,
    ) -> None:
        async def _stream():
            if False:
                yield {}
            raise RuntimeError(
                "compaction failed: context overflow"
            )

        state = SimpleNamespace(compaction_count=3)
        reset_session = Mock()
        audit_log = Mock()
        recovery = TurnRecovery(
            reset_session=reset_session,
            compress_session=AsyncMock(),
            audit_log=audit_log,
            sleep=AsyncMock(),
        )
        events = [
            event
            async for event in recovery.run(
                agent_id="main",
                session_id="s1",
                state=state,
                stream=_stream,
            )
        ]

        reset_session.assert_called_once_with("s1", "main")
        self.assertEqual(state.compaction_count, 0)
        audit_log.assert_called_once()
        self.assertEqual(
            [event["type"] for event in events],
            ["session_reset", "done"],
        )
        self.assertEqual(
            events[0]["memory"]["reason"],
            "compaction_failure",
        )

    async def test_resets_role_ordering_error(
        self,
    ) -> None:
        async def _stream():
            if False:
                yield {}
            raise RuntimeError(
                "roles must alternate between user and assistant"
            )

        state = SimpleNamespace(compaction_count=2)
        reset_session = Mock()
        recovery = TurnRecovery(
            reset_session=reset_session,
            compress_session=AsyncMock(),
            audit_log=Mock(),
            sleep=AsyncMock(),
        )
        events = [
            event
            async for event in recovery.run(
                agent_id="main",
                session_id="s1",
                state=state,
                stream=_stream,
            )
        ]

        reset_session.assert_called_once_with("s1", "main")
        self.assertEqual(state.compaction_count, 0)
        self.assertIn("消息顺序冲突", events[-1]["content"])

    async def test_retries_after_forced_compaction(
        self,
    ) -> None:
        attempts = 0

        async def _stream():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError(
                    "context overflow: too many tokens"
                )
            yield {
                "type": "done",
                "content": "ok",
                "session_id": "s1",
            }

        compress_session = AsyncMock(
            return_value={
                "compress": {},
                "post_compaction": {},
            }
        )
        audit_log = Mock()
        recovery = TurnRecovery(
            reset_session=Mock(),
            compress_session=compress_session,
            audit_log=audit_log,
            sleep=AsyncMock(),
        )
        events = [
            event
            async for event in recovery.run(
                agent_id="main",
                session_id="s1",
                state=SimpleNamespace(compaction_count=1),
                stream=_stream,
            )
        ]

        self.assertEqual(attempts, 2)
        compress_session.assert_awaited_once_with(
            "s1",
            "main",
            level="forced",
        )
        audit_log.assert_called_once()
        self.assertEqual(events[-1]["content"], "ok")

    async def test_emits_terminal_events_for_generic_error(
        self,
    ) -> None:
        async def _stream():
            if False:
                yield {}
            raise RuntimeError("unexpected failure")

        recovery = TurnRecovery(
            reset_session=Mock(),
            compress_session=AsyncMock(),
            audit_log=Mock(),
            sleep=AsyncMock(),
        )
        events = [
            event
            async for event in recovery.run(
                agent_id="main",
                session_id="s1",
                state=SimpleNamespace(compaction_count=1),
                stream=_stream,
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "error"],
        )
        self.assertEqual(
            events[-1]["error"],
            "unexpected failure",
        )


if __name__ == "__main__":
    unittest.main()
