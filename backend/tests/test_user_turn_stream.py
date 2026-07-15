from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime import user_turn_stream
from turns.events import TurnEvent


class UserTurnStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_events_remain_structured_until_transport_boundary(
        self,
    ) -> None:
        iterator = getattr(user_turn_stream, "iter_user_turn_events", None)
        self.assertIsNotNone(iterator, "iter_user_turn_events should be defined")

        async def fake_agent_stream(**_kwargs):
            yield {"type": "token", "content": "hello"}
            yield {"type": "done", "content": "hello", "session_id": "s1"}

        with (
            patch(
                "runtime.agent.agent_manager.astream",
                fake_agent_stream,
            ),
            patch(
                "sessions.session_manager.session_manager.load_session",
                return_value={"messages": [{"role": "user", "content": "existing"}]},
            ),
        ):
            events = [
                event
                async for event in iterator(
                    "hello",
                    "s1",
                    "main",
                    "turn-1",
                )
            ]

        self.assertEqual([event.type for event in events], ["token", "done"])
        self.assertTrue(all(isinstance(event, TurnEvent) for event in events))
        self.assertEqual(events[0].payload["content"], "hello")

    async def test_non_json_agent_event_becomes_structured_error(self) -> None:
        async def fake_agent_stream(**_kwargs):
            yield {"type": "done", "metadata": object()}

        with (
            patch(
                "runtime.agent.agent_manager.astream",
                fake_agent_stream,
            ),
            patch(
                "sessions.session_manager.session_manager.load_session",
                return_value={"messages": [{"role": "user", "content": "existing"}]},
            ),
        ):
            events = [
                event
                async for event in user_turn_stream.iter_user_turn_events(
                    "hello",
                    "s1",
                    "main",
                    "turn-1",
                )
            ]

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "error")
        self.assertIn("not JSON-compatible", events[0].error_message or "")


if __name__ == "__main__":
    unittest.main()
