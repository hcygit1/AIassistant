from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_dispatcher import SessionWorkItem as CompatibilityWorkItem
from sessions.session_work_item import SessionWorkItem
from turns.events import TurnEvent


class SessionWorkItemBoundaryTests(unittest.TestCase):
    def test_dispatcher_keeps_the_same_compatibility_export(self) -> None:
        self.assertIs(CompatibilityWorkItem, SessionWorkItem)

    def test_work_item_keeps_callback_and_stream_defaults(self) -> None:
        stream_queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        item = SessionWorkItem(
            kind="user",
            priority=-10,
            content="hello",
            agent_id="main",
            session_id="main-main",
            turn_id="turn-1",
            stream_queue=stream_queue,
        )

        self.assertEqual(item.prompt_mode, "minimal")
        self.assertEqual(item.persist_role, "system")
        self.assertIs(item.stream_queue, stream_queue)
        self.assertIsNone(item.result_handler)
        self.assertIsNone(item.on_failure_async)

    def test_queue_and_executors_depend_on_the_work_item_module(self) -> None:
        paths = [
            BACKEND_DIR / "sessions" / "session_work_queue.py",
            BACKEND_DIR / "sessions" / "session_system_work_executor.py",
            BACKEND_DIR / "sessions" / "session_user_turn_executor.py",
        ]

        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "from sessions.session_dispatcher import SessionWorkItem",
                source,
                path.name,
            )
            self.assertIn(
                "from sessions.session_work_item import SessionWorkItem",
                source,
                path.name,
            )


if __name__ == "__main__":
    unittest.main()
