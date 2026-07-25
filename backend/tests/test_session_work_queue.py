from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_dispatcher import SessionWorkItem
from sessions.session_work_queue import (
    PRIORITY_MIN_SYSTEM,
    PRIORITY_USER,
    SessionWorkQueue,
)


class SessionWorkQueueTests(unittest.TestCase):
    def _item(
        self,
        *,
        kind: str,
        priority: int,
        created_at: float,
        turn_id: str | None = None,
        work_id: str | None = None,
    ) -> SessionWorkItem:
        return SessionWorkItem(
            kind=kind,
            priority=priority,
            content=kind,
            agent_id="main",
            session_id="main-main",
            created_at=created_at,
            turn_id=turn_id,
            work_id=work_id,
        )

    def test_user_priority_and_system_aging_order_are_preserved(self) -> None:
        queue = SessionWorkQueue(now=lambda: 100.0)
        heartbeat = self._item(
            kind="heartbeat",
            priority=3,
            created_at=0.0,
        )
        announce = self._item(
            kind="announce",
            priority=0,
            created_at=99.0,
        )
        user = self._item(
            kind="user",
            priority=99,
            created_at=100.0,
        )
        aged_system = self._item(
            kind="cron",
            priority=-8,
            created_at=0.0,
        )

        queue.submit(heartbeat)
        queue.submit(announce)
        queue.submit(user)
        queue.submit(aged_system)

        self.assertEqual(
            [
                queue.pop_next(),
                queue.pop_next(),
                queue.pop_next(),
                queue.pop_next(),
            ],
            [user, aged_system, announce, heartbeat],
        )
        self.assertEqual(queue.effective_priority(user), PRIORITY_USER)
        self.assertEqual(
            queue.effective_priority(aged_system),
            PRIORITY_MIN_SYSTEM,
        )

    def test_default_clock_remains_dynamic_for_compatibility_patches(self) -> None:
        current_time = time.time()
        queue = SessionWorkQueue()
        item = self._item(
            kind="cron",
            priority=3,
            created_at=current_time,
        )

        with patch(
            "sessions.session_dispatcher.time.time",
            return_value=current_time + 30.0,
        ):
            priority = queue.effective_priority(item)

        self.assertEqual(priority, 2.0)

    def test_position_remove_and_drain_share_one_ordered_snapshot(self) -> None:
        queue = SessionWorkQueue(now=lambda: 100.0)
        cron = self._item(
            kind="cron",
            priority=1,
            created_at=90.0,
            turn_id="turn-cron",
            work_id="work-cron",
        )
        announce = self._item(
            kind="announce",
            priority=0,
            created_at=95.0,
            turn_id="turn-announce",
            work_id="work-announce",
        )

        self.assertEqual(queue.submit(cron), 1)
        self.assertEqual(queue.submit(announce), 2)
        self.assertEqual(queue.position("turn-announce"), 1)
        self.assertIs(queue.remove("work-announce"), announce)
        self.assertIsNone(queue.remove("missing"))
        self.assertEqual(queue.drain(), [cron])
        self.assertEqual(len(queue), 0)


if __name__ == "__main__":
    unittest.main()
