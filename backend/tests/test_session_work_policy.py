from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_policy import (
    PRIORITY_ANNOUNCE,
    PRIORITY_CRON,
    PRIORITY_HEARTBEAT,
    deliver_system_work,
)


class SessionWorkPolicyTests(unittest.TestCase):
    def test_system_work_kinds_apply_canonical_delivery_policy(self) -> None:
        cases = (
            ("announce", PRIORITY_ANNOUNCE, False),
            ("cron", PRIORITY_CRON, True),
            ("heartbeat", PRIORITY_HEARTBEAT, False),
        )

        for kind, priority, recover_on_restart in cases:
            with self.subTest(kind=kind):
                delivery = Mock()
                delivery.deliver.return_value = 3
                on_success = Mock()

                position = deliver_system_work(
                    delivery,
                    kind=kind,
                    content=f"{kind} content",
                    agent_id="main",
                    session_id="main-main",
                    run_id="run-1",
                    on_success=on_success,
                )

                self.assertEqual(position, 3)
                delivery.deliver.assert_called_once_with(
                    kind=kind,
                    priority=priority,
                    recover_on_restart=recover_on_restart,
                    content=f"{kind} content",
                    agent_id="main",
                    session_id="main-main",
                    run_id="run-1",
                    on_success=on_success,
                )

    def test_unknown_system_work_kind_is_rejected_before_delivery(self) -> None:
        delivery = Mock()

        with self.assertRaisesRegex(ValueError, "unknown system work kind"):
            deliver_system_work(
                delivery,
                kind="user",
                content="not system work",
                agent_id="main",
                session_id="main-main",
            )

        delivery.deliver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
