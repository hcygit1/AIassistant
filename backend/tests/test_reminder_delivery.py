from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from system_messages.reminder_delivery import ReminderDeliveryService
from sessions.session_work_store import SessionWorkStore


class _FakeLock:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()


class _FakeDispatcher:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, work_item) -> int:
        self.submitted.append(work_item)
        return len(self.submitted)


class ReminderDeliveryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        store = SessionWorkStore(Path(tempdir.name) / "session_work.db")
        store_patch = patch(
            "sessions.session_work_delivery.session_work_store",
            store,
        )
        store_patch.start()
        self.addCleanup(store_patch.stop)
        self.service = ReminderDeliveryService()

    def test_deliver_cron_reminder_submits_cron_work_item_to_main_session(self) -> None:
        dispatcher = _FakeDispatcher()
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"

        with (
            patch("sessions.session_manager.session_manager", session_manager),
            patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
            patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
        ):
            position = self.service.deliver_cron_reminder(
                agent_id="main",
                text="remember to review docs",
                run_id="cron-1",
            )

        self.assertEqual(position, 1)
        self.assertEqual(len(dispatcher.submitted), 1)
        work_item = dispatcher.submitted[0]
        self.assertEqual(work_item.kind, "cron")
        self.assertEqual(work_item.agent_id, "main")
        self.assertEqual(work_item.session_id, "main-main")
        self.assertEqual(work_item.run_id, "cron-1")
        self.assertIn("remember to review docs", work_item.content)

    def test_deliver_cron_reminder_respects_explicit_session_id(self) -> None:
        dispatcher = _FakeDispatcher()
        session_manager = Mock()

        with (
            patch("sessions.session_manager.session_manager", session_manager),
            patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
            patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
        ):
            self.service.deliver_cron_reminder(
                agent_id="main",
                session_id="subagent-1",
                text="ping child session",
                run_id="cron-2",
            )

        session_manager.resolve_main_session_id.assert_not_called()
        work_item = dispatcher.submitted[0]
        self.assertEqual(work_item.session_id, "subagent-1")
        self.assertIn("ping child session", work_item.content)

    def test_build_cron_prompt_handles_empty_text(self) -> None:
        prompt = self.service.build_cron_prompt("")
        self.assertIn("no reminder content", prompt.lower())

    def test_deliver_cron_reminder_uses_injected_dependencies(self) -> None:
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()
        work_delivery.deliver.return_value = 4
        service = ReminderDeliveryService(
            session_manager=session_manager,
            work_delivery=work_delivery,
        )

        position = service.deliver_cron_reminder(
            agent_id="main",
            text="review queue",
            run_id="cron-injected",
        )

        self.assertEqual(position, 4)
        session_manager.resolve_main_session_id.assert_called_once_with("main")
        work_delivery.deliver.assert_called_once()
        self.assertEqual(work_delivery.deliver.call_args.kwargs["kind"], "cron")
        self.assertEqual(
            work_delivery.deliver.call_args.kwargs["session_id"],
            "main-main",
        )


if __name__ == "__main__":
    unittest.main()
