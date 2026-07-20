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
from sessions.session_work_delivery import SessionWorkDelivery
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


class _FakeDispatcherManager:
    def __init__(self, dispatcher: _FakeDispatcher) -> None:
        self._dispatcher = dispatcher

    def get(self, agent_id: str, session_id: str, lock: asyncio.Lock):
        return self._dispatcher


class _FakeLockManager:
    def __init__(self) -> None:
        self._lock = _FakeLock()

    def get_lock(self, agent_id: str, session_id: str) -> _FakeLock:
        return self._lock


class ReminderDeliveryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.store = SessionWorkStore(Path(tempdir.name) / "session_work.db")

    def _service(
        self,
        dispatcher: _FakeDispatcher,
        session_manager: Mock,
    ) -> ReminderDeliveryService:
        work_delivery = SessionWorkDelivery(
            work_store=self.store,
            dispatcher_manager=_FakeDispatcherManager(dispatcher),
            lock_manager=_FakeLockManager(),
        )
        return ReminderDeliveryService(
            session_manager=session_manager,
            work_delivery=work_delivery,
        )

    def test_deliver_cron_reminder_submits_cron_work_item_to_main_session(self) -> None:
        dispatcher = _FakeDispatcher()
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        service = self._service(dispatcher, session_manager)

        position = service.deliver_cron_reminder(
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
        service = self._service(dispatcher, session_manager)

        service.deliver_cron_reminder(
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
        prompt = ReminderDeliveryService().build_cron_prompt("")
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
        on_record_created = Mock()
        on_success = Mock()
        on_failure = Mock()
        on_cancel = Mock()

        position = service.deliver_cron_reminder(
            agent_id="main",
            text="review queue",
            run_id="cron-injected",
            on_record_created=on_record_created,
            on_success=on_success,
            on_failure=on_failure,
            on_cancel=on_cancel,
        )

        self.assertEqual(position, 4)
        session_manager.resolve_main_session_id.assert_called_once_with("main")
        work_delivery.deliver.assert_called_once()
        self.assertEqual(work_delivery.deliver.call_args.kwargs["kind"], "cron")
        self.assertEqual(
            work_delivery.deliver.call_args.kwargs["session_id"],
            "main-main",
        )
        self.assertIs(
            work_delivery.deliver.call_args.kwargs["on_record_created"],
            on_record_created,
        )
        self.assertIs(
            work_delivery.deliver.call_args.kwargs["on_success"],
            on_success,
        )
        self.assertIs(
            work_delivery.deliver.call_args.kwargs["on_failure"],
            on_failure,
        )
        self.assertIs(
            work_delivery.deliver.call_args.kwargs["on_cancel"],
            on_cancel,
        )


if __name__ == "__main__":
    unittest.main()
