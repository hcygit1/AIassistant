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

from subagents.subagent_delivery import SubagentAnnounceDelivery
from subagents.subagent_registry import SubagentRunRecord
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


class SubagentAnnounceDeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.store = SessionWorkStore(Path(tempdir.name) / "session_work.db")

    def _delivery(self, dispatcher: _FakeDispatcher) -> SubagentAnnounceDelivery:
        work_delivery = SessionWorkDelivery(
            work_store=self.store,
            dispatcher_manager=_FakeDispatcherManager(dispatcher),
            lock_manager=_FakeLockManager(),
        )
        return SubagentAnnounceDelivery(work_delivery=work_delivery)

    async def test_deliver_to_main_requester_submits_announce_work_item(self) -> None:
        dispatcher = _FakeDispatcher()
        registry = Mock()
        event_bus = Mock()
        session_manager = Mock()
        delivery = self._delivery(dispatcher)
        session_manager.session_id_from_session_key.return_value = ("main", "main-main")
        session_manager.resolve_main_session_id.return_value = "main-main"

        with (
            patch("config.get_config", return_value={"app": {"locale": "en"}}),
            patch("sessions.session_manager.session_manager", session_manager),
            patch("subagents.subagent_registry.registry", registry),
            patch("infra.event_bus.event_bus", event_bus),
        ):
            await delivery.deliver_to_requester(
                requester_key="agent:main:main",
                child_session_key="agent:main:subagent:child-1",
                run_id="run-1",
                task="check docs",
                result="done",
            )

        self.assertEqual(len(dispatcher.submitted), 1)
        work_item = dispatcher.submitted[0]
        self.assertEqual(work_item.kind, "announce")
        self.assertEqual(work_item.agent_id, "main")
        self.assertEqual(work_item.session_id, "main-main")
        registry.set_result_delivery_state.assert_called_once_with("run-1", "queued")
        registry.set_delivery_work_id.assert_called_once()
        self.assertTrue(event_bus.emit.called)

    async def test_main_requester_failure_callback_falls_back_to_save_message(self) -> None:
        dispatcher = _FakeDispatcher()
        registry = Mock()
        event_bus = Mock()
        session_manager = Mock()
        delivery = self._delivery(dispatcher)
        session_manager.session_id_from_session_key.return_value = ("main", "main-main")
        session_manager.resolve_main_session_id.return_value = "main-main"

        with (
            patch("config.get_config", return_value={"app": {"locale": "en"}}),
            patch("sessions.session_manager.session_manager", session_manager),
            patch("subagents.subagent_registry.registry", registry),
            patch("infra.event_bus.event_bus", event_bus),
        ):
            await delivery.deliver_to_requester(
                requester_key="agent:main:main",
                child_session_key="agent:main:subagent:child-1",
                run_id="run-2",
                task="check docs",
                result="done",
            )

            work_item = dispatcher.submitted[0]
            self.assertIsNotNone(work_item.on_failure)
            work_item.on_failure()

        session_manager.save_message.assert_called_once()
        registry.mark_result_delivery_dropped.assert_called_once_with("run-2")

    async def test_main_requester_cancel_callback_marks_delivery_dropped(self) -> None:
        dispatcher = _FakeDispatcher()
        registry = Mock()
        event_bus = Mock()
        session_manager = Mock()
        delivery = self._delivery(dispatcher)
        session_manager.session_id_from_session_key.return_value = (
            "main",
            "main-main",
        )
        session_manager.resolve_main_session_id.return_value = "main-main"

        with (
            patch("config.get_config", return_value={"app": {"locale": "en"}}),
            patch("sessions.session_manager.session_manager", session_manager),
            patch("subagents.subagent_registry.registry", registry),
            patch("infra.event_bus.event_bus", event_bus),
        ):
            await delivery.deliver_to_requester(
                requester_key="agent:main:main",
                child_session_key="agent:main:subagent:child-1",
                run_id="run-cancelled",
                task="check docs",
                result="done",
            )
            work_item = dispatcher.submitted[0]
            self.assertIsNotNone(work_item.on_cancel)
            work_item.on_cancel()

        registry.mark_result_delivery_dropped.assert_called_once_with(
            "run-cancelled"
        )
        session_manager.save_message.assert_not_called()

    async def test_deliver_recovered_run_submits_announce_work_item(self) -> None:
        dispatcher = _FakeDispatcher()
        registry = Mock()
        event_bus = Mock()
        session_manager = Mock()
        delivery = self._delivery(dispatcher)
        session_manager.session_id_from_session_key.return_value = ("main", "main-main")
        session_manager.resolve_main_session_id.return_value = "main-main"

        entry = SubagentRunRecord(
            run_id="run-3",
            child_session_key="agent:main:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
            label="summary",
            result_summary="all good",
            outcome="completed",
            started_at=1.0,
            ended_at=3.0,
        )

        with (
            patch("config.get_config", return_value={"app": {"locale": "en"}}),
            patch("sessions.session_manager.session_manager", session_manager),
            patch("subagents.subagent_registry.registry", registry),
            patch("infra.event_bus.event_bus", event_bus),
        ):
            delivered = await delivery.deliver_recovered_run("run-3", entry)

        self.assertTrue(delivered)
        self.assertEqual(len(dispatcher.submitted), 1)
        work_item = dispatcher.submitted[0]
        self.assertEqual(work_item.kind, "announce")
        self.assertEqual(work_item.run_id, "run-3")
        self.assertIn("summary", work_item.content)

    async def test_recovered_run_success_callback_marks_delivered(self) -> None:
        dispatcher = _FakeDispatcher()
        registry = Mock()
        event_bus = Mock()
        session_manager = Mock()
        delivery = self._delivery(dispatcher)
        session_manager.session_id_from_session_key.return_value = ("main", "main-main")
        session_manager.resolve_main_session_id.return_value = "main-main"

        entry = SubagentRunRecord(
            run_id="run-4",
            child_session_key="agent:main:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
            result_summary="all good",
        )

        with (
            patch("config.get_config", return_value={"app": {"locale": "en"}}),
            patch("sessions.session_manager.session_manager", session_manager),
            patch("subagents.subagent_registry.registry", registry),
            patch("infra.event_bus.event_bus", event_bus),
        ):
            await delivery.deliver_recovered_run("run-4", entry)
            work_item = dispatcher.submitted[0]
            self.assertIsNotNone(work_item.on_success)
            work_item.on_success()

        registry.mark_result_delivery_delivered.assert_called_once_with("run-4")
        self.assertTrue(event_bus.emit.called)

    async def test_deliver_to_requester_uses_injected_dependencies(self) -> None:
        session_manager = Mock()
        session_manager.session_id_from_session_key.return_value = (
            "main",
            "main-main",
        )
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()
        registry = Mock()
        event_bus = Mock()
        delivery = SubagentAnnounceDelivery(
            session_manager=session_manager,
            work_delivery=work_delivery,
            registry=registry,
            event_bus=event_bus,
        )

        with patch("config.get_config", return_value={"app": {"locale": "en"}}):
            await delivery.deliver_to_requester(
                requester_key="agent:main:main",
                child_session_key="agent:main:subagent:child-1",
                run_id="run-injected",
                task="check dependencies",
                result="done",
            )

        work_delivery.deliver.assert_called_once()
        self.assertEqual(
            work_delivery.deliver.call_args.kwargs["kind"],
            "announce",
        )
        registry.set_result_delivery_state.assert_called_once_with(
            "run-injected",
            "queued",
        )
        self.assertTrue(event_bus.emit.called)


if __name__ == "__main__":
    unittest.main()
