from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.task_store import TaskKind, TaskStatus
from system_messages.heartbeat_history import (
    HeartbeatEvent,
    HeartbeatHistory,
    emit_heartbeat_event,
    get_heartbeat_history,
)


class HeartbeatHistoryTests(unittest.TestCase):
    def test_heartbeat_module_keeps_history_compatibility_exports(self) -> None:
        from system_messages import heartbeat

        self.assertIs(heartbeat.HeartbeatEvent, HeartbeatEvent)
        self.assertIs(heartbeat.emit_heartbeat_event, emit_heartbeat_event)
        self.assertIs(heartbeat.get_heartbeat_history, get_heartbeat_history)

    def test_history_is_bounded_and_returns_newest_events_first(self) -> None:
        task_store = Mock()
        history = HeartbeatHistory(
            task_store=task_store,
            max_events_resolver=lambda: 2,
            id_factory=iter(("event-1", "event-2", "event-3")).__next__,
        )
        history.emit(
            "main",
            HeartbeatEvent(ts=1, status="ok-empty", agent_id="main"),
        )
        history.emit(
            "main",
            HeartbeatEvent(
                ts=2,
                status="skipped",
                reason="quiet-hours",
                agent_id="main",
            ),
        )
        history.emit(
            "main",
            HeartbeatEvent(
                ts=3,
                status="sent",
                preview="hello",
                duration_ms=5,
                agent_id="main",
            ),
        )

        self.assertEqual(
            history.get("main", limit=30),
            [
                {
                    "ts": 3,
                    "status": "sent",
                    "reason": None,
                    "preview": "hello",
                    "duration_ms": 5,
                },
                {
                    "ts": 2,
                    "status": "skipped",
                    "reason": "quiet-hours",
                    "preview": None,
                    "duration_ms": None,
                },
            ],
        )

    def test_emit_projects_heartbeat_result_to_task_history(self) -> None:
        task_store = Mock()
        history = HeartbeatHistory(
            task_store=task_store,
            id_factory=lambda: "event-1",
        )
        event = HeartbeatEvent(
            ts=1_000,
            status="failed",
            reason="model failed",
            preview="partial",
            duration_ms=25,
            agent_id="main",
        )

        history.emit("main", event)

        record = task_store.insert.call_args.args[0]
        self.assertEqual(record.id, "event-1")
        self.assertEqual(record.kind, TaskKind.HEARTBEAT)
        self.assertEqual(record.status, TaskStatus.FAILED)
        self.assertEqual(record.name, "heartbeat:failed")
        self.assertEqual(record.created_at_ms, 1_000)
        self.assertEqual(record.started_at_ms, 1_000)
        self.assertEqual(record.ended_at_ms, 1_025)
        self.assertEqual(record.duration_ms, 25)
        self.assertEqual(record.preview, "partial")
        self.assertEqual(record.error, "model failed")

    def test_persistence_failure_keeps_in_memory_history(self) -> None:
        task_store = Mock()
        task_store.insert.side_effect = OSError("disk full")
        history = HeartbeatHistory(
            task_store=task_store,
            id_factory=lambda: "event-1",
        )

        history.emit(
            "main",
            HeartbeatEvent(
                ts=1,
                status="skipped",
                reason="session-busy",
                agent_id="main",
            ),
        )

        self.assertEqual(
            history.get("main"),
            [
                {
                    "ts": 1,
                    "status": "skipped",
                    "reason": "session-busy",
                    "preview": None,
                    "duration_ms": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
