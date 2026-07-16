from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRegistry


class SubagentRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(
            SubagentRegistry,
            "_restore_from_disk",
        ):
            self.registry = SubagentRegistry()
        self.registry._persist_to_disk = Mock()
        self.record = self.registry.register_run(
            run_id="run-1",
            child_session_key="agent:worker:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="inspect files",
        )
        self.registry._persist_to_disk.reset_mock()

    def test_list_runs_returns_registry_snapshot(
        self,
    ) -> None:
        runs = self.registry.list_runs()

        self.assertEqual(runs, [self.record])
        runs.clear()
        self.assertEqual(
            self.registry.get_run("run-1"),
            self.record,
        )

    def test_public_records_cannot_mutate_registry_state(
        self,
    ) -> None:
        get_snapshot = self.registry.get_run("run-1")
        list_snapshot = self.registry.list_runs()[0]
        entry_snapshot = self.registry.list_run_entries()[0][1]

        self.assertIsNotNone(get_snapshot)
        get_snapshot.outcome = "tampered"  # type: ignore[union-attr]
        list_snapshot.result_summary = "tampered"
        entry_snapshot.state = "failed"

        current = self.registry.get_run("run-1")
        self.assertIsNotNone(current)
        self.assertIsNone(current.outcome)  # type: ignore[union-attr]
        self.assertIsNone(current.result_summary)  # type: ignore[union-attr]
        self.assertEqual(current.state, "running")  # type: ignore[union-attr]

    def test_public_records_do_not_expose_task_handle(
        self,
    ) -> None:
        task = Mock()
        self.registry.set_task("run-1", task)

        self.assertIsNone(
            self.registry.get_run("run-1").asyncio_task  # type: ignore[union-attr]
        )
        self.assertIsNone(
            self.registry.list_runs()[0].asyncio_task
        )
        self.assertIsNone(
            self.registry.list_run_entries()[0][1].asyncio_task
        )

    def test_list_run_entries_preserves_registry_key(
        self,
    ) -> None:
        with self.registry._lock:
            self.registry._runs["run-1"].run_id = (
                "mismatched-record-id"
            )

        entries = self.registry.list_run_entries()

        self.assertEqual(entries[0][0], "run-1")
        self.assertEqual(
            entries[0][1].run_id,
            "mismatched-record-id",
        )

    def test_public_snapshot_waits_for_registry_lock(
        self,
    ) -> None:
        started = threading.Event()
        finished = threading.Event()

        def _read() -> None:
            started.set()
            self.registry.list_runs()
            finished.set()

        with self.registry._lock:
            thread = threading.Thread(target=_read)
            thread.start()
            self.assertTrue(started.wait(timeout=1))
            self.assertFalse(finished.wait(timeout=0.05))

        thread.join(timeout=1)
        self.assertTrue(finished.is_set())

    def test_remove_waits_for_registry_lock(
        self,
    ) -> None:
        started = threading.Event()
        finished = threading.Event()

        def _remove() -> None:
            started.set()
            self.registry.remove_run("run-1")
            finished.set()

        with self.registry._lock:
            thread = threading.Thread(target=_remove)
            thread.start()
            self.assertTrue(started.wait(timeout=1))
            self.assertFalse(finished.wait(timeout=0.05))

        thread.join(timeout=1)
        self.assertTrue(finished.is_set())

    def test_remove_run_persists_registry_change(
        self,
    ) -> None:
        removed = self.registry.remove_run("run-1")

        self.assertTrue(removed)
        self.assertIsNone(self.registry.get_run("run-1"))
        self.registry._persist_to_disk.assert_called_once_with()

    def test_remove_missing_run_is_noop(
        self,
    ) -> None:
        removed = self.registry.remove_run("missing")

        self.assertFalse(removed)
        self.registry._persist_to_disk.assert_not_called()

    def test_kill_cascades_with_one_persist_and_cancels_outside_lock(
        self,
    ) -> None:
        child = self.registry.register_run(
            run_id="run-2",
            child_session_key="agent:worker:subagent:child-2",
            requester_session_key="agent:worker:child-1",
            requester_agent_id="worker",
            target_agent_id="worker",
            task="inspect child files",
        )

        class ObservedTask:
            def __init__(self) -> None:
                self.cancelled = False
                self.lock_owned_during_cancel = False

            def cancel(inner_self) -> None:
                inner_self.cancelled = True
                inner_self.lock_owned_during_cancel = (
                    self.registry._lock._is_owned()
                )

        parent_task = ObservedTask()
        child_task = ObservedTask()
        self.registry.set_task(self.record.run_id, parent_task)
        self.registry.set_task(child.run_id, child_task)
        self.registry._persist_to_disk.reset_mock()

        killed = self.registry.kill(self.record.run_id)

        self.assertTrue(killed)
        self.registry._persist_to_disk.assert_called_once_with()
        self.assertEqual(
            self.registry.get_run(self.record.run_id).state,  # type: ignore[union-attr]
            "cancelled",
        )
        self.assertEqual(
            self.registry.get_run(child.run_id).state,  # type: ignore[union-attr]
            "cancelled",
        )
        self.assertTrue(parent_task.cancelled)
        self.assertTrue(child_task.cancelled)
        self.assertFalse(parent_task.lock_owned_during_cancel)
        self.assertFalse(child_task.lock_owned_during_cancel)

    def test_terminal_run_is_not_overwritten_by_late_callbacks(
        self,
    ) -> None:
        self.registry.mark_terminated("run-1", "killed")
        self.registry._persist_to_disk.reset_mock()

        self.registry.mark_terminated("run-1", "killed")
        self.registry.mark_completed("run-1", "late result")

        current = self.registry.get_run("run-1")
        self.assertIsNotNone(current)
        self.assertEqual(current.state, "cancelled")  # type: ignore[union-attr]
        self.assertEqual(current.outcome, "killed")  # type: ignore[union-attr]
        self.assertIsNone(current.result_summary)  # type: ignore[union-attr]
        self.registry._persist_to_disk.assert_not_called()

    def test_set_task_cancels_task_when_run_already_ended(
        self,
    ) -> None:
        task = Mock()
        self.registry.mark_terminated("run-1", "killed")

        assigned = self.registry.set_task("run-1", task)

        self.assertFalse(assigned)
        task.cancel.assert_called_once_with()
        with self.registry._lock:
            self.assertIsNone(
                self.registry._runs["run-1"].asyncio_task
            )


if __name__ == "__main__":
    unittest.main()
