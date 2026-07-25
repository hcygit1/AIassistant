from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_transitions import SessionWorkTransitions
from sessions.session_work_store import SessionWorkStore


def _normalized_sql(sql: str) -> str:
    return " ".join(sql.split())


class SessionWorkTransitionsTests(unittest.TestCase):
    def test_store_facade_delegates_all_transition_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionWorkStore(Path(tmp_dir) / "session-work.db")
            transitions = Mock()
            transitions.mark_running.return_value = True
            transitions.cancel_queued.return_value = False
            transitions.requeue_for_recovery.return_value = True
            transitions.fail_unrecoverable_pending.return_value = 2
            store._transitions = transitions

            self.assertTrue(store.mark_running("running"))
            self.assertFalse(store.cancel_queued("queued"))
            store.mark_cancelled("cancelled")
            self.assertTrue(store.requeue_for_recovery("recoverable"))
            self.assertEqual(store.fail_unrecoverable_pending("restart"), 2)
            store.mark_done("done")
            store.mark_failed("failed", "disk full")

        transitions.mark_running.assert_called_once_with("running")
        transitions.cancel_queued.assert_called_once_with("queued")
        transitions.mark_cancelled.assert_called_once_with("cancelled")
        transitions.requeue_for_recovery.assert_called_once_with("recoverable")
        transitions.fail_unrecoverable_pending.assert_called_once_with("restart")
        transitions.mark_done.assert_called_once_with("done")
        transitions.mark_failed.assert_called_once_with("failed", "disk full")

    def test_store_default_transition_clock_remains_dynamic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionWorkStore(Path(tmp_dir) / "session-work.db")
            record = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="report",
                priority=1,
            )
            store.insert(record)

            with patch(
                "sessions.session_work_store.time.time",
                return_value=12.345,
            ):
                store.mark_running(record.id)

            updated = store.get(record.id)

        self.assertIsNotNone(updated)
        self.assertEqual(updated.started_at_ms, 12_345)

    def test_mark_running_keeps_claim_guard_and_result(self) -> None:
        execute_update = Mock(return_value=1)
        transitions = SessionWorkTransitions(
            execute_update=execute_update,
            now_ms=lambda: 123,
        )

        claimed = transitions.mark_running("work-1")

        self.assertTrue(claimed)
        sql, params = execute_update.call_args.args
        self.assertEqual(
            _normalized_sql(sql),
            "UPDATE session_work SET status='running', started_at_ms=?, "
            "last_error=NULL WHERE id=? AND status IN ('queued', 'running')",
        )
        self.assertEqual(params, (123, "work-1"))

    def test_cancel_and_recovery_keep_status_guards(self) -> None:
        execute_update = Mock(side_effect=[0, 1, 2])
        transitions = SessionWorkTransitions(
            execute_update=execute_update,
            now_ms=lambda: 456,
        )

        cancelled = transitions.cancel_queued("work-1")
        requeued = transitions.requeue_for_recovery("work-2")
        failed = transitions.fail_unrecoverable_pending("restart")

        self.assertFalse(cancelled)
        self.assertTrue(requeued)
        self.assertEqual(failed, 2)
        calls = execute_update.call_args_list
        self.assertIn("status='queued'", _normalized_sql(calls[0].args[0]))
        self.assertEqual(calls[0].args[1], (456, "work-1"))
        self.assertIn(
            "recover_on_restart=1 AND status IN ('queued', 'running')",
            _normalized_sql(calls[1].args[0]),
        )
        self.assertEqual(calls[1].args[1], ("work-2",))
        self.assertIn(
            "recover_on_restart=0 AND status IN ('queued', 'running')",
            _normalized_sql(calls[2].args[0]),
        )
        self.assertEqual(calls[2].args[1], (456, "restart"))

    def test_terminal_updates_keep_timestamp_and_error_semantics(self) -> None:
        execute_update = Mock(return_value=1)
        transitions = SessionWorkTransitions(
            execute_update=execute_update,
            now_ms=lambda: 789,
        )

        transitions.mark_cancelled("work-cancelled")
        transitions.mark_done("work-done")
        transitions.mark_failed("work-failed", "disk full")

        self.assertEqual(
            [item.args[1] for item in execute_update.call_args_list],
            [
                (789, "work-cancelled"),
                (789, "work-done"),
                (789, "disk full", "work-failed"),
            ],
        )
        self.assertIn(
            "status IN ('queued', 'running')",
            _normalized_sql(execute_update.call_args_list[0].args[0]),
        )
        self.assertIn(
            "last_error=NULL",
            _normalized_sql(execute_update.call_args_list[1].args[0]),
        )
        self.assertIn(
            "last_error=?",
            _normalized_sql(execute_update.call_args_list[2].args[0]),
        )

    def test_store_keeps_only_transaction_execution_and_delegation(self) -> None:
        source = (
            BACKEND_DIR / "sessions" / "session_work_store.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("UPDATE session_work", source)
        self.assertEqual(source.count("self._transitions."), 7)


if __name__ == "__main__":
    unittest.main()
