from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_delivery import SubagentAnnounceDelivery
from subagents.subagent_registry import SubagentRegistry, SubagentRunRecord
from subagents.subagent_run_store import SubagentRunStore
from subagents.subagent_resume import (
    _resolve_orphan_reason,
    resume_subagent_runs,
)


class SubagentResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_orphan_check_uses_public_session_manager_boundary(self) -> None:
        entry = SubagentRunRecord(
            run_id="run-orphan",
            child_session_key="agent:main:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
        )
        manager = Mock(
            spec=[
                "get_session_index_entry",
                "session_file_exists",
            ]
        )
        manager.get_session_index_entry.return_value = {
            "sessionId": "child-1"
        }
        manager.session_file_exists.return_value = False

        with patch(
            "sessions.session_manager.session_manager",
            manager,
        ):
            reason = _resolve_orphan_reason(entry)

        self.assertEqual(reason, "missing-session-file")
        manager.get_session_index_entry.assert_called_once_with(
            "child-1",
            "main",
        )
        manager.session_file_exists.assert_called_once_with(
            "child-1",
            "main",
        )

    async def test_resume_marks_result_delivery_dropped_when_retry_limit_reached(self) -> None:
        entry = SubagentRunRecord(
            run_id="run-resume-1",
            child_session_key="agent:main:subagent:child-1",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
            result_summary="done",
            outcome="completed",
            ended_at=100.0,
            announce_retry_count=3,
        )
        registry = Mock()
        registry.list_run_entries.return_value = [
            ("run-resume-1", entry)
        ]

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch("subagents.subagent_resume._resolve_orphan_reason", return_value=None),
            patch("subagents.subagent_resume._deliver_announce_for_run", return_value=False),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        registry.mark_result_delivery_dropped.assert_not_called()
        registry.remove_run.assert_called_once_with(
            "run-resume-1"
        )

    async def test_resume_marks_result_delivery_dropped_when_retry_fails_after_delivery_attempt(self) -> None:
        entry = SubagentRunRecord(
            run_id="run-resume-2",
            child_session_key="agent:main:subagent:child-2",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
            result_summary="done",
            outcome="completed",
            ended_at=100.0,
            announce_retry_count=1,
        )
        registry = Mock()
        registry.list_run_entries.return_value = [
            ("run-resume-2", entry)
        ]
        registry.mark_announce_retry.return_value = False

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch("subagents.subagent_resume._resolve_orphan_reason", return_value=None),
            patch("subagents.subagent_resume._deliver_announce_for_run", return_value=False),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        registry.mark_result_delivery_dropped.assert_called_once_with("run-resume-2")
        registry.remove_run.assert_called_once_with(
            "run-resume-2"
        )

    async def test_resume_marks_active_run_interrupted_through_registry(self) -> None:
        entry = SubagentRunRecord(
            run_id="run-resume-3",
            child_session_key="agent:main:subagent:child-3",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
        )
        interrupted = replace(
            entry,
            ended_at=101.0,
            outcome="restart-interrupted",
            state="interrupted",
        )
        registry = Mock()
        registry.list_run_entries.return_value = [
            ("run-resume-3", entry)
        ]
        registry.get_run.return_value = interrupted
        registry.mark_announce_retry.return_value = True
        deliver = AsyncMock(return_value=False)

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch("subagents.subagent_resume._resolve_orphan_reason", return_value=None),
            patch("subagents.subagent_resume._deliver_announce_for_run", deliver),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        registry.mark_terminated.assert_called_once_with(
            "run-resume-3",
            "restart-interrupted",
        )
        deliver.assert_awaited_once_with(
            "run-resume-3",
            interrupted,
        )
        registry.mark_announce_retry.assert_called_once_with(
            "run-resume-3"
        )

    async def test_resume_keeps_run_after_recovered_announce_is_queued(self) -> None:
        entry = SubagentRunRecord(
            run_id="run-resume-queued",
            child_session_key="agent:main:subagent:child-queued",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
            result_summary="done",
            outcome="completed",
            ended_at=100.0,
        )
        registry = Mock()
        registry.list_run_entries.return_value = [
            (entry.run_id, entry)
        ]
        deliver = AsyncMock(return_value=True)

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch(
                "subagents.subagent_resume._resolve_orphan_reason",
                return_value=None,
            ),
            patch(
                "subagents.subagent_resume._deliver_announce_for_run",
                deliver,
            ),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        deliver.assert_awaited_once_with(entry.run_id, entry)
        registry.remove_run.assert_not_called()

    async def test_resume_resets_interrupted_delivery_before_requeue(self) -> None:
        for state in ("queued", "delivering", "retrying"):
            with self.subTest(state=state):
                entry = SubagentRunRecord(
                    run_id=f"run-resume-{state}",
                    child_session_key="agent:main:subagent:child-requeue",
                    requester_session_key="agent:main:main",
                    requester_agent_id="main",
                    target_agent_id="worker",
                    task="summarize",
                    result_summary="done",
                    outcome="completed",
                    ended_at=100.0,
                    result_delivery_state=state,
                )
                registry = Mock()
                registry.list_run_entries.return_value = [
                    (entry.run_id, entry)
                ]
                deliver = AsyncMock(return_value=True)

                with (
                    patch("subagents.subagent_resume.registry", registry),
                    patch(
                        "subagents.subagent_resume._resolve_orphan_reason",
                        return_value=None,
                    ),
                    patch(
                        "subagents.subagent_resume._deliver_announce_for_run",
                        deliver,
                    ),
                    patch("time.time", return_value=101.0),
                ):
                    await resume_subagent_runs()

                registry.set_result_delivery_state.assert_called_once_with(
                    entry.run_id,
                    "pending",
                )
                deliver.assert_awaited_once_with(entry.run_id, entry)
                registry.remove_run.assert_not_called()

    async def test_resume_resets_active_run_delivery_after_termination(self) -> None:
        entry = SubagentRunRecord(
            run_id="run-active-delivering",
            child_session_key="agent:main:subagent:child-active",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
        )
        interrupted = replace(
            entry,
            ended_at=101.0,
            outcome="restart-interrupted",
            state="interrupted",
            result_delivery_state="delivering",
        )
        registry = Mock()
        registry.list_run_entries.return_value = [
            (entry.run_id, entry)
        ]
        registry.get_run.return_value = interrupted
        deliver = AsyncMock(return_value=True)

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch(
                "subagents.subagent_resume._resolve_orphan_reason",
                return_value=None,
            ),
            patch(
                "subagents.subagent_resume._deliver_announce_for_run",
                deliver,
            ),
        ):
            await resume_subagent_runs()

        registry.set_result_delivery_state.assert_called_once_with(
            entry.run_id,
            "pending",
        )
        deliver.assert_awaited_once_with(entry.run_id, interrupted)

    async def test_persisted_queued_run_rebuilds_and_requeues(self) -> None:
        persisted: dict[str, SubagentRunRecord] = {}

        def save_runs(runs: dict[str, SubagentRunRecord]) -> None:
            persisted.clear()
            persisted.update(deepcopy(runs))

        first_store = SubagentRunStore(
            load_runs=lambda: {},
            save_runs=save_runs,
        )
        first_registry = SubagentRegistry(store=first_store)
        entry = SubagentRunRecord(
            run_id="run-persisted-requeue",
            child_session_key="agent:main:subagent:child-persisted",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="summarize",
            result_summary="done",
            outcome="completed",
            ended_at=100.0,
            state="succeeded",
        )
        with first_store.locked_records() as records:
            records[entry.run_id] = entry
        first_registry.set_result_delivery_state(entry.run_id, "queued")
        first_registry.set_delivery_work_id(entry.run_id, "work-before-crash")

        rebuilt_store = SubagentRunStore(
            load_runs=lambda: deepcopy(persisted),
            save_runs=save_runs,
        )
        rebuilt_registry = SubagentRegistry(store=rebuilt_store)
        session_manager = Mock()
        session_manager.session_id_from_session_key.return_value = (
            "main",
            "main-main",
        )
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()

        def enqueue(**kwargs):
            kwargs["on_record_created"](Mock(id="work-after-restart"))
            return 1

        work_delivery.deliver.side_effect = enqueue
        delivery = SubagentAnnounceDelivery(
            session_manager=session_manager,
            work_delivery=work_delivery,
            registry=rebuilt_registry,
            event_bus=Mock(),
        )

        with (
            patch("subagents.subagent_resume.registry", rebuilt_registry),
            patch(
                "subagents.subagent_resume._resolve_orphan_reason",
                return_value=None,
            ),
            patch(
                "subagents.subagent_resume._deliver_announce_for_run",
                delivery.deliver_recovered_run,
            ),
            patch("config.get_config", return_value={"app": {"locale": "en"}}),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        recovered = rebuilt_registry.get_run(entry.run_id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.result_delivery_state, "queued")
        self.assertEqual(recovered.delivery_work_id, "work-after-restart")
        work_delivery.deliver.assert_called_once()


if __name__ == "__main__":
    unittest.main()
