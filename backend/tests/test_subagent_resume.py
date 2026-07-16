from __future__ import annotations

import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRunRecord
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


if __name__ == "__main__":
    unittest.main()
