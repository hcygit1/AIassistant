from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRunRecord
from subagents.subagent_resume import resume_subagent_runs


class SubagentResumeTests(unittest.IsolatedAsyncioTestCase):
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
        registry._runs = {"run-resume-1": entry}

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch("subagents.subagent_resume._resolve_orphan_reason", return_value=None),
            patch("subagents.subagent_resume._deliver_announce_for_run", return_value=False),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        registry.mark_result_delivery_dropped.assert_not_called()
        self.assertNotIn("run-resume-1", registry._runs)
        self.assertTrue(registry._persist_to_disk.called)

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
        registry._runs = {"run-resume-2": entry}
        registry.mark_announce_retry.return_value = False

        with (
            patch("subagents.subagent_resume.registry", registry),
            patch("subagents.subagent_resume._resolve_orphan_reason", return_value=None),
            patch("subagents.subagent_resume._deliver_announce_for_run", return_value=False),
            patch("time.time", return_value=101.0),
        ):
            await resume_subagent_runs()

        registry.mark_result_delivery_dropped.assert_called_once_with("run-resume-2")
        self.assertNotIn("run-resume-2", registry._runs)
        self.assertTrue(registry._persist_to_disk.called)


if __name__ == "__main__":
    unittest.main()
