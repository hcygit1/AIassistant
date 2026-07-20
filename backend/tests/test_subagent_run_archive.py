from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_registry import SubagentRunRecord
from subagents.subagent_run_archive import SubagentRunArchiveService
from subagents.subagent_run_store import SubagentRunStore


class SubagentRunArchiveServiceTests(unittest.TestCase):
    def test_sweep_removes_only_expired_terminal_runs(self) -> None:
        expired = SubagentRunRecord(
            run_id="expired",
            child_session_key="agent:worker:subagent:expired",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="expired",
            ended_at=90.0,
            archive_at_ms=1_000.0,
        )
        active = SubagentRunRecord(
            run_id="active",
            child_session_key="agent:worker:subagent:active",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="active",
            archive_at_ms=1_000.0,
        )
        recent = SubagentRunRecord(
            run_id="recent",
            child_session_key="agent:worker:subagent:recent",
            requester_session_key="agent:main:main",
            requester_agent_id="main",
            target_agent_id="worker",
            task="recent",
            ended_at=99.0,
            archive_at_ms=3_000.0,
        )
        records = {
            item.run_id: item
            for item in (expired, active, recent)
        }
        store = SubagentRunStore(
            load_runs=lambda: records,
            save_runs=lambda _runs: None,
        )
        store.restore()
        state = Mock()
        persist = Mock()
        emit = Mock()
        on_expire = Mock()
        service = SubagentRunArchiveService(
            store=store,
            state=state,
            persist=persist,
            now=lambda: 2.0,
            emit_event=emit,
        )

        removed = service.sweep_expired(on_expire=on_expire)

        self.assertEqual(removed, 1)
        self.assertEqual(
            set(store.records),
            {"active", "recent"},
        )
        state.mark_archived.assert_called_once_with(expired)
        persist.assert_called_once_with()
        emit.assert_called_once_with("expired", expired)
        on_expire.assert_called_once_with(expired)


if __name__ == "__main__":
    unittest.main()
