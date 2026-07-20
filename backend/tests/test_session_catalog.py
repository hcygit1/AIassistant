from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_catalog import SessionCatalog
from sessions.session_repository import SessionRepository


class SessionCatalogTests(unittest.TestCase):
    def test_lists_repository_sessions_and_filters_by_requester(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repository = SessionRepository(
                resolve_sessions_dir=lambda _agent_id: root,
            )
            repository.save_session(
                "agent-1-main",
                "agent-1",
                {
                    "session_id": "agent-1-main",
                    "created_at": 1,
                    "updated_at": 1,
                    "messages": [],
                    "label": "主会话",
                },
            )
            repository.save_session(
                "subagent-1",
                "agent-1",
                {
                    "session_id": "subagent-1",
                    "created_at": 2,
                    "updated_at": 3,
                    "messages": [],
                    "spawned_by": "agent:agent-1:main",
                },
            )
            catalog = SessionCatalog(
                repository=repository,
                load_store=repository.load_index,
                save_store=repository.save_index,
                load_session=lambda session_id, agent_id: repository.load_session(
                    session_id,
                    agent_id,
                ),
                derive_title=lambda data, **_kwargs: data.get("label", "未命名"),
                resolve_main_session_id=lambda agent_id: f"{agent_id}-main",
                session_key_from_session_id=lambda agent_id, session_id: (
                    f"agent:{agent_id}:main"
                    if session_id == f"{agent_id}-main"
                    else f"agent:{agent_id}:subagent:{session_id}"
                ),
                resolve_requester=lambda _session_key: None,
                run_maintenance=lambda agent_id, store=None, **_kwargs: (
                    store or {}, {}
                ),
            )

            sessions = catalog.list_sessions("agent-1")
            children = catalog.list_sessions(
                "agent-1",
                spawned_by_session_key="agent:agent-1:main",
            )

        self.assertEqual(
            [session["session_id"] for session in sessions],
            ["subagent-1", "agent-1-main"],
        )
        self.assertEqual([session["session_id"] for session in children], ["subagent-1"])


if __name__ == "__main__":
    unittest.main()
