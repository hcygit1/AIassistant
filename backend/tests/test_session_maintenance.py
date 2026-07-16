from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_maintenance import SessionMaintenanceService
from sessions.session_repository import SessionRepository


class SessionMaintenanceServiceTests(unittest.TestCase):
    def test_prune_archives_expired_subsession_but_keeps_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            for session_id in ("agent-1-main", "subagent-old", "subagent-new"):
                repository.save_session(
                    session_id,
                    "agent-1",
                    {"messages": [], "session_id": session_id},
                )
            repository.save_index(
                "agent-1",
                {
                    "agent:agent-1:main": {
                        "sessionId": "agent-1-main",
                        "updatedAt": 1,
                    },
                    "agent:agent-1:subagent:subagent-old": {
                        "sessionId": "subagent-old",
                        "updatedAt": 1,
                    },
                    "agent:agent-1:subagent:subagent-new": {
                        "sessionId": "subagent-new",
                        "updatedAt": 99_000_000,
                    },
                },
            )
            service = SessionMaintenanceService(
                repository=repository,
                get_config=lambda: {
                    "session": {
                        "maintenance": {
                            "mode": "enforce",
                            "pruneAfter": "1d",
                            "maxEntries": 10,
                        }
                    }
                },
                now_ms=lambda: 100_000_000,
            )

            store, report = service.run("agent-1", enforce=True)

            self.assertEqual(report["pruned"], 1)
            self.assertIn("agent:agent-1:main", store)
            self.assertNotIn(
                "agent:agent-1:subagent:subagent-old",
                store,
            )
            self.assertIsNone(
                repository.load_session("subagent-old", "agent-1")
            )
            self.assertEqual(
                len(list((root / "archive").glob("subagent-old.deleted.*.json"))),
                1,
            )

    def test_dry_run_reports_without_mutating_files_or_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            repository.save_session(
                "subagent-old",
                "agent-1",
                {"messages": []},
            )
            original = {
                "agent:agent-1:subagent:subagent-old": {
                    "sessionId": "subagent-old",
                    "updatedAt": 1,
                }
            }
            repository.save_index("agent-1", original)
            service = SessionMaintenanceService(
                repository=repository,
                get_config=lambda: {
                    "session": {
                        "maintenance": {"pruneAfter": "1d"}
                    }
                },
                now_ms=lambda: 100_000_000,
            )

            _, report = service.run("agent-1", dry_run=True)

            self.assertEqual(report["pruned"], 1)
            self.assertEqual(repository.load_index("agent-1"), original)
            self.assertTrue(
                repository.session_file_exists("subagent-old", "agent-1")
            )

    def test_disk_budget_cleanup_releases_deleted_session_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = SessionRepository(
                resolve_sessions_dir=lambda agent_id: root
            )
            repository.save_session(
                "subagent-old",
                "agent-1",
                {"messages": [{"content": "x" * 200}]},
            )
            repository.save_index(
                "agent-1",
                {
                    "agent:agent-1:subagent:subagent-old": {
                        "sessionId": "subagent-old",
                        "updatedAt": 100_000_000,
                    }
                },
            )
            service = SessionMaintenanceService(
                repository=repository,
                get_config=lambda: {
                    "session": {
                        "maintenance": {
                            "mode": "enforce",
                            "pruneAfter": "30d",
                            "maxEntries": 10,
                            "maxDiskBytes": "1b",
                            "highWaterBytes": "1b",
                        }
                    }
                },
                now_ms=lambda: 100_000_000,
            )

            with patch(
                "sessions.session_lock_manager.cleanup_session_runtime"
            ) as cleanup:
                service.run("agent-1", enforce=True)

        cleanup.assert_called_once_with("agent-1", "subagent-old")


if __name__ == "__main__":
    unittest.main()
