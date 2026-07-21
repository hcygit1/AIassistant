from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_manager import SessionManager
from sessions.session_manager_assembly import SessionManagerComponents


class SessionManagerComponentsTests(unittest.TestCase):
    def test_install_on_preserves_manager_compatibility_fields(self) -> None:
        owned = [object() for _ in range(8)]
        components = SessionManagerComponents(*owned)
        manager = SimpleNamespace()

        components.install_on(manager)

        self.assertEqual(
            [
                manager._maintenance,
                manager._title_service,
                manager._reader,
                manager._history_archive,
                manager._catalog,
                manager._persistence,
                manager._message_writer,
                manager._lifecycle,
            ],
            owned,
        )

    def test_assembled_services_keep_dynamic_manager_callbacks(self) -> None:
        repository = Mock()
        manager = SessionManager(repository=repository)
        loaded = {"messages": []}
        manager.load_session = Mock(return_value=loaded)
        manager.run_maintenance = Mock(return_value=({}, {}))

        archive_result = manager._history_archive._load_session(
            "session-1",
            "agent-1",
        )
        manager._catalog._run_maintenance("agent-1", enforce=True)

        self.assertIs(archive_result, loaded)
        manager.load_session.assert_called_once_with(
            "session-1",
            "agent-1",
        )
        manager.run_maintenance.assert_called_once_with(
            "agent-1",
            enforce=True,
        )

    def test_injected_requester_resolver_is_shared_by_session_services(
        self,
    ) -> None:
        resolver = Mock(return_value=("agent:parent:main", "run-1"))
        manager = SessionManager(
            repository=Mock(),
            resolve_requester=resolver,
        )

        catalog_result = manager._catalog._resolve_requester(
            "agent:worker:subagent:child-1"
        )
        writer_result = manager._message_writer._resolve_requester(
            "agent:worker:subagent:child-2"
        )

        self.assertEqual(catalog_result, ("agent:parent:main", "run-1"))
        self.assertEqual(writer_result, ("agent:parent:main", "run-1"))
        self.assertEqual(resolver.call_count, 2)
        self.assertNotIn(
            "_resolve_requester_for_child_session",
            SessionManager.__dict__,
        )


if __name__ == "__main__":
    unittest.main()
