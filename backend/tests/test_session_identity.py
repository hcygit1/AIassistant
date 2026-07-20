from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_identity import (
    main_session_key,
    resolve_main_session_id,
    session_id_from_session_key,
    session_key_from_session_id,
)


class SessionIdentityTests(unittest.TestCase):
    def test_builds_main_and_subagent_session_references(self) -> None:
        self.assertEqual(resolve_main_session_id("agent-1"), "agent-1-main")
        self.assertEqual(main_session_key("agent-1"), "agent:agent-1:main")
        self.assertEqual(
            session_key_from_session_id("agent-1", "agent-1-main"),
            "agent:agent-1:main",
        )
        self.assertEqual(
            session_key_from_session_id("agent-1", "subagent-1"),
            "agent:agent-1:subagent:subagent-1",
        )

    def test_preserves_existing_session_key_parsing(self) -> None:
        self.assertEqual(
            session_id_from_session_key("agent:agent-1:main"),
            ("agent-1", "agent-1-main"),
        )
        self.assertEqual(
            session_id_from_session_key(
                "agent:agent-1:subagent:subagent-1"
            ),
            ("agent-1", "subagent-1"),
        )
        self.assertEqual(
            session_id_from_session_key("Agent:agent-1:legacy:part"),
            ("agent-1", "legacy:part"),
        )
        self.assertIsNone(session_id_from_session_key("agent:agent-1"))
        self.assertIsNone(session_id_from_session_key("owner:agent-1:main"))

    def test_session_manager_keeps_identity_compatibility_aliases(self) -> None:
        from sessions.session_manager import SessionManager

        self.assertIs(
            SessionManager.resolve_main_session_id,
            resolve_main_session_id,
        )
        self.assertIs(
            SessionManager.session_key_from_session_id,
            session_key_from_session_id,
        )
        self.assertIs(
            SessionManager.session_id_from_session_key,
            session_id_from_session_key,
        )


if __name__ == "__main__":
    unittest.main()
