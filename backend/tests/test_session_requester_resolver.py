from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_requester_resolver import SessionRequesterResolver


class SessionRequesterResolverTests(unittest.TestCase):
    def test_supports_late_provider_binding(self) -> None:
        resolver = SessionRequesterResolver()
        provider = Mock(return_value=("agent:parent:main", "run-1"))

        self.assertIsNone(resolver("agent:worker:subagent:child-1"))

        resolver.bind(provider)

        self.assertEqual(
            resolver("agent:worker:subagent:child-1"),
            ("agent:parent:main", "run-1"),
        )
        provider.assert_called_once_with(
            "agent:worker:subagent:child-1"
        )

    def test_provider_failure_preserves_none_fallback(self) -> None:
        provider = Mock(side_effect=RuntimeError("registry unavailable"))
        resolver = SessionRequesterResolver(provider)

        self.assertIsNone(
            resolver("agent:worker:subagent:child-1")
        )


if __name__ == "__main__":
    unittest.main()
