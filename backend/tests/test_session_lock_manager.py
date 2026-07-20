from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_lock_manager import cleanup_session_runtime


class _DispatcherManager:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def cleanup(self, agent_id: str, session_id: str) -> None:
        self.calls.append(("dispatcher", agent_id, session_id))


class _LockManager:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def cleanup(self, agent_id: str, session_id: str) -> None:
        self.calls.append(("lock", agent_id, session_id))


class _TurnCoordinator:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def clear_session(self, agent_id: str, session_id: str) -> None:
        self.calls.append(("turn", agent_id, session_id))


class SessionRuntimeCleanupTests(unittest.TestCase):
    def test_cleanup_uses_injected_runtime_components(self) -> None:
        calls: list[tuple] = []

        cleanup_session_runtime(
            "main",
            "main-main",
            dispatcher_manager=_DispatcherManager(calls),
            lock_manager=_LockManager(calls),
            turn_coordinator=_TurnCoordinator(calls),
        )

        self.assertEqual(
            calls,
            [
                ("dispatcher", "main", "main-main"),
                ("lock", "main", "main-main"),
                ("turn", "main", "main-main"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
