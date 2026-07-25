from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_dispatcher import (
    DispatcherManager as CompatibilityManager,
)
from sessions.session_dispatcher import (
    dispatcher_manager as compatibility_manager,
)
from sessions.session_dispatcher_manager import (
    DispatcherManager,
    dispatcher_manager,
)


class DispatcherManagerBoundaryTests(unittest.TestCase):
    def test_dispatcher_keeps_manager_compatibility_exports(self) -> None:
        self.assertIs(CompatibilityManager, DispatcherManager)
        self.assertIs(compatibility_manager, dispatcher_manager)

    def test_single_session_dispatcher_no_longer_owns_manager_class(self) -> None:
        dispatcher_source = (
            BACKEND_DIR / "sessions" / "session_dispatcher.py"
        ).read_text(encoding="utf-8")
        manager_source = (
            BACKEND_DIR / "sessions" / "session_dispatcher_manager.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("class DispatcherManager:", dispatcher_source)
        self.assertIn("class DispatcherManager:", manager_source)


if __name__ == "__main__":
    unittest.main()
