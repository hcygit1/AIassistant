from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_dispatcher import DispatcherManager, dispatcher_manager
from sessions.session_lock_manager import SessionLockManager, session_lock_manager
from sessions.session_work_runtime import (
    SessionWorkRuntime,
    session_work_runtime,
)
from sessions.session_work_store import SessionWorkStore, session_work_store


class SessionWorkRuntimeTests(unittest.TestCase):
    def test_default_runtime_reuses_compatibility_globals(self) -> None:
        self.assertIs(session_work_runtime.work_store, session_work_store)
        self.assertIs(session_work_runtime.lock_manager, session_lock_manager)
        self.assertIs(
            session_work_runtime.dispatcher_manager,
            dispatcher_manager,
        )

    def test_resolve_builds_consistent_dispatcher_from_store_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session-work.db")
            lock_manager = SessionLockManager()

            runtime = SessionWorkRuntime.resolve(
                work_store=store,
                lock_manager=lock_manager,
            )

        self.assertIs(runtime.work_store, store)
        self.assertIs(runtime.lock_manager, lock_manager)
        self.assertIs(runtime.dispatcher_manager.work_store, store)
        self.assertIs(runtime.dispatcher_manager.lock_manager, lock_manager)

    def test_resolve_inherits_dispatcher_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session-work.db")
            lock_manager = SessionLockManager()
            dispatcher = DispatcherManager(
                work_store=store,
                lock_manager=lock_manager,
            )

            runtime = SessionWorkRuntime.resolve(
                dispatcher_manager=dispatcher,
            )

        self.assertIs(runtime.work_store, store)
        self.assertIs(runtime.lock_manager, lock_manager)
        self.assertIs(runtime.dispatcher_manager, dispatcher)

    def test_resolve_falls_back_when_dispatcher_dependencies_are_none(self) -> None:
        dispatcher = SimpleNamespace(
            work_store=None,
            lock_manager=None,
        )

        runtime = SessionWorkRuntime.resolve(
            dispatcher_manager=dispatcher,
        )

        self.assertIs(runtime.work_store, session_work_store)
        self.assertIs(runtime.lock_manager, session_lock_manager)
        self.assertIs(runtime.dispatcher_manager, dispatcher)

    def test_runtime_rejects_dispatcher_store_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_store = SessionWorkStore(Path(tmpdir) / "runtime.db")
            dispatcher_store = SessionWorkStore(
                Path(tmpdir) / "dispatcher.db"
            )
            dispatcher = DispatcherManager(work_store=dispatcher_store)

            with self.assertRaisesRegex(
                ValueError,
                "work_store must match dispatcher_manager",
            ):
                SessionWorkRuntime(
                    work_store=runtime_store,
                    lock_manager=dispatcher.lock_manager,
                    dispatcher_manager=dispatcher,
                )


if __name__ == "__main__":
    unittest.main()
