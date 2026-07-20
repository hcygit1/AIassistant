from __future__ import annotations

import sys
import asyncio
import unittest
from pathlib import Path
from unittest.mock import Mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_dispatcher import (
    DispatcherManager,
    PRIORITY_CRON,
    SessionWorkItem,
)
from sessions.session_lock_manager import (
    SessionLockManager,
    cleanup_session_runtime,
)


class _DispatcherManager:
    def __init__(
        self,
        calls: list[tuple],
        lock_manager: _LockManager | None = None,
    ) -> None:
        self.calls = calls
        self.lock_manager = lock_manager

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

    def test_cleanup_inherits_dispatcher_lock_manager(self) -> None:
        calls: list[tuple] = []
        lock_manager = _LockManager(calls)
        dispatcher_manager = _DispatcherManager(calls, lock_manager)

        cleanup_session_runtime(
            "main",
            "main-main",
            dispatcher_manager=dispatcher_manager,
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


class SessionRuntimeAsyncCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_keeps_lock_until_dispatcher_exits(self) -> None:
        started = asyncio.Event()
        cancellation_started = asyncio.Event()
        release_cancellation = asyncio.Event()
        runtime_cleared = asyncio.Event()

        async def blocking_stream(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_started.set()
                await release_cancellation.wait()
                raise
            yield {"type": "done", "content": "unexpected"}

        class RecordingLockManager(SessionLockManager):
            def __init__(self) -> None:
                super().__init__()
                self.cleaned: list[tuple[str, str]] = []

            def cleanup(self, agent_id: str, session_id: str) -> None:
                self.cleaned.append((agent_id, session_id))
                super().cleanup(agent_id, session_id)

        class RecordingCoordinator:
            def clear_session(self, agent_id: str, session_id: str) -> None:
                runtime_cleared.set()

        lock_manager = RecordingLockManager()
        manager = DispatcherManager(
            work_store=Mock(mark_running=Mock(return_value=True)),
            lock_manager=lock_manager,
            system_stream=blocking_stream,
        )
        session_lock = lock_manager.get_lock("main", "main-main")
        dispatcher = manager.get("main", "main-main")
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="running",
                agent_id="main",
                session_id="main-main",
                work_id="work-running",
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        try:
            cleanup_session_runtime(
                "main",
                "main-main",
                dispatcher_manager=manager,
                lock_manager=lock_manager,
                turn_coordinator=RecordingCoordinator(),
            )
            await asyncio.wait_for(cancellation_started.wait(), timeout=1)

            self.assertEqual(lock_manager.cleaned, [])
            self.assertIs(
                lock_manager.get_lock("main", "main-main"),
                session_lock,
            )

            release_cancellation.set()
            await asyncio.wait_for(runtime_cleared.wait(), timeout=1)
        finally:
            release_cancellation.set()
            await asyncio.sleep(0)

        self.assertEqual(
            lock_manager.cleaned,
            [("main", "main-main")],
        )


if __name__ == "__main__":
    unittest.main()
