"""Lifecycle and cache management for per-session dispatchers."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from sessions.session_dispatcher_factory import SessionDispatcherFactory

if TYPE_CHECKING:
    from sessions.session_dispatcher import (
        SessionDispatcher,
        SystemStream,
        UserStream,
    )
    from sessions.session_lock_manager import SessionLockManager
    from sessions.session_work_store import SessionWorkStore


logger = logging.getLogger(__name__)


def _default_work_store() -> SessionWorkStore:
    from sessions.session_work_store import session_work_store

    return session_work_store


def _default_lock_manager() -> SessionLockManager:
    from sessions.session_lock_manager import session_lock_manager

    return session_lock_manager


def _safe_call(callback: Callable[[], Any]) -> None:
    try:
        callback()
    except Exception as exc:
        logger.warning("Dispatcher callback error: %s", exc)


class DispatcherManager:
    """Cache and close one dispatcher for each agent/session pair."""

    def __init__(
        self,
        work_store: SessionWorkStore | None = None,
        system_stream: SystemStream | None = None,
        user_stream: UserStream | None = None,
        turn_coordinator: Any | None = None,
        lock_manager: SessionLockManager | None = None,
        dispatcher_factory: SessionDispatcherFactory | None = None,
    ) -> None:
        self._dispatchers: dict[str, SessionDispatcher] = {}
        if dispatcher_factory is not None and any(
            dependency is not None
            for dependency in (
                work_store,
                system_stream,
                user_stream,
                turn_coordinator,
            )
        ):
            raise ValueError(
                "dispatcher_factory cannot be combined with legacy "
                "dispatcher dependencies"
            )
        self._lock_manager = (
            lock_manager if lock_manager is not None else _default_lock_manager()
        )
        if dispatcher_factory is None:
            resolved_store = (
                work_store if work_store is not None else _default_work_store()
            )
            dispatcher_factory = SessionDispatcherFactory(
                work_store=resolved_store,
                system_stream=system_stream,
                user_stream=user_stream,
                turn_coordinator=turn_coordinator,
            )
        self._dispatcher_factory = dispatcher_factory
        self._closing: dict[str, asyncio.Task | None] = {}

    @property
    def work_store(self) -> SessionWorkStore:
        return self._dispatcher_factory.work_store

    @property
    def lock_manager(self) -> SessionLockManager:
        return self._lock_manager

    def get(
        self,
        agent_id: str,
        session_id: str,
        lock: asyncio.Lock | None = None,
    ) -> SessionDispatcher:
        key = f"{agent_id}:{session_id}"
        if key not in self._dispatchers:
            if lock is None:
                lock = self._lock_manager.get_lock(agent_id, session_id).lock
            dispatcher = self._dispatcher_factory.create(lock=lock)
            dispatcher.start()
            self._dispatchers[key] = dispatcher
        return self._dispatchers[key]

    def cleanup(
        self,
        agent_id: str,
        session_id: str,
    ) -> asyncio.Task | None:
        return self.cleanup_when_closed(agent_id, session_id)

    def cleanup_when_closed(
        self,
        agent_id: str,
        session_id: str,
        *,
        on_closed: Callable[[], Any] | None = None,
    ) -> asyncio.Task | None:
        key = f"{agent_id}:{session_id}"
        dispatcher = self._dispatchers.get(key)
        if dispatcher is None:
            if on_closed:
                _safe_call(on_closed)
            return None

        if key in self._closing:
            task = self._closing[key]
            if task is not None and on_closed:
                task.add_done_callback(lambda _task: _safe_call(on_closed))
            return task

        task = dispatcher.stop()
        self._closing[key] = task

        def finish(_task: asyncio.Task | None = None) -> None:
            if self._dispatchers.get(key) is dispatcher:
                self._dispatchers.pop(key, None)
            self._closing.pop(key, None)
            if on_closed:
                _safe_call(on_closed)

        if task is None or task.done():
            finish(task)
        else:
            task.add_done_callback(finish)
        return task

    async def aclose_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> None:
        closed = asyncio.Event()
        self.cleanup_when_closed(
            agent_id,
            session_id,
            on_closed=closed.set,
        )
        await closed.wait()

    def cancel_work(
        self,
        agent_id: str,
        session_id: str,
        work_id: str,
    ) -> bool:
        dispatcher = self._dispatchers.get(f"{agent_id}:{session_id}")
        if dispatcher is None:
            return False
        return dispatcher.cancel_work(work_id)


dispatcher_manager = DispatcherManager()
