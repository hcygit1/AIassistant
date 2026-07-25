"""Composition boundary for per-session dispatcher instances."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sessions.session_dispatcher import (
        SessionDispatcher,
        SystemStream,
        UserStream,
    )
    from sessions.session_work_store import SessionWorkStore

DispatcherType = Callable[..., "SessionDispatcher"]


class SessionDispatcherFactory:
    def __init__(
        self,
        *,
        work_store: "SessionWorkStore",
        system_stream: "SystemStream | None" = None,
        user_stream: "UserStream | None" = None,
        turn_coordinator: Any | None = None,
        dispatcher_type: DispatcherType | None = None,
    ) -> None:
        self._work_store = work_store
        self._system_stream = system_stream
        self._user_stream = user_stream
        self._turn_coordinator = turn_coordinator
        self._dispatcher_type = dispatcher_type

    @property
    def work_store(self) -> "SessionWorkStore":
        return self._work_store

    def create(self, *, lock: asyncio.Lock) -> "SessionDispatcher":
        dispatcher_type = self._dispatcher_type
        if dispatcher_type is None:
            from sessions.session_dispatcher import SessionDispatcher

            dispatcher_type = SessionDispatcher
        return dispatcher_type(
            lock=lock,
            work_store=self._work_store,
            system_stream=self._system_stream,
            user_stream=self._user_stream,
            turn_coordinator=self._turn_coordinator,
        )
