"""Composition boundary for the session work runtime object graph."""

from __future__ import annotations

from dataclasses import dataclass

from sessions.session_dispatcher import (
    DispatcherManager,
    dispatcher_manager as default_dispatcher_manager,
)
from sessions.session_lock_manager import (
    SessionLockManager,
    session_lock_manager,
)
from sessions.session_work_store import (
    SessionWorkStore,
    session_work_store,
)


@dataclass(frozen=True, slots=True)
class SessionWorkRuntime:
    """A consistent Store, LockManager and DispatcherManager composition."""

    work_store: SessionWorkStore
    lock_manager: SessionLockManager
    dispatcher_manager: DispatcherManager

    def __post_init__(self) -> None:
        dispatcher_store = getattr(
            self.dispatcher_manager,
            "work_store",
            None,
        )
        if (
            dispatcher_store is not None
            and dispatcher_store is not self.work_store
        ):
            raise ValueError(
                "work_store must match dispatcher_manager.work_store"
            )
        dispatcher_lock_manager = getattr(
            self.dispatcher_manager,
            "lock_manager",
            None,
        )
        if (
            dispatcher_lock_manager is not None
            and dispatcher_lock_manager is not self.lock_manager
        ):
            raise ValueError(
                "lock_manager must match dispatcher_manager.lock_manager"
            )

    @classmethod
    def resolve(
        cls,
        *,
        work_store: SessionWorkStore | None = None,
        lock_manager: SessionLockManager | None = None,
        dispatcher_manager: DispatcherManager | None = None,
    ) -> "SessionWorkRuntime":
        if dispatcher_manager is not None:
            inherited_store = getattr(
                dispatcher_manager,
                "work_store",
                None,
            )
            if work_store is not None:
                resolved_store = work_store
            elif inherited_store is not None:
                resolved_store = inherited_store
            else:
                resolved_store = session_work_store
            inherited_lock_manager = getattr(
                dispatcher_manager,
                "lock_manager",
                None,
            )
            if lock_manager is not None:
                resolved_lock_manager = lock_manager
            elif inherited_lock_manager is not None:
                resolved_lock_manager = inherited_lock_manager
            else:
                resolved_lock_manager = session_lock_manager
            return cls(
                work_store=resolved_store,
                lock_manager=resolved_lock_manager,
                dispatcher_manager=dispatcher_manager,
            )

        resolved_store = (
            work_store if work_store is not None else session_work_store
        )
        resolved_lock_manager = (
            lock_manager
            if lock_manager is not None
            else session_lock_manager
        )
        if work_store is None and lock_manager is None:
            resolved_dispatcher_manager = default_dispatcher_manager
        else:
            resolved_dispatcher_manager = DispatcherManager(
                work_store=resolved_store,
                lock_manager=resolved_lock_manager,
            )
        return cls(
            work_store=resolved_store,
            lock_manager=resolved_lock_manager,
            dispatcher_manager=resolved_dispatcher_manager,
        )


session_work_runtime = SessionWorkRuntime(
    work_store=session_work_store,
    lock_manager=session_lock_manager,
    dispatcher_manager=default_dispatcher_manager,
)
