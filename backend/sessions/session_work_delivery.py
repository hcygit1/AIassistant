"""Session work delivery.

统一投递系统消息类 SessionWorkItem。

这里先只承接 announce / heartbeat / cron 这类系统工作项，
用户消息仍保留 UserTurnService / UserTurnCoordinator 的专属路径。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from sessions.session_dispatcher_manager import DispatcherManager
from sessions.session_lock_manager import SessionLockManager
from sessions.session_work_item import SessionWorkItem
from sessions.session_work_runtime import (
    SessionWorkRuntime,
    session_work_runtime,
)
from sessions.session_work_recovery_resolver import (
    session_work_recovery_resolver,
)
from sessions.session_work_record import SessionWorkRecord
from sessions.session_work_store import SessionWorkStore

INTERRUPTED_WORK_ERROR = "interrupted by process restart"
STALE_RECOVERABLE_WORK_ERROR = "stale recoverable work claim"

logger = logging.getLogger(__name__)


class SessionWorkDelivery:
    def __init__(
        self,
        *,
        work_store: SessionWorkStore | None = None,
        dispatcher_manager: DispatcherManager | None = None,
        lock_manager: SessionLockManager | None = None,
        runtime: SessionWorkRuntime | None = None,
        recovery_callback_resolver: (
            Callable[
                [SessionWorkRecord],
                dict[str, Any] | None,
            ] | None
        ) = None,
    ) -> None:
        legacy_dependencies = (
            work_store,
            dispatcher_manager,
            lock_manager,
        )
        if runtime is not None and any(
            dependency is not None for dependency in legacy_dependencies
        ):
            raise ValueError(
                "runtime cannot be combined with legacy dependencies"
            )
        uses_default_runtime = (
            runtime is None
            and work_store is None
            and dispatcher_manager is None
            and lock_manager is None
        )
        self._runtime = runtime or SessionWorkRuntime.resolve(
            work_store=work_store,
            dispatcher_manager=dispatcher_manager,
            lock_manager=lock_manager,
        )
        if recovery_callback_resolver is not None:
            self._recovery_callback_resolver = recovery_callback_resolver
        elif uses_default_runtime:
            self._recovery_callback_resolver = (
                session_work_recovery_resolver.resolve
            )
        else:
            self._recovery_callback_resolver = lambda _record: {}

    @property
    def runtime(self) -> SessionWorkRuntime:
        return self._runtime

    @property
    def work_store(self) -> SessionWorkStore:
        return self.runtime.work_store

    @property
    def dispatcher_manager(self) -> DispatcherManager:
        return self.runtime.dispatcher_manager

    @property
    def lock_manager(self) -> SessionLockManager:
        return self.runtime.lock_manager

    @property
    def recovery_callback_resolver(
        self,
    ) -> Callable[[SessionWorkRecord], dict[str, Any] | None]:
        return self._recovery_callback_resolver

    def _submit_record(
        self,
        record: SessionWorkRecord,
        *,
        result_handler: Callable[[str], Awaitable[None]] | None = None,
        on_success: Callable[[], Any] | None = None,
        on_failure: Callable[[], Any] | None = None,
        on_failure_async: Callable[[Exception], Awaitable[None]] | None = None,
        on_cancel: Callable[[], Any] | None = None,
    ) -> int:
        session_lock = self.lock_manager.get_lock(
            record.agent_id,
            record.session_id,
        )
        dispatcher = self.dispatcher_manager.get(
            record.agent_id,
            record.session_id,
            session_lock.lock,
        )
        return dispatcher.submit(
            SessionWorkItem(
                kind=record.kind,
                priority=record.priority,
                content=record.content,
                agent_id=record.agent_id,
                session_id=record.session_id,
                prompt_mode=record.prompt_mode,
                persist_role=record.persist_role,
                run_id=record.run_id,
                work_id=record.id,
                result_handler=result_handler,
                on_success=on_success,
                on_failure=on_failure,
                on_failure_async=on_failure_async,
                on_cancel=on_cancel,
            )
        )

    def _mark_submission_failed(
        self,
        record: SessionWorkRecord,
        error: Exception,
    ) -> None:
        error_text = str(error).strip() or type(error).__name__
        try:
            self.work_store.mark_failed(record.id, error_text)
        except Exception:
            logger.exception(
                "Failed to mark session work %s after submission error",
                record.id,
            )

    def deliver(
        self,
        *,
        kind: str,
        priority: int,
        content: str,
        agent_id: str,
        session_id: str,
        prompt_mode: str = "minimal",
        persist_role: str = "system",
        run_id: str | None = None,
        result_handler: Callable[[str], Awaitable[None]] | None = None,
        on_success: Callable[[], Any] | None = None,
        on_failure: Callable[[], Any] | None = None,
        on_failure_async: Callable[[Exception], Awaitable[None]] | None = None,
        on_cancel: Callable[[], Any] | None = None,
        on_record_created: Callable[[SessionWorkRecord], Any] | None = None,
        recover_on_restart: bool = False,
    ) -> int:
        record = self.work_store.create_record(
            kind=kind,
            agent_id=agent_id,
            session_id=session_id,
            content=content,
            priority=priority,
            prompt_mode=prompt_mode,
            persist_role=persist_role,
            run_id=run_id,
            recover_on_restart=recover_on_restart,
        )
        self.work_store.insert(record)
        try:
            if on_record_created:
                on_record_created(record)
            return self._submit_record(
                record,
                result_handler=result_handler,
                on_success=on_success,
                on_failure=on_failure,
                on_failure_async=on_failure_async,
                on_cancel=on_cancel,
            )
        except Exception as exc:
            self._mark_submission_failed(record, exc)
            raise

    def recover_pending_work(self) -> int:
        recovered = 0
        for record in self.work_store.get_recoverable_pending():
            if not self.work_store.requeue_for_recovery(record.id):
                continue
            record.status = "queued"
            record.started_at_ms = None
            record.finished_at_ms = None
            record.last_error = None
            callbacks = self._recovery_callback_resolver(record)
            if callbacks is None:
                self.work_store.mark_failed(
                    record.id,
                    STALE_RECOVERABLE_WORK_ERROR,
                )
                continue
            self._submit_record(
                record,
                result_handler=callbacks.get("result_handler"),
                on_success=callbacks.get("on_success"),
                on_failure=callbacks.get("on_failure"),
                on_failure_async=callbacks.get("on_failure_async"),
                on_cancel=callbacks.get("on_cancel"),
            )
            recovered += 1
        return recovered

    def fail_unrecoverable_pending(self) -> int:
        return self.work_store.fail_unrecoverable_pending(
            INTERRUPTED_WORK_ERROR
        )


session_work_delivery = SessionWorkDelivery(
    runtime=session_work_runtime,
    recovery_callback_resolver=session_work_recovery_resolver.resolve,
)
