"""Session work delivery.

统一投递系统消息类 SessionWorkItem。

这里先只承接 announce / heartbeat / cron 这类系统工作项，
用户消息仍保留 UserTurnService / UserTurnCoordinator 的专属路径。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from sessions.session_dispatcher import SessionWorkItem, dispatcher_manager
from sessions.session_lock_manager import session_lock_manager
from sessions.session_work_store import SessionWorkRecord, session_work_store


class SessionWorkDelivery:
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
        session_lock = session_lock_manager.get_lock(record.agent_id, record.session_id)
        dispatcher = dispatcher_manager.get(record.agent_id, record.session_id, session_lock.lock)
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
        record = session_work_store.create_record(
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
        session_work_store.insert(record)
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

    def recover_pending_work(self) -> int:
        recovered = 0
        for record in session_work_store.get_recoverable_pending():
            if not session_work_store.requeue_for_recovery(record.id):
                continue
            record.status = "queued"
            record.started_at_ms = None
            record.finished_at_ms = None
            record.last_error = None
            self._submit_record(record)
            recovered += 1
        return recovered


session_work_delivery = SessionWorkDelivery()
