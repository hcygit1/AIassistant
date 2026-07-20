"""Construction of the SessionManager component graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sessions.session_catalog import SessionCatalog
from sessions.session_history_archive import SessionHistoryArchive
from sessions.session_lifecycle import SessionLifecycleService
from sessions.session_maintenance import SessionMaintenanceService
from sessions.session_message_writer import SessionMessageWriter
from sessions.session_reader import SessionReader
from sessions.session_repository import SessionRepository
from sessions.session_title import SessionTitleService


@dataclass
class SessionManagerComponents:
    maintenance: Any
    title_service: Any
    reader: Any
    history_archive: Any
    catalog: Any
    message_writer: Any
    lifecycle: Any

    def install_on(self, manager: Any) -> None:
        manager._maintenance = self.maintenance
        manager._title_service = self.title_service
        manager._reader = self.reader
        manager._history_archive = self.history_archive
        manager._catalog = self.catalog
        manager._message_writer = self.message_writer
        manager._lifecycle = self.lifecycle


class SessionManagerAssembler:
    def __init__(
        self,
        *,
        repository: SessionRepository,
        get_config: Callable[[], dict[str, Any]],
        maintenance: SessionMaintenanceService | None = None,
        title_service: SessionTitleService | None = None,
        reader: SessionReader | None = None,
        cleanup_runtime: Callable[[str, str], None] | None = None,
    ) -> None:
        self._repository = repository
        self._get_config = get_config
        self._maintenance = maintenance
        self._title_service = title_service
        self._reader = reader
        self._cleanup_runtime = cleanup_runtime

    def build(self, manager: Any) -> SessionManagerComponents:
        maintenance = self._maintenance or SessionMaintenanceService(
            repository=self._repository,
            get_config=self._get_config,
            cleanup_runtime=self._cleanup_runtime,
        )
        title_service = self._title_service or SessionTitleService()
        reader = self._reader or SessionReader(
            repository=self._repository,
            is_bootstrap_text=lambda text: manager._is_bootstrap_text(text),
        )
        history_archive = SessionHistoryArchive(
            repository=self._repository,
            load_session=lambda session_id, agent_id: manager.load_session(
                session_id,
                agent_id,
            ),
            save_session=lambda session_id, agent_id, data: (
                manager._save_session_data(session_id, agent_id, data)
            ),
        )
        catalog = SessionCatalog(
            repository=self._repository,
            load_store=lambda agent_id: manager._load_session_store(agent_id),
            save_store=lambda agent_id, store: manager._save_session_store(
                agent_id,
                store,
            ),
            load_session=lambda session_id, agent_id: manager.load_session(
                session_id,
                agent_id,
            ),
            derive_title=lambda data, **kwargs: manager.derive_session_title(
                data,
                **kwargs,
            ),
            resolve_main_session_id=manager.resolve_main_session_id,
            session_key_from_session_id=manager.session_key_from_session_id,
            resolve_requester=manager._resolve_requester_for_child_session,
            run_maintenance=lambda agent_id, **kwargs: manager.run_maintenance(
                agent_id,
                **kwargs,
            ),
        )
        message_writer = SessionMessageWriter(
            transaction=manager._session_transaction,
            load_session=lambda session_id, agent_id: manager.load_session(
                session_id,
                agent_id,
            ),
            persist_session=lambda session_id, agent_id, data: (
                manager._save_session_data(session_id, agent_id, data)
            ),
            session_key_from_id=manager.session_key_from_session_id,
            resolve_requester=manager._resolve_requester_for_child_session,
        )
        lifecycle = SessionLifecycleService(
            repository=self._repository,
            transaction=manager._session_transaction,
            load_session=lambda session_id, agent_id: manager.load_session(
                session_id,
                agent_id,
            ),
            save_session_data=lambda session_id, agent_id, data: (
                manager._save_session_data(session_id, agent_id, data)
            ),
            remove_index=lambda agent_id, session_key: (
                manager._remove_session_store_entry(agent_id, session_key)
            ),
            session_key_from_id=manager.session_key_from_session_id,
            ensure_session=lambda session_id, agent_id: manager.ensure_session(
                session_id,
                agent_id,
            ),
            cleanup_runtime=self._cleanup_runtime,
        )
        return SessionManagerComponents(
            maintenance=maintenance,
            title_service=title_service,
            reader=reader,
            history_archive=history_archive,
            catalog=catalog,
            message_writer=message_writer,
            lifecycle=lifecycle,
        )
