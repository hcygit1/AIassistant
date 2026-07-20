"""会话管理器 — JSON 文件持久化 + 生命周期事件 + LRU 缓存"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable

from config import get_config, resolve_agent_sessions_dir
from sessions.session_repository import (
    SessionDataCorruptionError,
    SessionRepository,
)
from sessions.session_maintenance import SessionMaintenanceService
from sessions.session_catalog import SessionCatalog
from sessions.session_history_archive import SessionHistoryArchive
from sessions.session_identity import (
    resolve_main_session_id as _resolve_main_session_id,
    session_id_from_session_key as _session_id_from_session_key,
    session_key_from_session_id as _session_key_from_session_id,
)
from sessions.session_lifecycle import SessionLifecycleService
from sessions.session_manager_assembly import SessionManagerAssembler
from sessions.session_message_writer import SessionMessageWriter
from sessions.session_reader import SessionReader
from sessions.session_title import SessionTitleService


class SessionManager:
    def __init__(
        self,
        repository: SessionRepository | None = None,
        maintenance: SessionMaintenanceService | None = None,
        title_service: SessionTitleService | None = None,
        cleanup_runtime: Callable[[str, str], None] | None = None,
        reader: SessionReader | None = None,
    ) -> None:
        self._repository = repository or SessionRepository(
            resolve_sessions_dir=resolve_agent_sessions_dir
        )
        components = SessionManagerAssembler(
            repository=self._repository,
            get_config=get_config,
            maintenance=maintenance,
            title_service=title_service,
            reader=reader,
            cleanup_runtime=cleanup_runtime,
        ).build(self)
        components.install_on(self)

    def _is_bootstrap_text(self, text: str | None) -> bool:
        return self._title_service.is_bootstrap_text(text)

    @staticmethod
    def _resolve_requester_for_child_session(
        child_session_key: str,
    ) -> tuple[str, str] | None:
        try:
            from subagents.subagent_registry import registry

            return registry.resolve_requester_for_child_session(child_session_key)
        except Exception:
            return None

    @contextmanager
    def _session_transaction(self, session_id: str, agent_id: str):
        with self._repository.get_agent_lock(agent_id):
            with self._repository.get_session_lock(session_id, agent_id):
                yield

    resolve_main_session_id = staticmethod(_resolve_main_session_id)
    session_key_from_session_id = staticmethod(_session_key_from_session_id)
    session_id_from_session_key = staticmethod(_session_id_from_session_key)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def session_file_exists(self, session_id: str, agent_id: str) -> bool:
        return self._repository.session_file_exists(session_id, agent_id)

    def load_session(self, session_id: str, agent_id: str) -> dict[str, Any] | None:
        return self._reader.load_session(session_id, agent_id)

    def load_session_for_agent(
        self, session_id: str, agent_id: str
    ) -> list[dict[str, Any]]:
        """
        为 LLM 优化的消息列表：
        - 合并连续 assistant 消息
        """
        return self._reader.load_session_for_agent(session_id, agent_id)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def ensure_session(
        self,
        session_id: str,
        agent_id: str,
        spawned_by: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        return self._message_writer.ensure_session(
            session_id,
            agent_id,
            spawned_by=spawned_by,
            label=label,
        )

    def _ensure_session_locked(
        self,
        session_id: str,
        agent_id: str,
        spawned_by: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        return self._message_writer._ensure_session_locked(
            session_id,
            agent_id,
            spawned_by=spawned_by,
            label=label,
        )

    def rollback_last_turn(self, session_id: str, agent_id: str) -> bool:
        """移除最后一轮 user + assistant 消息（用于心跳 HEARTBEAT_OK 时不持久化）"""
        return self._message_writer.rollback_last_turn(session_id, agent_id)

    def _rollback_last_turn_locked(
        self,
        session_id: str,
        agent_id: str,
    ) -> bool:
        return self._message_writer._rollback_last_turn_locked(
            session_id,
            agent_id,
        )

    def save_message(
        self,
        session_id: str,
        agent_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self._message_writer.save_message(
            session_id,
            agent_id,
            role,
            content,
            tool_calls=tool_calls,
        )

    def _session_store_path(self, agent_id: str) -> Path:
        """sessions.json 索引路径"""
        return self._repository.index_path(agent_id)

    def _load_session_store(self, agent_id: str) -> dict[str, dict[str, Any]]:
        """加载 sessions.json 索引"""
        return self._repository.load_index(agent_id)

    def _save_session_store(
        self, agent_id: str, store: dict[str, dict[str, Any]]
    ) -> None:
        """持久化 sessions.json"""
        self._repository.save_index(agent_id, store)

    def get_session_index_entry(
        self,
        session_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        return self._catalog.get_entry(session_id, agent_id)

    def _update_session_store_entry(
        self,
        agent_id: str,
        session_key: str,
        session_id: str,
        updated_at: float,
        label: str = "",
        spawned_by: str | None = None,
    ) -> None:
        """更新 sessions.json 中的会话条目，并在 mode=enforce 时执行 maintenance"""
        self._catalog.update_entry(
            agent_id,
            session_key,
            session_id,
            updated_at,
            label=label,
            spawned_by=spawned_by,
        )

    def _remove_session_store_entry(self, agent_id: str, session_key: str) -> None:
        """从 sessions.json 移除会话"""
        self._catalog.remove_entry(agent_id, session_key)

    def run_maintenance(
        self,
        agent_id: str,
        store: dict | None = None,
        enforce: bool = False,
        dry_run: bool = False,
    ) -> tuple[dict, dict[str, Any]]:
        return self._maintenance.run(
            agent_id,
            store=store,
            enforce=enforce,
            dry_run=dry_run,
        )

    def _save_session_data(
        self, session_id: str, agent_id: str, data: dict[str, Any]
    ) -> None:
        with self._repository.get_agent_lock(agent_id):
            with self._repository.get_session_lock(session_id, agent_id):
                self._repository.save_session(session_id, agent_id, data)
                session_key = self.session_key_from_session_id(
                    agent_id,
                    session_id,
                )
                self._update_session_store_entry(
                    agent_id,
                    session_key,
                    session_id,
                    data.get("updated_at", time.time()),
                    label=data.get("label", ""),
                    spawned_by=data.get("spawned_by"),
                )

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def list_sessions(
        self,
        agent_id: str,
        spawned_by_session_key: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._catalog.list_sessions(
            agent_id,
            spawned_by_session_key=spawned_by_session_key,
        )

    def create_session(self, agent_id: str, title: str = "新会话") -> str:
        return self._lifecycle.create_session(agent_id, title)

    def rename_session(self, session_id: str, agent_id: str, title: str) -> bool:
        return self._lifecycle.rename_session(session_id, agent_id, title)

    def delete_session(self, session_id: str, agent_id: str) -> bool:
        return self._lifecycle.delete_session(session_id, agent_id)

    # ------------------------------------------------------------------
    # 压缩
    # ------------------------------------------------------------------

    def compress_history(
        self,
        session_id: str,
        agent_id: str,
        n_messages: int,
    ) -> dict[str, int]:
        with self._session_transaction(session_id, agent_id):
            return self._compress_history_locked(
                session_id,
                agent_id,
                n_messages,
            )

    def _compress_history_locked(
        self,
        session_id: str,
        agent_id: str,
        n_messages: int,
    ) -> dict[str, int]:
        """归档旧消息并截断 messages。

        结构化摘要由 MemStore.session_summaries 管理。
        """
        return self._history_archive.compress_history(
            session_id,
            agent_id,
            n_messages,
        )

    # ------------------------------------------------------------------
    # 会话标题推导
    # ------------------------------------------------------------------

    def _get_title_max_len(self) -> int:
        """从配置获取标题最大长度"""
        try:
            cfg = get_config()
            return cfg.get("session", {}).get("titleMaxLen", 60)
        except Exception:
            return 60

    def derive_session_title(
        self, data: dict[str, Any] | None, session_id: str = "", updated_at: float | None = None
    ) -> str:
        return self._title_service.derive(
            data,
            session_id=session_id,
            updated_at=updated_at,
            max_length=self._get_title_max_len(),
        )

    # ------------------------------------------------------------------
    # 会话重置
    # ------------------------------------------------------------------

    def reset_session(self, session_id: str, agent_id: str) -> dict[str, Any]:
        """
        重置会话：归档旧 JSON 并创建空白会话。
        返回 {"archived": bool, "archive_file": str}
        标题：使用 derive_session_title 推导，避免重复追加 (续)
        """
        return self._lifecycle.reset_session(session_id, agent_id)

    # ------------------------------------------------------------------
    # 获取活跃会话 ID
    # ------------------------------------------------------------------

    def get_active_session_id(self, agent_id: str) -> str | None:
        """返回 Agent 最近活跃的会话 ID"""
        sessions = self.list_sessions(agent_id)
        return sessions[0]["session_id"] if sessions else None


session_manager = SessionManager()
