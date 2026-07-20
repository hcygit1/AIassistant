"""会话管理器 — JSON 文件持久化 + 生命周期事件 + LRU 缓存"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import get_config, resolve_agent_sessions_dir
from sessions.session_repository import (
    SessionDataCorruptionError,
    SessionRepository,
)
from sessions.session_maintenance import SessionMaintenanceService
from sessions.session_catalog import SessionCatalog
from sessions.session_history_archive import SessionHistoryArchive
from sessions.session_message_writer import SessionMessageWriter
from sessions.session_title import SessionTitleService

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        repository: SessionRepository | None = None,
        maintenance: SessionMaintenanceService | None = None,
        title_service: SessionTitleService | None = None,
    ) -> None:
        self._repository = repository or SessionRepository(
            resolve_sessions_dir=resolve_agent_sessions_dir
        )
        self._maintenance = maintenance or SessionMaintenanceService(
            repository=self._repository,
            get_config=get_config,
        )
        self._title_service = title_service or SessionTitleService()
        self._history_archive = SessionHistoryArchive(
            repository=self._repository,
            load_session=lambda session_id, agent_id: self.load_session(
                session_id,
                agent_id,
            ),
            save_session=lambda session_id, agent_id, data: self._save_session_data(
                session_id,
                agent_id,
                data,
            ),
        )
        self._catalog = SessionCatalog(
            repository=self._repository,
            load_store=lambda agent_id: self._load_session_store(agent_id),
            save_store=lambda agent_id, store: self._save_session_store(
                agent_id,
                store,
            ),
            load_session=lambda session_id, agent_id: self.load_session(
                session_id,
                agent_id,
            ),
            derive_title=lambda data, **kwargs: self.derive_session_title(
                data,
                **kwargs,
            ),
            resolve_main_session_id=self.resolve_main_session_id,
            session_key_from_session_id=self.session_key_from_session_id,
            resolve_requester=self._resolve_requester_for_child_session,
            run_maintenance=lambda agent_id, **kwargs: self.run_maintenance(
                agent_id,
                **kwargs,
            ),
        )
        self._message_writer = SessionMessageWriter(
            transaction=self._session_transaction,
            load_session=lambda session_id, agent_id: self.load_session(
                session_id,
                agent_id,
            ),
            save_session_data=lambda session_id, agent_id, data: (
                self._save_session_data(session_id, agent_id, data)
            ),
            update_index=lambda *args, **kwargs: self._update_session_store_entry(
                *args,
                **kwargs,
            ),
            session_key_from_id=self.session_key_from_session_id,
            resolve_requester=self._resolve_requester_for_child_session,
        )

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

    def _get_store_lock(self, agent_id: str):
        return self._repository.get_agent_lock(agent_id)

    @contextmanager
    def _session_transaction(self, session_id: str, agent_id: str):
        with self._repository.get_agent_lock(agent_id):
            with self._repository.get_session_lock(session_id, agent_id):
                yield

    # ------------------------------------------------------------------
    # 主会话 — 每个 Agent 有且仅有一个主会话
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_main_session_id(agent_id: str) -> str:
        """返回 Agent 的固定主会话 ID（用于文件命名）"""
        return f"{agent_id}-main"

    @staticmethod
    def session_key_from_session_id(agent_id: str, session_id: str) -> str:
        """session_id -> session_key（agent:agentId:main / agent:agentId:subagent:xxx）"""
        main_sid = f"{agent_id}-main"
        if session_id == main_sid:
            return f"agent:{agent_id}:main"
        return f"agent:{agent_id}:subagent:{session_id}"

    @staticmethod
    def session_id_from_session_key(session_key: str) -> tuple[str, str] | None:
        """session_key -> (agent_id, session_id)。主会话 session_id=agent_id-main；子会话 session_id=subagent-xxx"""
        parts = (session_key or "").strip().split(":")
        if len(parts) < 3:
            return None
        if parts[0].lower() != "agent":
            return None
        agent_id = parts[1]
        rest = ":".join(parts[2:])
        if rest == "main":
            return (agent_id, f"{agent_id}-main")
        if len(parts) >= 4 and parts[2].lower() == "subagent":
            return (agent_id, parts[3])  # session_id = subagent-xxx
        return (agent_id, rest)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str, agent_id: str) -> Path:
        return self._repository.session_path(session_id, agent_id)

    def session_file_exists(self, session_id: str, agent_id: str) -> bool:
        return self._repository.session_file_exists(session_id, agent_id)

    def load_session(self, session_id: str, agent_id: str) -> dict[str, Any] | None:
        data = self._repository.load_session(session_id, agent_id)
        if data is None:
            return None

        current_label = str(data.get("label", "")).strip()
        if current_label and self._is_bootstrap_text(current_label):
            data.pop("label", None)
        if not data.get("label") and data.get("title"):
            candidate = str(data.get("title", "")).strip()
            if candidate and not self._is_bootstrap_text(candidate):
                data["label"] = candidate

        return data

    def load_session_for_agent(
        self, session_id: str, agent_id: str
    ) -> list[dict[str, Any]]:
        """
        为 LLM 优化的消息列表：
        - 合并连续 assistant 消息
        """
        data = self.load_session(session_id, agent_id)
        if data is None:
            return []

        messages: list[dict[str, Any]] = []

        raw_messages = data.get("messages", [])
        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if messages and messages[-1]["role"] == "assistant" and role == "assistant":
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": role, "content": content})

        return messages

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
        session_id = uuid.uuid4().hex[:12]
        data = {
            "session_id": session_id,
            "agent_id": agent_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
        }
        if title and title.strip():
            data["label"] = title.strip()
        self._save_session_data(session_id, agent_id, data)
        return session_id

    def rename_session(self, session_id: str, agent_id: str, title: str) -> bool:
        with self._session_transaction(session_id, agent_id):
            data = self.load_session(session_id, agent_id)
            if data is None:
                return False
            data["label"] = title
            data["updated_at"] = time.time()
            self._save_session_data(session_id, agent_id, data)
            return True

    def delete_session(self, session_id: str, agent_id: str) -> bool:
        with self._repository.get_agent_lock(agent_id):
            if self._repository.delete_session_file(session_id, agent_id):
                session_key = self.session_key_from_session_id(
                    agent_id,
                    session_id,
                )
                self._remove_session_store_entry(agent_id, session_key)
                try:
                    from sessions.session_lock_manager import (
                        cleanup_session_runtime,
                    )

                    cleanup_session_runtime(agent_id, session_id)
                except Exception as e:
                    logger.warning(
                        "cleanup_session_runtime after delete_session: %s",
                        e,
                    )
                return True
        return False

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
        import time as _time

        result: dict[str, Any] = {"archived": False}

        with self._repository.get_agent_lock(agent_id):
            path = self._session_path(session_id, agent_id)
            if path.exists():
                archive_dir = self._repository.sessions_dir(agent_id) / "archive"
                try:
                    ts = int(_time.time())
                    archive_name = f"{session_id}.reset.{ts}.json"
                    archive_path = archive_dir / archive_name
                    self._repository.archive_session_file(
                        session_id,
                        agent_id,
                        archive_path,
                    )
                    result["archived"] = True
                    result["archive_file"] = f"archive/{archive_name}"
                except OSError as e:
                    logger.warning(f"Failed to archive session {session_id}: {e}")
                    try:
                        self._repository.delete_session_file(
                            session_id,
                            agent_id,
                        )
                    except OSError:
                        pass

            self._repository.invalidate_session(session_id, agent_id)
            session_key = self.session_key_from_session_id(agent_id, session_id)
            self._remove_session_store_entry(agent_id, session_key)
            self.ensure_session(session_id, agent_id)

        return result

    # ------------------------------------------------------------------
    # 获取活跃会话 ID
    # ------------------------------------------------------------------

    def get_active_session_id(self, agent_id: str) -> str | None:
        """返回 Agent 最近活跃的会话 ID"""
        sessions = self.list_sessions(agent_id)
        return sessions[0]["session_id"] if sessions else None


session_manager = SessionManager()
