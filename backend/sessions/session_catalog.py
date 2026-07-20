"""会话索引、列表和会话目录维护。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class SessionCatalog:
    """管理 sessions.json 索引，不负责会话正文读写。"""

    def __init__(
        self,
        *,
        repository: Any,
        load_store: Callable[[str], dict[str, dict[str, Any]]],
        save_store: Callable[[str, dict[str, dict[str, Any]]], None],
        load_session: Callable[[str, str], dict[str, Any] | None],
        derive_title: Callable[..., str],
        resolve_main_session_id: Callable[[str], str],
        session_key_from_session_id: Callable[[str, str], str],
        resolve_requester: Callable[[str], tuple[str, str] | None],
        run_maintenance: Callable[..., tuple[dict, dict[str, Any]]],
    ) -> None:
        self._repository = repository
        self._load_store_callback = load_store
        self._save_store_callback = save_store
        self._load_session = load_session
        self._derive_title = derive_title
        self._resolve_main_session_id = resolve_main_session_id
        self._session_key_from_session_id = session_key_from_session_id
        self._resolve_requester = resolve_requester
        self._run_maintenance = run_maintenance

    def _load_store(self, agent_id: str) -> dict[str, dict[str, Any]]:
        return self._load_store_callback(agent_id)

    def _save_store(
        self,
        agent_id: str,
        store: dict[str, dict[str, Any]],
    ) -> None:
        self._save_store_callback(agent_id, store)

    def get_entry(
        self,
        session_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        session_key = self._session_key_from_session_id(agent_id, session_id)
        entry = self._load_store(agent_id).get(session_key)
        return dict(entry) if isinstance(entry, dict) else None

    def update_entry(
        self,
        agent_id: str,
        session_key: str,
        session_id: str,
        updated_at: float,
        label: str = "",
        spawned_by: str | None = None,
    ) -> None:
        lock = self._repository.get_agent_lock(agent_id)
        with lock:
            store = self._load_store(agent_id)
            entry = store.get(session_key, {})
            entry["sessionId"] = session_id
            entry["updatedAt"] = int(updated_at * 1000)
            if label:
                entry["label"] = label
            elif "label" in entry:
                entry.pop("label", None)
            if "title" in entry:
                entry.pop("title", None)
            if spawned_by:
                entry["spawnedBy"] = spawned_by
            store[session_key] = entry
            store, _ = self._run_maintenance(
                agent_id,
                store=store,
                enforce=False,
            )
            self._save_store(agent_id, store)

    def remove_entry(self, agent_id: str, session_key: str) -> None:
        lock = self._repository.get_agent_lock(agent_id)
        with lock:
            store = self._load_store(agent_id)
            store.pop(session_key, None)
            self._save_store(agent_id, store)

    def list_sessions(
        self,
        agent_id: str,
        spawned_by_session_key: str | None = None,
    ) -> list[dict[str, Any]]:
        sessions_dir = self._repository.sessions_dir(agent_id)
        store = self._load_store(agent_id)
        main_sid = self._resolve_main_session_id(agent_id)
        main_key = self._session_key_from_session_id(agent_id, main_sid)

        if main_key not in store:
            main_data = self._load_session(main_sid, agent_id)
            if main_data:
                if isinstance(main_data, list):
                    main_data = {"messages": main_data, "label": "未命名"}
            else:
                main_data = {
                    "messages": [],
                    "label": "主会话",
                    "created_at": 0,
                    "updated_at": 0,
                }
            self.update_entry(
                agent_id,
                main_key,
                main_sid,
                main_data.get("updated_at", 0) or 0,
                label=main_data.get("label", "主会话"),
            )
            store[main_key] = {
                "sessionId": main_sid,
                "updatedAt": int((main_data.get("updated_at") or 0) * 1000),
                "label": main_data.get("label", "主会话"),
            }

        if sessions_dir.exists():
            for path in sessions_dir.glob("subagent-*.json"):
                session_id = path.stem
                session_key = self._session_key_from_session_id(
                    agent_id,
                    session_id,
                )
                if session_key in store:
                    continue
                try:
                    data = self._load_session(session_id, agent_id)
                    if data is None:
                        continue
                    spawned_by = data.get("spawned_by")
                    if not spawned_by:
                        resolved = self._resolve_requester(session_key)
                        if resolved:
                            spawned_by = resolved[0]
                    self.update_entry(
                        agent_id,
                        session_key,
                        session_id,
                        data.get("updated_at", path.stat().st_mtime),
                        label=data.get("label", ""),
                        spawned_by=spawned_by,
                    )
                    store[session_key] = {
                        "sessionId": session_id,
                        "updatedAt": int(
                            (data.get("updated_at") or path.stat().st_mtime)
                            * 1000
                        ),
                        "label": data.get("label", ""),
                        "spawnedBy": spawned_by,
                    }
                except Exception:
                    continue

        result: list[dict[str, Any]] = []
        for session_key, entry in store.items():
            if (
                spawned_by_session_key
                and entry.get("spawnedBy") != spawned_by_session_key
            ):
                continue
            session_id = entry.get("sessionId", "")
            data = self._load_session(session_id, agent_id)
            if data is None and session_id != main_sid:
                continue
            if data:
                if isinstance(data, list):
                    data = {"messages": data, "label": "未命名"}
            else:
                data = {
                    "messages": [],
                    "label": "主会话",
                    "created_at": 0,
                    "updated_at": 0,
                }
            result.append(
                {
                    "session_id": session_id,
                    "session_key": session_key,
                    "title": self._derive_title(
                        data,
                        session_id=session_id,
                        updated_at=data.get("updated_at"),
                    ),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                    "spawned_by": entry.get("spawnedBy"),
                }
            )

        result.sort(key=lambda item: item.get("updated_at") or 0, reverse=True)
        return result
