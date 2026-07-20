"""Transcript archiving used by session compaction."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from sessions.session_repository import SessionRepository


class SessionHistoryArchive:
    """Archive old transcript messages and record compaction metadata."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        load_session: Callable[[str, str], dict[str, Any] | None],
        save_session: Callable[[str, str, dict[str, Any]], None],
        now: Callable[[], float] | None = None,
    ) -> None:
        self._repository = repository
        self._load_session = load_session
        self._save_session = save_session
        self._now = now or time.time

    def compress_history(
        self,
        session_id: str,
        agent_id: str,
        n_messages: int,
    ) -> dict[str, int]:
        data = self._load_session(session_id, agent_id)
        if data is None:
            return {"archived_count": 0, "remaining_count": 0}

        messages = data.get("messages", [])
        if len(messages) < 4:
            return {"archived_count": 0, "remaining_count": len(messages)}

        archive_count = min(n_messages, len(messages))
        archived = messages[:archive_count]
        remaining = messages[archive_count:]

        archive_dir = self._repository.sessions_dir(agent_id) / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{session_id}_{int(self._now())}.json"
        with open(archive_path, "w", encoding="utf-8") as file:
            json.dump(archived, file, ensure_ascii=False, indent=2)

        data["messages"] = remaining
        data["updated_at"] = self._now()
        self._save_session(session_id, agent_id, data)

        compactions_path = (
            self._repository.sessions_dir(agent_id) / "compactions.jsonl"
        )
        try:
            record = {
                "session_id": session_id,
                "agent_id": agent_id,
                "ts": self._now(),
                "archived_count": archive_count,
                "remaining_count": len(remaining),
            }
            with open(compactions_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return {
            "archived_count": archive_count,
            "remaining_count": len(remaining),
        }
