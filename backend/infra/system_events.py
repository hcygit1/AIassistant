"""System Events 队列 — 供 Cron 入队、Heartbeat 读取

持久化到 JSON 文件，进程重启后自动恢复未消费的事件。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_max_events() -> int:
    try:
        from config import get_config
        cfg = get_config()
        return cfg.get("app", {}).get("systemEvents", {}).get("maxEvents", 20)
    except Exception:
        return 20


def _events_dir() -> Path:
    from config import DATA_DIR
    p = DATA_DIR / "system_events"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _events_file(session_key: str) -> Path:
    safe_key = session_key.replace(":", "_").replace("/", "_")
    return _events_dir() / f"{safe_key}.json"


def _read_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt system events file: %s", path)
        return []


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    try:
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.error("Failed to write system events: %s", e)


def _normalize_context_key(key: str | None) -> str | None:
    if not key or not str(key).strip():
        return None
    return str(key).strip().lower()


def enqueue_system_event(
    text: str,
    session_key: str,
    context_key: str | None = None,
) -> None:
    """入队系统事件，持久化到文件。"""
    key = (session_key or "").strip()
    if not key:
        return
    cleaned = (text or "").strip()
    if not cleaned:
        return

    ctx = _normalize_context_key(context_key)
    path = _events_file(key)
    entries = _read_entries(path)

    if ctx:
        entries = [e for e in entries if e.get("contextKey") != ctx]

    entries.append({"text": cleaned, "ts": int(time.time() * 1000), "contextKey": ctx})

    max_events = _get_max_events()
    if len(entries) > max_events:
        entries = entries[-max_events:]

    _write_entries(path, entries)


def peek_system_event_entries(session_key: str) -> list[dict[str, Any]]:
    """读取 pending 事件（不消费）。"""
    key = (session_key or "").strip()
    if not key:
        return []
    return _read_entries(_events_file(key))


def drain_system_event_entries(session_key: str) -> list[dict[str, Any]]:
    """取出并消费 pending 事件。"""
    key = (session_key or "").strip()
    if not key:
        return []
    path = _events_file(key)
    entries = _read_entries(path)
    if entries:
        _write_entries(path, [])
    return entries


def peek_system_event_entries_for_agent(agent_id: str) -> list[dict[str, Any]]:
    """按 agent_id 取主会话的 pending 事件。"""
    from graph.session_manager import session_manager
    main_sid = session_manager.resolve_main_session_id(agent_id)
    session_key = session_manager.session_key_from_session_id(agent_id, main_sid)
    return peek_system_event_entries(session_key)
