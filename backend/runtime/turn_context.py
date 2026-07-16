"""Prompt, message, and session context assembly with bounded caches."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_heartbeat_config, resolve_agent_config


@dataclass
class PromptCacheEntry:
    key: tuple[Any, ...]
    system_prompt: str
    prompt_report: Any
    prompt_tokens: int


@dataclass
class SessionContextCacheEntry:
    agent_id: str
    session_id: str
    session_file_mtime: float | None
    summary_fingerprint: str | None
    raw_history: list[dict[str, Any]]
    summary_text: str
    history_with_summary: list[dict[str, Any]]
    pruned_history: list[dict[str, Any]]
    summary_tokens: int
    history_tokens: int
    pruning_signature: str = ""


class _BoundedLRUCache(OrderedDict[Any, Any]):
    """Ordered mapping that enforces capacity for every public write."""

    def __init__(
        self,
        max_entries: int,
        initial: Mapping[Any, Any] | None = None,
    ) -> None:
        self.max_entries = max(1, max_entries)
        super().__init__()
        if initial:
            self.update(initial)

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        while len(self) > self.max_entries:
            self.popitem(last=False)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 1:
            raise TypeError(
                f"update expected at most 1 argument, got {len(args)}"
            )
        if args:
            source = args[0]
            items = (
                list(source.items())
                if hasattr(source, "items")
                else list(source)
            )
            for key, value in items:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __ior__(self, other: Mapping[Any, Any]):
        self.update(other)
        return self


class TurnContext:
    def __init__(
        self,
        *,
        max_prompt_entries: int = 128,
        max_session_entries: int = 1024,
    ) -> None:
        self._max_prompt_entries = max(1, max_prompt_entries)
        self._max_session_entries = max(1, max_session_entries)
        self._prompt_cache = _BoundedLRUCache(
            self._max_prompt_entries
        )
        self._session_context_cache = _BoundedLRUCache(
            self._max_session_entries
        )

    @property
    def prompt_cache(
        self,
    ) -> _BoundedLRUCache:
        return self._prompt_cache

    @prompt_cache.setter
    def prompt_cache(
        self,
        value: Mapping[tuple[Any, ...], PromptCacheEntry],
    ) -> None:
        if (
            isinstance(value, _BoundedLRUCache)
            and value.max_entries == self._max_prompt_entries
        ):
            self._prompt_cache = value
            return
        self._prompt_cache = _BoundedLRUCache(
            self._max_prompt_entries,
            value,
        )

    @property
    def session_context_cache(
        self,
    ) -> _BoundedLRUCache:
        return self._session_context_cache

    @session_context_cache.setter
    def session_context_cache(
        self,
        value: Mapping[tuple[str, str], SessionContextCacheEntry],
    ) -> None:
        if (
            isinstance(value, _BoundedLRUCache)
            and value.max_entries == self._max_session_entries
        ):
            self._session_context_cache = value
            return
        self._session_context_cache = _BoundedLRUCache(
            self._max_session_entries,
            value,
        )

    @staticmethod
    def build_messages(
        history: list[dict[str, Any]],
        new_message: str,
        *,
        human_message: Callable[..., Any] = HumanMessage,
        ai_message: Callable[..., Any] = AIMessage,
        system_message: Callable[..., Any] = SystemMessage,
    ) -> list:
        messages = []
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "user":
                messages.append(human_message(content=content))
            elif role == "assistant":
                messages.append(ai_message(content=content))
            elif role == "system":
                if not messages:
                    messages.append(system_message(content=content))
                else:
                    messages.append(human_message(content=content))
        messages.append(human_message(content=new_message))
        return messages

    @staticmethod
    def safe_mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime if path.exists() else None
        except Exception:
            return None

    @staticmethod
    def project_context_signature(
        agent_id: str,
        prompt_mode: str,
        *,
        resolve_workspace: Callable[[str], Path],
        resolve_agent_dir: Callable[[str], Path],
        safe_mtime: Callable[[Path], float | None],
    ) -> tuple[Any, ...]:
        workspace = resolve_workspace(agent_id)
        files: list[Path] = [workspace / "AGENTS.md"]
        if prompt_mode == "full":
            files.extend([workspace / "IDENTITY.md", workspace / "USER.md"])
        snapshot = resolve_agent_dir(agent_id) / "SKILLS_SNAPSHOT.md"
        bootstrap = workspace / "BOOTSTRAP.md"
        return (
            tuple((str(path), safe_mtime(path)) for path in files),
            safe_mtime(snapshot),
            bootstrap.exists(),
        )

    @staticmethod
    def prompt_runtime_signature(
        agent_id: str,
        *,
        resolve_agent_config_fn: Callable[[str], dict[str, Any]] | None = None,
        get_heartbeat_config_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        resolve_config = resolve_agent_config_fn or resolve_agent_config
        resolve_heartbeat = (
            get_heartbeat_config_fn or get_heartbeat_config
        )
        agent_config = resolve_config(agent_id)
        heartbeat_config = resolve_heartbeat(agent_id)
        config_payload = {
            "model": agent_config.get("model"),
            "thinkingDefault": agent_config.get("thinkingDefault"),
            "user_timezone": agent_config.get("user_timezone"),
            "heartbeat_prompt": heartbeat_config.get("prompt"),
        }
        serialized = json.dumps(
            config_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        config_signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        minute_bucket = datetime.now().strftime("%Y-%m-%d %H:%M")
        return config_signature, minute_bucket

    @staticmethod
    def pruning_signature(
        agent_id: str,
        *,
        resolve_agent_config_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> str:
        resolve_config = resolve_agent_config_fn or resolve_agent_config
        agent_config = resolve_config(agent_id)
        payload = {
            "contextTokens": agent_config.get("contextTokens"),
            "contextBudget": agent_config.get("contextBudget"),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_or_build_prompt(
        self,
        *,
        agent_id: str,
        prompt_mode: str,
        available_tool_names: list[str] | None,
        extra_system_prompt: str | None,
        locale: str,
        static_signature: tuple[Any, ...],
        runtime_signature: tuple[Any, ...],
        build_prompt: Callable[[Any], tuple[str, Any]],
        count_tokens: Callable[[str], int],
    ) -> tuple[str, Any, int]:
        from runtime.prompt_builder import PromptParams

        tool_key = tuple(sorted(available_tool_names or []))
        cache_key = (
            agent_id,
            prompt_mode,
            tool_key,
            extra_system_prompt or "",
            locale,
            static_signature,
            runtime_signature,
        )
        cached = self.prompt_cache.get(cache_key)
        if cached is not None:
            return (
                cached.system_prompt,
                cached.prompt_report,
                cached.prompt_tokens,
            )

        params = PromptParams(
            agent_id=agent_id,
            mode=prompt_mode,
            available_tools=available_tool_names,
            extra_system_prompt=extra_system_prompt or None,
            locale=locale,
        )
        system_prompt, prompt_report = build_prompt(params)
        prompt_tokens = count_tokens(system_prompt)
        entry = PromptCacheEntry(
            key=cache_key,
            system_prompt=system_prompt,
            prompt_report=prompt_report,
            prompt_tokens=prompt_tokens,
        )
        self.prompt_cache[cache_key] = entry
        return system_prompt, prompt_report, prompt_tokens

    @staticmethod
    def session_summary_fingerprint(summary: Any) -> str | None:
        if not summary:
            return None
        try:
            if isinstance(summary, dict):
                payload = summary
            else:
                payload = {
                    "goal": getattr(summary, "goal", None),
                    "decisions": getattr(summary, "decisions", None),
                    "progress": getattr(summary, "progress", None),
                    "open_items": getattr(summary, "open_items", None),
                    "entities": getattr(summary, "entities", None),
                    "user_preferences": getattr(
                        summary,
                        "user_preferences",
                        None,
                    ),
                    "raw_summary": getattr(summary, "raw_summary", None),
                }
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(summary)

    def get_or_build_session_context(
        self,
        *,
        agent_id: str,
        session_id: str,
        session_path: Path,
        store: Any,
        safe_mtime: Callable[[Path], float | None],
        summary_fingerprint: Callable[[Any], str | None],
        load_history: Callable[[str, str], list[dict[str, Any]]],
        format_summary: Callable[[Any], str],
        prune_history: Callable[..., list[dict[str, Any]]],
        count_tokens: Callable[[str], int],
        count_messages_tokens: Callable[[list[dict[str, Any]]], int],
        pruning_signature: str,
    ) -> SessionContextCacheEntry:
        cache_key = (agent_id, session_id)
        session_file_mtime = safe_mtime(session_path)
        session_summary = (
            store.get_session_summary(session_id, agent_id)
            if store
            else None
        )
        current_summary_fingerprint = summary_fingerprint(session_summary)

        cached = self.session_context_cache.get(cache_key)
        if (
            cached is not None
            and cached.session_file_mtime == session_file_mtime
            and cached.summary_fingerprint == current_summary_fingerprint
            and cached.pruning_signature == pruning_signature
        ):
            return cached

        raw_history = load_history(session_id, agent_id)
        summary_text = ""
        history_with_summary = list(raw_history)
        if session_summary:
            summary_text = format_summary(session_summary)
            if summary_text:
                history_with_summary = [
                    {"role": "system", "content": summary_text},
                    *history_with_summary,
                ]

        pruned_history = prune_history(
            history_with_summary,
            agent_id=agent_id,
        )
        summary_tokens = count_tokens(summary_text) if summary_text else 0
        history_tokens = count_messages_tokens(pruned_history)
        entry = SessionContextCacheEntry(
            agent_id=agent_id,
            session_id=session_id,
            session_file_mtime=session_file_mtime,
            summary_fingerprint=current_summary_fingerprint,
            raw_history=raw_history,
            summary_text=summary_text,
            history_with_summary=history_with_summary,
            pruned_history=pruned_history,
            summary_tokens=summary_tokens,
            history_tokens=history_tokens,
            pruning_signature=pruning_signature,
        )
        self.session_context_cache[cache_key] = entry
        return entry
