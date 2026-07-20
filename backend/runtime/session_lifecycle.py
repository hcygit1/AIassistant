"""Session lifecycle orchestration around compaction and memory persistence."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from infra.event_bus import Events
from runtime.session_compactor import SessionCompactor

logger = logging.getLogger(__name__)


class SessionLifecycle:
    """Own automatic compaction and the SessionCompactor resource boundary."""

    def __init__(
        self,
        *,
        compactor: Any,
        resolve_agent_config: Callable[[str], dict[str, Any]],
        load_session: Callable[[str, str], dict[str, Any] | None],
        detect_compaction_level: Callable[..., str],
        audit_log: Callable[[str, str, dict[str, Any]], None],
        emit_event: Callable[[str, dict[str, Any]], None],
        get_store: Callable[[str], Any],
        get_state: Callable[[str], Any],
        batch_ingest_messages: Callable[..., Awaitable[None]],
        pending_tasks: set[asyncio.Task],
    ) -> None:
        self._compactor = compactor
        self._resolve_agent_config = resolve_agent_config
        self._load_session = load_session
        self._detect_compaction_level = detect_compaction_level
        self._audit_log = audit_log
        self._emit_event = emit_event
        self._get_store = get_store
        self._get_state = get_state
        self._batch_ingest_messages = batch_ingest_messages
        self._pending_tasks = pending_tasks

    async def maybe_auto_compact(
        self,
        session_id: str,
        agent_id: str,
        *,
        overhead_tokens: int = 0,
        compress_session: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        compaction_config = self._resolve_agent_config(agent_id).get(
            "compaction",
            {},
        )
        if not compaction_config.get("enabled", True):
            return

        data = self._load_session(session_id, agent_id)
        if not data:
            return

        level = self._detect_compaction_level(
            data.get("messages", []),
            agent_id=agent_id,
            overhead_tokens=overhead_tokens,
        )
        if level == "none":
            return

        logger.info(
            "Auto-compaction triggered: level=%s agent=%s session=%s",
            level,
            agent_id,
            session_id,
        )
        self._audit_log(
            agent_id,
            "auto_compact_trigger",
            {"session_id": session_id, "level": level},
        )
        self._emit_event(
            agent_id,
            Events.auto_compact_start(
                session_id=session_id,
                level=level,
            ),
        )
        try:
            await compress_session(session_id, agent_id, level=level)
            self._emit_event(
                agent_id,
                Events.auto_compact_done(session_id=session_id),
            )
        except Exception as error:
            logger.error("Auto-compaction failed: %s", error)
            self._audit_log(
                agent_id,
                "auto_compact_error",
                {"error": str(error)},
            )

    async def generate_structured_summary(
        self,
        agent_id: str,
        session_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
        *,
        plain_fallback: Callable[
            [str, list[dict[str, Any]], str],
            Awaitable[dict[str, Any]],
        ] | None = None,
    ) -> dict[str, Any]:
        return await self._compactor.generate_structured_summary(
            agent_id,
            session_id,
            to_compress,
            text_to_summarize,
            store=self._get_store(agent_id),
            plain_fallback=plain_fallback or self.summarize_plain_fallback,
        )

    async def summarize_plain_fallback(
        self,
        agent_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        return await self._compactor.summarize_plain_fallback(
            agent_id,
            to_compress,
            text_to_summarize,
        )

    async def compress_session(
        self,
        session_id: str,
        agent_id: str,
        level: str = "sliding",
        *,
        generate_summary: Callable[
            [str, str, list[dict[str, Any]], str],
            Awaitable[dict[str, Any]],
        ] | None = None,
        batch_ingest_messages: Callable[..., Awaitable[None]] | None = None,
        pending_tasks: set[asyncio.Task] | None = None,
    ) -> dict[str, Any]:
        return await self._compactor.compress_session(
            session_id,
            agent_id,
            level=level,
            get_store=self._get_store,
            get_state=self._get_state,
            generate_summary=generate_summary or self.generate_structured_summary,
            batch_ingest_messages=batch_ingest_messages or self._batch_ingest_messages,
            pending_tasks=(
                pending_tasks
                if pending_tasks is not None
                else self._pending_tasks
            ),
        )

    @staticmethod
    def calc_compress_count_by_turns(
        messages: list[dict[str, Any]],
        keep_turns: int,
    ) -> int:
        return SessionCompactor.calc_compress_count_by_turns(
            messages,
            keep_turns,
        )
