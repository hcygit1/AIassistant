"""Result and failure lifecycle for one heartbeat work item."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from system_messages.heartbeat_history import HeartbeatEvent
from system_messages.heartbeat_utils import strip_heartbeat_token


class HeartbeatRunLifecycle:
    """Build terminal callbacks for one dispatched heartbeat run."""

    def __init__(
        self,
        *,
        rollback_last_turn: Callable[[str, str], Any],
        event_sink: Callable[[str, HeartbeatEvent], None],
        audit_event: Callable[[str, str, dict[str, Any]], None],
        emit_webchat_message: Callable[[str, str], None],
        now: Callable[[], float] = time.time,
    ) -> None:
        self._rollback_last_turn = rollback_last_turn
        self._event_sink = event_sink
        self._audit_event = audit_event
        self._emit_webchat_message = emit_webchat_message
        self._now = now

    def callbacks(
        self,
        *,
        agent_id: str,
        session_id: str,
        config: dict[str, Any],
        started_at: float,
    ) -> dict[str, Callable[..., Any]]:
        async def handle_result(response: str) -> None:
            ack_max = config.get("ackMaxChars", 300)
            should_skip, stripped = strip_heartbeat_token(
                response,
                max_ack_chars=ack_max,
            )

            if should_skip:
                self._rollback_last_turn(
                    session_id,
                    agent_id,
                )
                status = "ok-empty" if not response.strip() else "ok-token"
                self._emit_event(
                    agent_id,
                    started_at,
                    status=status,
                )
                self._audit_event(agent_id, "heartbeat_ok", {})
                return

            if config.get("target", "webchat") == "webchat":
                self._emit_event(
                    agent_id,
                    started_at,
                    status="sent",
                    preview=stripped[:200] if stripped else None,
                )
                self._emit_webchat_message(agent_id, session_id)
            self._audit_event(
                agent_id,
                "heartbeat_response",
                {"response": response[:500]},
            )

        async def handle_failure(error: Exception) -> None:
            error_text = str(error) or "heartbeat execution failed"
            if isinstance(error, asyncio.TimeoutError):
                self._emit_event(
                    agent_id,
                    started_at,
                    status="skipped",
                    reason="session-busy",
                )
                self._audit_event(
                    agent_id,
                    "heartbeat_skipped",
                    {"reason": "session-busy"},
                )
                return
            self._emit_event(
                agent_id,
                started_at,
                status="failed",
                reason=error_text,
            )
            self._audit_event(
                agent_id,
                "heartbeat_failed",
                {"error": error_text},
            )

        return {
            "result_handler": handle_result,
            "on_failure_async": handle_failure,
        }

    def _emit_event(
        self,
        agent_id: str,
        started_at: float,
        *,
        status: str,
        reason: str | None = None,
        preview: str | None = None,
    ) -> None:
        self._event_sink(
            agent_id,
            HeartbeatEvent(
                ts=int(self._now() * 1000),
                status=status,
                reason=reason,
                preview=preview,
                duration_ms=int((self._now() - started_at) * 1000),
                agent_id=agent_id,
            ),
        )
