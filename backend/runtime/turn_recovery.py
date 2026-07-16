"""Recovery policies around a complete Agent fallback stream."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Awaitable, Callable

from infra.errors import (
    is_compaction_failure_error,
    is_likely_context_overflow_error,
    is_role_ordering_error,
    is_session_corruption_error,
    is_transient_http_error,
)
from infra.event_bus import Events


logger = logging.getLogger(__name__)

TRANSIENT_HTTP_RETRY_DELAY_MS = 2500


class TurnRecovery:
    def __init__(
        self,
        *,
        reset_session: Callable[[str, str], None],
        compress_session: Callable[
            ...,
            Awaitable[dict[str, Any]],
        ],
        audit_log: Callable[
            [str, str, dict[str, Any]],
            None,
        ],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._reset_session = reset_session
        self._compress_session = compress_session
        self._audit_log = audit_log
        self._sleep = sleep

    async def run(
        self,
        *,
        agent_id: str,
        session_id: str,
        state: Any,
        stream: Callable[
            [],
            AsyncGenerator[dict[str, Any], None],
        ],
    ) -> AsyncGenerator[dict[str, Any], None]:
        did_retry_transient = False
        did_reset_compaction = False
        did_retry_forced_compaction = False

        while True:
            try:
                async for event in stream():
                    yield event
                return
            except Exception as error:
                message = str(error)
                if bool(getattr(error, "committed", False)):
                    yield Events.turn_error(error=message)
                    yield {
                        "type": "error",
                        "error": message,
                    }
                    return

                if (
                    is_transient_http_error(message)
                    and not did_retry_transient
                ):
                    did_retry_transient = True
                    logger.warning(
                        "Transient HTTP error (%s). "
                        "Retrying in %sms.",
                        message[:150],
                        TRANSIENT_HTTP_RETRY_DELAY_MS,
                    )
                    await self._sleep(
                        TRANSIENT_HTTP_RETRY_DELAY_MS / 1000
                    )
                    continue

                if (
                    is_compaction_failure_error(message)
                    and not did_reset_compaction
                ):
                    did_reset_compaction = True
                    self._reset_session(session_id, agent_id)
                    state.compaction_count = 0
                    self._audit_log(
                        agent_id,
                        "session_reset_compaction_failure",
                        {"error": message[:200]},
                    )
                    yield {
                        "type": "session_reset",
                        "session_id": session_id,
                        "memory": {
                            "saved": False,
                            "reason": "compaction_failure",
                        },
                    }
                    yield {
                        "type": "done",
                        "content": (
                            "⚠️ 上下文超出限制，压缩失败。"
                            "已重置会话，请重试。\n\n"
                            "建议在 config 中提高 "
                            "agents.defaults.compaction."
                            "reserveTokensFloor（如 20000）"
                            "以降低此问题。"
                        ),
                        "session_id": session_id,
                    }
                    return

                if is_role_ordering_error(message):
                    self._reset_session(session_id, agent_id)
                    state.compaction_count = 0
                    yield {
                        "type": "session_reset",
                        "session_id": session_id,
                        "memory": {"saved": False},
                    }
                    yield {
                        "type": "done",
                        "content": (
                            "⚠️ 消息顺序冲突，已重置会话，"
                            "请重试。"
                        ),
                        "session_id": session_id,
                    }
                    return

                if is_session_corruption_error(message):
                    self._reset_session(session_id, agent_id)
                    state.compaction_count = 0
                    yield {
                        "type": "session_reset",
                        "session_id": session_id,
                        "memory": {"saved": False},
                    }
                    yield {
                        "type": "done",
                        "content": (
                            "⚠️ 会话历史损坏，已重置，请重试。"
                        ),
                        "session_id": session_id,
                    }
                    return

                if is_likely_context_overflow_error(message):
                    if not did_retry_forced_compaction:
                        did_retry_forced_compaction = True
                        logger.warning(
                            "Context overflow detected for "
                            "agent=%s session=%s. Attempting "
                            "forced compaction retry.",
                            agent_id,
                            session_id,
                        )
                        try:
                            forced_result = (
                                await self._compress_session(
                                    session_id,
                                    agent_id,
                                    level="forced",
                                )
                            )
                            if "error" not in forced_result:
                                self._audit_log(
                                    agent_id,
                                    "forced_compaction_retry",
                                    {
                                        "session_id": session_id,
                                        "reason": message[:200],
                                    },
                                )
                                continue
                            logger.warning(
                                "Forced compaction retry skipped "
                                "for agent=%s session=%s: %s",
                                agent_id,
                                session_id,
                                forced_result.get(
                                    "error",
                                    "unknown",
                                ),
                            )
                        except Exception as forced_error:
                            logger.warning(
                                "Forced compaction retry failed "
                                "for agent=%s session=%s: %s",
                                agent_id,
                                session_id,
                                forced_error,
                            )

                    yield {
                        "type": "error",
                        "error": (
                            "⚠️ 上下文溢出，已尝试紧急压缩但仍失败。"
                            "请缩短消息或使用更大 context 的模型。"
                        ),
                    }
                    return

                yield Events.turn_error(error=message)
                yield {
                    "type": "error",
                    "error": message,
                }
                return
