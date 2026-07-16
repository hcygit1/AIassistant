"""Session command side effects for reset and manual compaction."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Awaitable, Callable

from infra.event_bus import Events
from infra.token_counter import (
    count_messages_tokens,
    resolve_compaction_threshold,
)


class SessionCommands:
    def __init__(
        self,
        *,
        load_session: Callable[[str, str], dict[str, Any] | None],
        reset_session: Callable[[str, str], None],
        resolve_agent_config: Callable[[str], dict[str, Any]],
        emit_event: Callable[[str, dict[str, Any]], None],
        audit_log: Callable[[str, str, dict[str, Any]], None],
    ) -> None:
        self._load_session = load_session
        self._reset_session = reset_session
        self._resolve_agent_config = resolve_agent_config
        self._emit_event = emit_event
        self._audit_log = audit_log

    async def handle_reset(
        self,
        session_id: str,
        agent_id: str,
        *,
        model_override: str | None,
        get_store: Callable[[str], Any],
        get_state: Callable[[str], Any],
        batch_ingest_messages: Callable[..., Awaitable[None]],
        switch_model: Callable[[str, str], str],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Reset a session after flushing its transcript to memory."""
        yield {
            "type": "command_response",
            "response": "正在重置会话...",
        }

        data = self._load_session(session_id, agent_id)
        if data:
            messages = data.get("messages", [])
            if messages:
                await batch_ingest_messages(
                    agent_id,
                    session_id,
                    messages,
                    session_end=True,
                )

        self._reset_session(session_id, agent_id)
        store = get_store(agent_id)
        if store:
            store.delete_session_summary(session_id, agent_id)

        get_state(agent_id).compaction_count = 0
        model_message = ""
        if model_override:
            try:
                new_name = switch_model(agent_id, model_override)
                model_message = f" 模型已切换到 {new_name}。"
            except Exception as error:
                model_message = f" 模型切换失败: {error}"

        self._audit_log(
            agent_id,
            "session_reset",
            {
                "session_id": session_id,
                "model_override": model_override,
            },
        )
        message = "会话已重置。" + model_message
        yield {
            "type": "command_response",
            "response": message,
        }
        yield {
            "type": "session_reset",
            "session_id": session_id,
        }

    async def handle_reset_noflush(
        self,
        session_id: str,
        agent_id: str,
        *,
        get_store: Callable[[str], Any],
        get_state: Callable[[str], Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Reset a session without writing its transcript to memory."""
        yield {
            "type": "command_response",
            "response": "正在重置会话（不写入长期记忆）...",
        }

        self._reset_session(session_id, agent_id)
        store = get_store(agent_id)
        if store:
            store.delete_session_summary(session_id, agent_id)
        get_state(agent_id).compaction_count = 0

        self._audit_log(
            agent_id,
            "session_reset",
            {
                "session_id": session_id,
                "memory_saved": False,
                "mode": "no_memory",
            },
        )

        message = "会话已重置（本轮对话未写入长期记忆）。"
        yield {
            "type": "command_response",
            "response": message,
        }
        yield {
            "type": "session_reset",
            "session_id": session_id,
            "memory": {
                "saved": False,
                "reason": "no-flush",
            },
        }

    async def handle_compact(
        self,
        session_id: str,
        agent_id: str,
        *,
        compress_session: Callable[
            [str, str],
            Awaitable[dict[str, Any]],
        ],
        calc_compress_count_by_turns: Callable[
            [list[dict[str, Any]], int],
            int,
        ],
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "type": "command_response",
            "response": "正在执行压缩...",
        }
        self._emit_event(
            agent_id,
            Events.manual_compact_start(session_id=session_id),
        )

        try:
            result = await compress_session(session_id, agent_id)
            if "error" in result:
                reason = str(result.get("error") or "未知原因")
                session_data = (
                    self._load_session(session_id, agent_id) or {}
                )
                messages = session_data.get("messages", []) or []
                message_tokens = 0
                total_tokens = 0
                threshold = 0
                compressible_count = 0
                try:
                    message_tokens = count_messages_tokens(messages)
                    total_tokens = message_tokens
                    threshold = resolve_compaction_threshold(agent_id)
                except Exception:
                    pass

                try:
                    compaction_config = self._resolve_agent_config(
                        agent_id
                    ).get("compaction", {})
                    keep_recent_turns = int(
                        compaction_config.get(
                            "keepRecentTurns",
                            12,
                        )
                        or 12
                    )
                except Exception:
                    keep_recent_turns = 8

                try:
                    compressible_count = (
                        calc_compress_count_by_turns(
                            messages,
                            keep_recent_turns,
                        )
                    )
                except Exception:
                    compressible_count = 0

                suggestion = (
                    "建议：继续对话累积上下文，"
                    "或降低 compaction.keepRecentTurns。"
                )
                if reason == "消息过少，无需压缩":
                    suggestion = (
                        "建议：至少累积到 4 条以上消息后再尝试。"
                    )
                elif reason == "无足够消息可压缩":
                    suggestion = (
                        f"建议：当前轮次不足 keepRecentTurns"
                        f"({keep_recent_turns})，可继续对话后重试，"
                        "或调低 compaction.keepRecentTurns。"
                    )
                elif reason == "会话不存在":
                    suggestion = (
                        "建议：先发送一条消息创建会话，再执行 /compact。"
                    )

                message = (
                    f"压缩未执行：{reason}\n"
                    f"\n当前状态（动态）:\n"
                    f"- 消息数: {len(messages)}\n"
                    f"- 消息 tokens: {message_tokens}\n"
                    f"- 总 tokens: {total_tokens}\n"
                    f"- 压缩阈值(sliding threshold): {threshold}\n"
                    "- 保留轮次(compaction.keepRecentTurns): "
                    f"{keep_recent_turns}\n"
                    f"- 当前可压缩消息数: {compressible_count}\n"
                    f"\n{suggestion}"
                )
                yield {
                    "type": "command_response",
                    "response": message,
                }
                self._emit_event(
                    agent_id,
                    Events.manual_compact_skipped(
                        session_id=session_id,
                        reason=result.get("error", ""),
                    ),
                )
                yield {
                    "type": "done",
                    "content": message,
                    "session_id": session_id,
                }
                return

            compressed = result.get("compress", {}) or {}
            message = (
                "压缩完成。\n"
                f"- 归档消息：{compressed.get('archived_count', 0)} 条\n"
                f"- 剩余消息：{compressed.get('remaining_count', 0)} 条"
            )
            yield {
                "type": "command_response",
                "response": message,
            }
            yield {
                "type": "session_compacted",
                "result": result,
            }
            self._emit_event(
                agent_id,
                Events.manual_compact_done(
                    session_id=session_id,
                    data={
                        "archived_count": compressed.get(
                            "archived_count",
                            0,
                        ),
                        "remaining_count": compressed.get(
                            "remaining_count",
                            0,
                        ),
                    },
                ),
            )
            yield {
                "type": "done",
                "content": message,
                "session_id": session_id,
            }
        except Exception as error:
            self._emit_event(
                agent_id,
                Events.manual_compact_error(
                    session_id=session_id,
                    error=str(error)[:200],
                ),
            )
            yield {
                "type": "error",
                "error": f"压缩失败: {error}",
            }
