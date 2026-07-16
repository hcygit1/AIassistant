"""Session summary generation and transcript compaction."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from infra.token_counter import count_tokens
from llm.model_selection import (
    get_model_context_window,
    resolve_agent_model,
)
from llm.retry import retry_async
from runtime.context_budget import resolve_budget


logger = logging.getLogger(__name__)


class SessionCompactor:
    def __init__(
        self,
        *,
        resolve_agent_config: Callable[[str], dict[str, Any]],
        load_session: Callable[[str, str], dict[str, Any] | None],
        compress_history: Callable[[str, str, int], dict[str, Any]],
        get_llm: Callable[[str], Any],
        log_compress: Callable[[str, str, int, int], None],
    ) -> None:
        self._resolve_agent_config = resolve_agent_config
        self._load_session = load_session
        self._compress_history = compress_history
        self._get_llm = get_llm
        self._log_compress = log_compress

    async def generate_structured_summary(
        self,
        agent_id: str,
        session_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
        *,
        store: Any,
        plain_fallback: Callable[
            [str, list[dict[str, Any]], str],
            Awaitable[dict[str, Any]],
        ],
    ) -> dict[str, Any]:
        """Generate a structured summary and fall back to plain text."""
        prev_summary: dict[str, Any] = {}
        if store:
            existing = store.get_session_summary(session_id, agent_id)
            if existing:
                prev_summary = {
                    "goal": existing.goal,
                    "decisions": (
                        json.loads(existing.decisions)
                        if existing.decisions
                        else []
                    ),
                    "progress": existing.progress,
                    "open_items": (
                        json.loads(existing.open_items)
                        if existing.open_items
                        else []
                    ),
                    "entities": (
                        json.loads(existing.entities)
                        if existing.entities
                        else []
                    ),
                    "user_preferences": (
                        json.loads(existing.user_preferences)
                        if existing.user_preferences
                        else []
                    ),
                }

        prev_block = ""
        if prev_summary:
            prev_block = (
                "\n\n## 上一版摘要（请在此基础上更新，而非重新生成）\n"
                f"{json.dumps(prev_summary, ensure_ascii=False, indent=2)}"
            )

        system_prompt = (
            "你是一个对话摘要生成器。将对话历史压缩为结构化 JSON 摘要。\n\n"
            "关键语言规则：使用与用户消息相同的语言输出。"
            "中文输入→中文输出。英文输入→英文输出。\n\n"
            "输出严格的 JSON（无 markdown 代码块包裹）：\n"
            "{\n"
            '  "goal": "用户的总体目标（1 句话）",\n'
            '  "decisions": ["关键决策1", "关键决策2"],\n'
            '  "progress": "当前进展到哪一步",\n'
            '  "open_items": ["待办事项1", "未解决问题1"],\n'
            '  "entities": ["关键实体: 版本号/路径/配置值等"],\n'
            '  "user_preferences": ["用户偏好1"]\n'
            "}\n\n"
            "规则：\n"
            "- 保留所有关键信息：命令、文件路径、配置值、版本号、错误信息\n"
            "- 丢弃寒暄、填充词\n"
            "- 如果有上一版摘要，在其基础上更新（合并/覆盖），"
            "不要丢弃仍然有效的信息\n"
            "- 敏感信息替换为 [REDACTED]\n"
            "- 只输出 JSON，不要输出其他内容"
        )

        summary_max_tokens = resolve_budget(
            agent_id
        ).session_summary_tokens

        async def _do_structured(text: str) -> dict[str, Any]:
            llm = self._get_llm(agent_id)
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=text + prev_block),
                ],
                max_tokens=summary_max_tokens,
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = (
                    raw.split("\n", 1)[-1]
                    .rsplit("```", 1)[0]
                    .strip()
                )
            parsed = json.loads(raw)
            parsed["raw_summary"] = raw
            parsed["token_count"] = count_tokens(raw)
            return parsed

        try:
            return await retry_async(
                lambda: _do_structured(text_to_summarize),
                attempts=3,
                min_delay_ms=500,
                max_delay_ms=5000,
                jitter=0.2,
                should_retry=lambda error, _: (
                    "AbortError" not in type(error).__name__
                ),
            )
        except Exception as error:
            logger.warning(
                "Structured summary failed, falling back to plain text: %s",
                error,
            )

        return await plain_fallback(
            agent_id,
            to_compress,
            text_to_summarize,
        )

    async def summarize_plain_fallback(
        self,
        agent_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        """Generate a plain summary after structured output fails."""
        summary_max_tokens = resolve_budget(
            agent_id
        ).session_summary_tokens

        async def _do_summarize(text: str) -> str:
            llm = self._get_llm(agent_id)
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你是一个对话摘要生成器。请将以下对话历史压缩为"
                            "简洁的摘要，不超过500字。使用与用户消息相同的语言。"
                            "保留关键信息、决定、上下文和待办事项。"
                        )
                    ),
                    HumanMessage(content=text),
                ],
                max_tokens=summary_max_tokens,
            )
            return response.content.strip()

        try:
            text = await retry_async(
                lambda: _do_summarize(text_to_summarize),
                attempts=3,
                min_delay_ms=500,
                max_delay_ms=5000,
                jitter=0.2,
                should_retry=lambda error, _: (
                    "AbortError" not in type(error).__name__
                ),
            )
            return {
                "raw_summary": text,
                "token_count": count_tokens(text),
            }
        except Exception as full_error:
            logger.warning(
                "Full summarization failed, trying partial: %s",
                full_error,
            )

        try:
            ref = resolve_agent_model(agent_id)
            context_window = get_model_context_window(ref)
            small_messages: list[dict[str, Any]] = []
            oversized_notes: list[str] = []
            for message in to_compress:
                content = message.get("content", "")
                tokens = count_tokens(content) + 4
                if tokens > context_window * 0.5:
                    role = message.get("role", "message")
                    oversized_notes.append(
                        f"[Large {role} (~{tokens // 1000}K tokens) "
                        "omitted from summary]"
                    )
                else:
                    small_messages.append(message)

            if small_messages:
                partial_text = "\n".join(
                    f"[{message.get('role', '?')}] "
                    f"{message.get('content', '')}"
                    for message in small_messages
                )
                partial = await retry_async(
                    lambda: _do_summarize(partial_text),
                    attempts=2,
                    min_delay_ms=500,
                    max_delay_ms=3000,
                    jitter=0.2,
                )
                notes = (
                    "\n\n" + "\n".join(oversized_notes)
                    if oversized_notes
                    else ""
                )
                text = partial + notes
                return {
                    "raw_summary": text,
                    "token_count": count_tokens(text),
                }
        except Exception as partial_error:
            logger.warning(
                "Partial summarization failed: %s",
                partial_error,
            )

        fallback = (
            f"Context contained {len(to_compress)} messages. "
            "Summary unavailable due to size limits."
        )
        return {
            "raw_summary": fallback,
            "token_count": count_tokens(fallback),
        }

    async def compress_session(
        self,
        session_id: str,
        agent_id: str,
        level: str = "sliding",
        *,
        get_store: Callable[[str], Any],
        get_state: Callable[[str], Any],
        generate_summary: Callable[
            [str, str, list[dict[str, Any]], str],
            Awaitable[dict[str, Any]],
        ],
        batch_ingest_messages: Callable[..., Awaitable[None]],
        pending_tasks: set[asyncio.Task],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "compress": None,
            "post_compaction": None,
        }

        compaction_config = self._resolve_agent_config(
            agent_id
        ).get("compaction", {})
        if level == "forced":
            keep_turns = int(
                compaction_config.get("forcedKeepRecentTurns", 4)
            )
        else:
            keep_turns = int(
                compaction_config.get("keepRecentTurns", 12)
            )

        data = self._load_session(session_id, agent_id)
        if not data:
            return {**result, "error": "会话不存在"}

        messages = data.get("messages", [])
        if len(messages) < 4:
            return {**result, "error": "消息过少，无需压缩"}

        compress_count = self.calc_compress_count_by_turns(
            messages,
            keep_turns,
        )
        if compress_count < 2:
            return {**result, "error": "无足够消息可压缩"}

        to_compress = messages[:compress_count]
        text_to_summarize = "\n".join(
            f"[{message.get('role', '?')}] "
            f"{message.get('content', '')}"
            for message in to_compress
        )
        summary = await generate_summary(
            agent_id,
            session_id,
            to_compress,
            text_to_summarize,
        )

        store = get_store(agent_id)
        if store:
            store.upsert_session_summary(
                session_id,
                agent_id,
                summary,
            )

        compress_result = self._compress_history(
            session_id,
            agent_id,
            compress_count,
        )
        result["compress"] = {
            "summary": summary,
            "level": level,
            **compress_result,
        }

        state = get_state(agent_id)
        state.compaction_count += 1
        archived_count = compress_result.get("archived_count", 0)
        remaining_count = compress_result.get("remaining_count", 0)
        self._log_compress(
            agent_id,
            session_id,
            archived_count,
            remaining_count,
        )

        task = asyncio.create_task(
            batch_ingest_messages(
                agent_id,
                session_id,
                to_compress,
                session_end=False,
            )
        )
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

        return result

    @staticmethod
    def calc_compress_count_by_turns(
        messages: list[dict[str, Any]],
        keep_turns: int,
    ) -> int:
        """Return the number of old messages before recent complete turns."""
        if not messages:
            return 0

        turn_boundaries: list[int] = []
        index = len(messages) - 1
        while index >= 0:
            if (
                messages[index].get("role") == "assistant"
                and index > 0
                and messages[index - 1].get("role") == "user"
            ):
                turn_boundaries.append(index - 1)
                index -= 2
            else:
                turn_boundaries.append(index)
                index -= 1

        turn_boundaries.reverse()
        if len(turn_boundaries) <= keep_turns:
            return 0

        compress_count = turn_boundaries[-keep_turns]
        return (
            max(compress_count, 2)
            if compress_count >= 2
            else 0
        )
