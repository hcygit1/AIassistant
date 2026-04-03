"""任务层处理器 — 边界检测 + finalize + Task embedding

按 docs/memory-system-refactor.md §4.2 ⑥:
  - session 变化 → finalize 旧 Task，创建新 Task
  - 超 idle_timeout_hours 空闲 → finalize 旧 Task，创建新 Task
  - LLM 判断话题切换 → finalize 旧 Task，创建新 Task
  - finalize: LLM 结构化摘要 → embedding → tasks + tasks_fts + vec_tasks
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import httpx

from mem.store import MemStore, Task, Chunk
from mem.embedder import MemEmbedder

logger = logging.getLogger(__name__)

TRIVIAL_RE = re.compile(
    r"^(test|testing|hello|hi|hey|ok|okay|yes|no|yeah|nope|sure|thanks|thx|ping|pong|"
    r"哈哈|好的|嗯|是的|不是|谢谢|你好|测试)\s*[.!?。！？]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TASK_SUMMARY_PROMPT = (
    "You create a DETAILED task summary from a multi-turn conversation. "
    "This summary will be the ONLY record of this conversation, so it must "
    "preserve ALL important information.\n\n"
    "CRITICAL LANGUAGE RULE: Write in the SAME language as the user's messages. "
    "Chinese input → Chinese output. English input → English output.\n\n"
    "Output EXACTLY this structure:\n\n"
    "📌 Title\n<short descriptive title, max 60 chars>\n\n"
    "🎯 Goal\n<what the user wanted to achieve>\n\n"
    "📋 Key Steps\n<numbered list of important actions taken>\n\n"
    "✅ Outcome\n<final result — success/failure/partial + key details>\n\n"
    "💡 Insights\n<lessons learned, gotchas, key decisions>\n\n"
    "Rules:\n"
    "- Preserve ALL: exact commands, file paths, config values, version numbers, error messages\n"
    "- DISCARD only: greetings, filler\n"
    "- Replace secrets with [REDACTED]\n"
    "- Target length: 30-50% of original conversation. Output summary only."
)

TOPIC_JUDGE_PROMPT = (
    "You are a conversation topic boundary detector. "
    "Given the CURRENT task context and a single NEW user message, "
    "decide if the new message belongs to the SAME task or starts a NEW one.\n\n"
    "Answer ONLY \"NEW\" or \"SAME\".\n\n"
    "SAME — the new message:\n"
    "- Continues, follows up on, refines, or corrects the same subject\n"
    "- References entities/concepts from the current task\n"
    "- Is a natural next step in the same workflow\n\n"
    "NEW — the new message:\n"
    "- Introduces a completely unrelated topic\n"
    "- Cannot be interpreted as a follow-up to the current task\n\n"
    "When in doubt → SAME. Output exactly one word: NEW or SAME"
)

OnTaskCompleted = Callable[[Task], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# MemTaskProcessor
# ---------------------------------------------------------------------------

class MemTaskProcessor:
    """Detects task boundaries and finalizes tasks with LLM summaries."""

    def __init__(
        self,
        store: MemStore,
        embedder: MemEmbedder,
        *,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        idle_timeout_ms: int = 2 * 3600 * 1000,
        on_task_completed: OnTaskCompleted | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.idle_timeout_ms = idle_timeout_ms
        self._on_task_completed = on_task_completed
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public entry — called after chunks are ingested
    # ------------------------------------------------------------------

    async def on_chunks_ingested(
        self, session_key: str, session_end: bool, owner: str = "agent:main",
    ) -> None:
        async with self._lock:
            await self._detect_and_process(session_key, owner)
            if session_end:
                active = self.store.get_active_task_by_session(session_key, owner)
                if active:
                    await self._finalize_task(active)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _detect_and_process(self, session_key: str, owner: str) -> None:
        all_active = self.store.get_all_active_tasks(owner)
        for t in all_active:
            if t.session_key != session_key:
                logger.info("Session changed: finalizing task=%s from session=%s", t.id, t.session_key)
                await self._finalize_task(t)

        active = self.store.get_active_task_by_session(session_key, owner)
        if not active:
            active = self._create_task(session_key, owner)

        await self._process_chunks_incrementally(active, session_key, owner)

    async def _process_chunks_incrementally(
        self, active_task: Task, session_key: str, owner: str,
    ) -> None:
        unassigned = self.store.get_unassigned_chunks(session_key, owner)
        if not unassigned:
            return

        task_chunks = self.store.get_chunks_by_task(active_task.id)

        if task_chunks:
            last_task_ts = max(c.created_at for c in task_chunks)
            first_unassigned_ts = min(c.created_at for c in unassigned)
            gap = first_unassigned_ts - last_task_ts
            if gap > self.idle_timeout_ms:
                logger.info("Task boundary: time gap %dmin", gap // 60_000)
                await self._finalize_task(active_task)
                new_task = self._create_task(session_key, owner)
                return await self._process_chunks_incrementally(new_task, session_key, owner)

        turns = self._group_into_turns(unassigned)
        if not turns:
            self._assign_chunks(unassigned, active_task.id)
            return

        current_task = active_task
        current_chunks = list(task_chunks)

        for turn in turns:
            user_chunk = next((c for c in turn if c.role == "user"), None)

            if not user_chunk:
                self._assign_chunks(turn, current_task.id)
                current_chunks.extend(turn)
                continue

            if current_chunks:
                last_ts = max(c.created_at for c in current_chunks)
                if user_chunk.created_at - last_ts > self.idle_timeout_ms:
                    logger.info("Task boundary: time gap per-turn")
                    await self._finalize_task(current_task)
                    current_task = self._create_task(session_key, owner)
                    current_chunks = []
                    self._assign_chunks(turn, current_task.id)
                    current_chunks.extend(turn)
                    continue

            existing_user_count = sum(1 for c in current_chunks if c.role == "user")
            if existing_user_count < 1:
                self._assign_chunks(turn, current_task.id)
                current_chunks.extend(turn)
                continue

            context = self._build_context_summary(current_chunks)
            new_msg = user_chunk.content[:500]
            is_new = await self._judge_new_topic(context, new_msg)

            if is_new is None:
                self._assign_chunks(turn, current_task.id)
                current_chunks.extend(turn)
                continue

            if is_new:
                logger.info("Task boundary: LLM judged new topic")
                await self._finalize_task(current_task)
                current_task = self._create_task(session_key, owner)
                current_chunks = []

            self._assign_chunks(turn, current_task.id)
            current_chunks.extend(turn)

    # ------------------------------------------------------------------
    # Task CRUD helpers
    # ------------------------------------------------------------------

    def _create_task(self, session_key: str, owner: str) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            session_key=session_key,
            owner=owner,
            title="",
            summary="",
            status="active",
        )
        self.store.insert_task(task)
        logger.info("Created new task=%s session=%s", task.id, session_key)
        return task

    def _assign_chunks(self, chunks: list[Chunk], task_id: str) -> None:
        if not chunks:
            return
        self.store.assign_chunks_to_task([c.id for c in chunks], task_id)

    async def _finalize_task(self, task: Task) -> None:
        chunks = self.store.get_chunks_by_task(task.id)
        fallback_title = self._extract_title(chunks)

        if not chunks:
            self.store.finalize_task(task.id, fallback_title, "无对话内容", "skipped")
            return

        skip_reason = self._should_skip_summary(chunks)
        if skip_reason:
            logger.info("Task %s skipped: %s", task.id, skip_reason)
            self.store.finalize_task(task.id, fallback_title, skip_reason, "skipped")
            for c in chunks:
                self.store.mark_dedup_status(c.id, "orphaned", reason="task_skipped")
            return

        conversation_text = self._build_conversation_text(chunks)
        try:
            summary = await self._llm_call(TASK_SUMMARY_PROMPT, conversation_text, max_tokens=4096, temperature=0.1)
        except Exception as e:
            logger.warning("Task summary failed for %s: %s", task.id, e)
            summary = self._fallback_summary(chunks)

        title, body = self._parse_title_from_summary(summary)
        title = title or fallback_title
        self.store.finalize_task(task.id, title, body, "completed")

        try:
            vec = await self.embedder.embed_query(f"{title} {body[:300]}")
            self.store.upsert_task_embedding(task.id, vec)
        except Exception as e:
            logger.warning("Task embedding failed for %s: %s", task.id, e)

        logger.info("Finalized task=%s title='%s' chunks=%d", task.id, title[:60], len(chunks))

        if self._on_task_completed:
            finalized = self.store.get_task(task.id)
            if finalized:
                try:
                    await self._on_task_completed(finalized)
                except Exception as e:
                    logger.warning("on_task_completed callback error: %s", e)

    # ------------------------------------------------------------------
    # Boundary detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_into_turns(chunks: list[Chunk]) -> list[list[Chunk]]:
        turns: list[list[Chunk]] = []
        current: list[Chunk] = []
        for c in chunks:
            if c.role == "user" and current:
                turns.append(current)
                current = []
            current.append(c)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _build_context_summary(chunks: list[Chunk]) -> str:
        conv = [c for c in chunks if c.role in ("user", "assistant")]
        if not conv:
            return ""

        def _fmt(c: Chunk) -> str:
            label = "User" if c.role == "user" else "Assistant"
            max_len = 500 if c.role == "user" else 200
            text = c.summary or c.content[:max_len]
            return f"[{label}]: {text}"

        if len(conv) <= 10:
            return "\n".join(_fmt(c) for c in conv)

        opening = [_fmt(c) for c in conv[:6]]
        recent = [_fmt(c) for c in conv[-4:]]
        return "\n".join(["--- Task opening ---", *opening, "--- Recent exchanges ---", *recent])

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_skip_summary(chunks: list[Chunk]) -> str | None:
        user_chunks = [c for c in chunks if c.role == "user"]
        asst_chunks = [c for c in chunks if c.role == "assistant"]

        if len(chunks) < 4:
            return f"消息过少（{len(chunks)} < 4）"

        turns = min(len(user_chunks), len(asst_chunks))
        if turns < 2:
            return f"对话轮次不足（{turns} < 2）"

        if not user_chunks:
            return "无用户消息"

        total_len = sum(len(c.content) for c in chunks)
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_chunks[0].content))
        min_len = 80 if has_cjk else 200
        if total_len < min_len:
            return f"内容过短（{total_len} < {min_len}）"

        user_content = " ".join(c.content for c in user_chunks)
        lines = [l.strip() for l in user_content.split("\n") if l.strip()]
        if lines:
            trivial = sum(1 for l in lines if len(l) < 5 or TRIVIAL_RE.match(l))
            if trivial / len(lines) > 0.7:
                return "用户内容为简单问候或测试"

        if len(user_chunks) >= 3:
            unique = set(c.content.strip().lower() for c in user_chunks)
            if len(unique) / len(user_chunks) < 0.4:
                return "用户消息高度重复"

        return None

    @staticmethod
    def _build_conversation_text(chunks: list[Chunk]) -> str:
        lines = []
        for c in chunks:
            label = "User" if c.role == "user" else ("Assistant" if c.role == "assistant" else c.role)
            lines.append(f"[{label}]: {c.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_title_from_summary(summary: str) -> tuple[str, str]:
        m = re.search(r"📌\s*(?:Title|标题)\s*\n(.+)", summary)
        if m:
            title = m.group(1).strip()[:80]
            body = summary[:m.start()].strip() + summary[m.end():].strip()
            return title, body.strip()
        return "", summary

    @staticmethod
    def _extract_title(chunks: list[Chunk]) -> str:
        first_user = next((c for c in chunks if c.role == "user"), None)
        if not first_user:
            return "Untitled Task"
        text = first_user.content.strip()
        return text[:60] if len(text) <= 60 else text[:57] + "..."

    @staticmethod
    def _fallback_summary(chunks: list[Chunk]) -> str:
        title_chunk = next((c for c in chunks if c.role == "user"), None)
        title = title_chunk.content[:60].strip() if title_chunk else "Untitled"
        summaries = [f"- {c.summary}" for c in chunks if c.summary]
        return "\n".join([f"🎯 Goal\n{title}", "", "📋 Key Steps", *summaries[:20]])

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _judge_new_topic(self, context: str, new_message: str) -> bool | None:
        user_content = (
            f"CURRENT TASK CONTEXT:\n{context}\n\n"
            f"NEW USER MESSAGE:\n{new_message}"
        )
        try:
            result = await self._llm_call(
                TOPIC_JUDGE_PROMPT, user_content, max_tokens=10, temperature=0,
            )
            result_upper = result.strip().upper()
            if "NEW" in result_upper:
                return True
            if "SAME" in result_upper:
                return False
            return None
        except Exception as e:
            logger.warning("Topic judge failed: %s", e)
            return None

    async def _llm_call(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        endpoint = f"{self.llm_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        body = {
            "model": self.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        store: MemStore,
        embedder: MemEmbedder,
        on_task_completed: OnTaskCompleted | None = None,
    ) -> MemTaskProcessor:
        llm_cfg = config.get("llm", {})
        embedding_cfg = config.get("embedding", {})
        task_cfg = config.get("task", {})
        idle_hours = task_cfg.get("idle_timeout_hours", 2)
        return cls(
            store=store,
            embedder=embedder,
            llm_base_url=llm_cfg.get("base_url") or embedding_cfg.get("base_url", ""),
            llm_api_key=llm_cfg.get("api_key") or embedding_cfg.get("api_key", ""),
            llm_model=llm_cfg.get("model") or "qwen-plus",
            idle_timeout_ms=int(idle_hours * 3600 * 1000),
            on_task_completed=on_task_completed,
        )
