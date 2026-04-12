"""记忆入库队列 — hash 去重 + LLM summary + sqlite-vec ANN 语义去重

流程严格按 docs/memory-system-refactor.md §4.2:
  ① 精确 hash 去重
  ② LLM 生成 summary（保留数字/配置值/结论）
  ③ Embedder 对 summary 嵌入
  ④ sqlite-vec ANN 候选 → LLM judgeDedup（summary + content 前 300 字）
  ⑤ INSERT chunks + vec_chunks
  ⑥ 通知 TaskProcessor（阶段四接入）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Literal

import httpx

from mem.store import Chunk, MemStore, SearchHit, _content_hash, _now_ms
from mem.embedder import MemEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = (
    "Summarize the text in ONE concise sentence (max 120 characters). "
    "IMPORTANT: Use the SAME language as the input text — if the input is Chinese, write Chinese; "
    "if English, write English. "
    "Preserve ALL numbers, config values, version numbers, code identifiers, error codes. "
    "Preserve conclusions (success/failure and reasons). "
    "No bullet points, no preamble — output only the sentence."
)

DEDUP_JUDGE_PROMPT = """\
You are a memory deduplication system.

LANGUAGE RULE (MUST FOLLOW): You MUST reply in the SAME language as the input memories. \
如果输入是中文，reason 和 mergedSummary 必须用中文。If input is English, reply in English.

Given a NEW memory and several EXISTING memories (each with summary and content excerpt), \
determine the relationship.

For each EXISTING memory, the NEW memory is either:
- "DUPLICATE": Content is identical or conveys the same information with no new details. \
Only choose DUPLICATE when content is truly the same or adds zero new information.
- "UPDATE": NEW contains meaningful additional information that supplements an EXISTING memory \
(new data, status change, concrete detail not present before)
- "NEW": NEW covers a genuinely different topic/event with no semantic overlap

Pick the BEST match among all candidates. If none match well, choose "NEW".

Output a single JSON object (reason and mergedSummary MUST match input language):
- If DUPLICATE: {"action":"DUPLICATE","targetIndex":2,"reason":"与已有记忆内容相同"}
- If UPDATE: {"action":"UPDATE","targetIndex":3,"reason":"新增了额外细节",\
"mergedSummary":"合并后的完整摘要，保留新旧所有信息"}
- If NEW: {"action":"NEW","reason":"不同主题，无关联"}

Output ONLY the JSON object, no other text."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class IngestMessage:
    role: str
    content: str
    session_key: str
    turn_id: str
    owner: str = "agent:main"
    timestamp: int = 0


@dataclass
class DedupResult:
    action: Literal["DUPLICATE", "UPDATE", "NEW"]
    target_index: int | None = None
    reason: str = ""
    merged_summary: str | None = None


IngestAction = Literal["stored", "duplicate", "merged", "skipped", "error"]

OnChunksIngested = Callable[[str, bool], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# MemWorker
# ---------------------------------------------------------------------------


class MemWorker:
    """异步入库队列，按 §4.2 流程处理每条消息。"""

    def __init__(
        self,
        store: MemStore,
        embedder: MemEmbedder,
        *,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.60,
        on_chunks_ingested: OnChunksIngested | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.dedup_threshold = dedup_threshold
        self._on_chunks_ingested = on_chunks_ingested

        self._queue: deque[tuple[IngestMessage, bool]] = deque()
        self._processing = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        messages: list[IngestMessage],
        session_end: bool = False,
    ) -> dict[str, int]:
        """Enqueue messages for ingestion. Returns stats after processing."""
        for msg in messages:
            self._queue.append((msg, session_end))

        return await self._process_queue()

    # ------------------------------------------------------------------
    # Queue processor
    # ------------------------------------------------------------------

    async def _process_queue(self) -> dict[str, int]:
        async with self._lock:
            if self._processing:
                return {"queued": len(self._queue)}
            self._processing = True

        stats = {"stored": 0, "duplicate": 0, "merged": 0, "skipped": 0, "error": 0}
        last_session: str | None = None
        any_session_end = False

        try:
            while self._queue:
                msg, is_session_end = self._queue.popleft()
                last_session = msg.session_key
                any_session_end = any_session_end or is_session_end
                try:
                    action = await self._ingest_one(msg)
                    stats[action] += 1
                except Exception:
                    logger.exception("Failed to ingest message turn=%s", msg.turn_id)
                    stats["error"] += 1

            if last_session and self._on_chunks_ingested:
                try:
                    await self._on_chunks_ingested(last_session, any_session_end)
                except Exception:
                    logger.exception("on_chunks_ingested callback failed")
        finally:
            self._processing = False

        logger.info("Ingest batch: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Per-message pipeline
    # ------------------------------------------------------------------

    async def _ingest_one(self, msg: IngestMessage) -> IngestAction:
        content = msg.content.strip()
        if not content:
            return "skipped"

        # ① 精确 hash 去重
        existing = self.store.find_active_chunk_by_hash(content, msg.owner)
        if existing:
            logger.debug("Exact hash dup → %s", existing)
            return "skipped"

        chunk_id = uuid.uuid4().hex[:16]
        kind = "tool_result" if msg.role == "tool" else "paragraph"

        # ② LLM summary
        summary = await self._generate_summary(content)

        # ③ Embed summary
        embedding: list[float] | None = None
        try:
            embedding = await self.embedder.embed_query(summary)
        except Exception:
            logger.warning("Embedding failed for chunk=%s, storing without vector", chunk_id)

        # ④ 语义去重 (ANN + LLM judge)
        dedup_status: Literal["active", "duplicate", "merged"] = "active"
        dedup_target: str | None = None
        dedup_reason: str | None = None
        merged_from_old: str | None = None

        if embedding:
            candidates = self.store.ann_dedup_candidates(
                embedding, self.dedup_threshold, top_k=5, owner=msg.owner
            )
            if candidates:
                result = await self._judge_dedup(summary, content, candidates)
                if result and result.action == "DUPLICATE" and result.target_index is not None:
                    idx = result.target_index - 1
                    if 0 <= idx < len(candidates):
                        dedup_status = "duplicate"
                        dedup_target = candidates[idx].chunk_id
                        dedup_reason = result.reason
                        logger.debug("Dedup DUPLICATE → %s", dedup_target)

                if dedup_status == "active" and result and result.action == "UPDATE" and result.target_index is not None and result.merged_summary:
                    idx = result.target_index - 1
                    if 0 <= idx < len(candidates):
                        old_chunk_id = candidates[idx].chunk_id
                        summary = result.merged_summary
                        try:
                            embedding = await self.embedder.embed_query(summary)
                        except Exception:
                            logger.warning("Re-embed after merge failed")

                        self.store.mark_dedup_status(old_chunk_id, "merged", chunk_id, result.reason)
                        self.store.delete_chunk_embedding(old_chunk_id)
                        merged_from_old = old_chunk_id
                        dedup_reason = result.reason
                        logger.debug("Dedup UPDATE → old=%s retired", old_chunk_id)

        # ⑤ INSERT
        ts = msg.timestamp or _now_ms()
        chunk = Chunk(
            id=chunk_id,
            session_key=msg.session_key,
            turn_id=msg.turn_id,
            seq=0,
            role=msg.role,
            content=content,
            kind=kind,
            summary=summary,
            owner=msg.owner,
            content_hash=_content_hash(content),
            dedup_status=dedup_status,
            dedup_target=dedup_target,
            dedup_reason=dedup_reason,
            created_at=ts,
            updated_at=ts,
        )
        self.store.insert_chunk(chunk)

        if embedding and dedup_status == "active":
            self.store.upsert_chunk_embedding(chunk_id, embedding)

        logger.debug(
            "Stored chunk=%s role=%s dedup=%s len=%d",
            chunk_id, msg.role, dedup_status, len(content),
        )

        if dedup_status == "duplicate":
            return "duplicate"
        if merged_from_old:
            return "merged"
        return "stored"

    # ------------------------------------------------------------------
    # LLM: summary generation
    # ------------------------------------------------------------------

    async def _generate_summary(self, content: str) -> str:
        """Generate a concise summary via LLM. Falls back to truncation."""
        if not self.llm_api_key:
            return self._fallback_summary(content)

        try:
            text = content[:2000] if len(content) > 2000 else content
            result = await self._llm_chat(
                system=SUMMARY_SYSTEM_PROMPT,
                user=text,
                max_tokens=200,
                temperature=0.0,
            )
            return result.strip() if result else self._fallback_summary(content)
        except Exception:
            logger.warning("Summary LLM failed, using fallback")
            return self._fallback_summary(content)

    @staticmethod
    def _fallback_summary(content: str) -> str:
        lines = content.strip().split("\n")
        first = lines[0].strip() if lines else content[:120]
        if len(first) > 120:
            first = first[:117] + "..."
        return first

    # ------------------------------------------------------------------
    # LLM: dedup judge
    # ------------------------------------------------------------------

    async def _judge_dedup(
        self,
        new_summary: str,
        new_content: str,
        candidates: list[SearchHit],
    ) -> DedupResult | None:
        """Ask LLM to judge dedup (summary + content 前 300 字)."""
        if not self.llm_api_key:
            return None

        new_excerpt = new_content[:300]
        candidate_lines: list[str] = []
        for i, c in enumerate(candidates, 1):
            candidate_lines.append(
                f"{i}. Summary: {c.summary}\n   Excerpt: {c.content_excerpt[:300]}"
            )
        candidate_text = "\n".join(candidate_lines)

        user_msg = (
            f"NEW MEMORY:\nSummary: {new_summary}\nExcerpt: {new_excerpt}\n\n"
            f"EXISTING MEMORIES:\n{candidate_text}"
        )

        try:
            raw = await self._llm_chat(
                system=DEDUP_JUDGE_PROMPT,
                user=user_msg,
                max_tokens=300,
                temperature=0.0,
            )
            return self._parse_dedup_result(raw)
        except Exception:
            logger.warning("Dedup judge LLM failed, treating as NEW")
            return DedupResult(action="NEW", reason="llm_failed")

    @staticmethod
    def _parse_dedup_result(raw: str) -> DedupResult:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                obj = json.loads(match.group(0))
                action = obj.get("action", "NEW").upper()
                if action not in ("DUPLICATE", "UPDATE", "NEW"):
                    action = "NEW"
                return DedupResult(
                    action=action,
                    target_index=obj.get("targetIndex"),
                    reason=obj.get("reason", ""),
                    merged_summary=obj.get("mergedSummary"),
                )
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse dedup result: %s", raw[:200])
        return DedupResult(action="NEW", reason="parse_failed")

    # ------------------------------------------------------------------
    # LLM: generic chat completion
    # ------------------------------------------------------------------

    async def _llm_chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
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

        async with httpx.AsyncClient(timeout=30.0) as client:
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
        on_chunks_ingested: OnChunksIngested | None = None,
    ) -> MemWorker:
        """Build from the full `mem` config dict."""
        llm_cfg = config.get("llm", {})
        embedding_cfg = config.get("embedding", {})
        dedup_cfg = config.get("dedup", {})
        return cls(
            store=store,
            embedder=embedder,
            llm_base_url=llm_cfg.get("base_url") or embedding_cfg.get("base_url", ""),
            llm_api_key=llm_cfg.get("api_key") or embedding_cfg.get("api_key", ""),
            llm_model=llm_cfg.get("model") or "qwen-plus",
            dedup_threshold=dedup_cfg.get("similarity_threshold", 0.60),
            on_chunks_ingested=on_chunks_ingested,
        )
