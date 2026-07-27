"""记忆检索引擎 — 渐进式瀑布搜索

两层记忆架构:
  短期 = session history (不经过此模块)
  中期 = Task 摘要 + 挂靠 Chunk 片段

Skill 技能由 skills_scanner 静态扫描注入系统 Prompt，不经过此模块检索。

搜索流程 (瀑布式):
  ① 搜 Tasks — FTS + ANN → RRF
  ② Task 不够 min_task_hits → 补搜 orphan Chunks
  ③ 对命中的 Task 搜其下属 Chunks → 按 query 相关度取 top-N 拼接
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mem.embedder import MemEmbedder
from mem.recall_store import MemRecallStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RecallHit:
    chunk_id: str
    score: float
    summary: str = ""
    content_excerpt: str = ""
    role: str = ""
    session_key: str = ""
    task_id: str | None = None
    created_at: int = 0


@dataclass
class TaskGroup:
    task_id: str
    title: str
    summary: str
    session_date: str
    task_score: float
    chunks: list[RecallHit] = field(default_factory=list)


@dataclass
class RecallResult:
    task_groups: list[TaskGroup] = field(default_factory=list)
    orphan_hits: list[RecallHit] = field(default_factory=list)
    total_candidates: int = 0
    note: str = ""

    @property
    def hits(self) -> list[RecallHit]:
        """Flat list for backward compatibility."""
        result: list[RecallHit] = []
        for g in self.task_groups:
            result.append(RecallHit(
                chunk_id=f"task:{g.task_id}",
                score=g.task_score,
                summary=g.title,
                content_excerpt=g.summary[:600],
                role="task",
                task_id=g.task_id,
                created_at=0,
            ))
            result.extend(g.chunks)
        result.extend(self.orphan_hits)
        return result

    @property
    def has_content(self) -> bool:
        return bool(self.task_groups or self.orphan_hits)

    @property
    def max_score(self) -> float:
        scores: list[float] = [g.task_score for g in self.task_groups]
        scores.extend(h.score for h in self.orphan_hits)
        return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Pure functions: RRF, recency
# ---------------------------------------------------------------------------

def rrf_fuse(
    lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_list in lists:
        for rank, (item_id, _score) in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def apply_recency_decay(
    candidates: list[tuple[str, float, int]],
    half_life_days: float = 14.0,
    now_ms: int | None = None,
) -> list[tuple[str, float]]:
    current = now_ms or int(time.time() * 1000)
    half_life_ms = half_life_days * 24 * 3600 * 1000
    alpha = 0.3
    result: list[tuple[str, float]] = []
    for cid, score, created_at in candidates:
        age_ms = max(0, current - created_at)
        decay = 0.5 ** (age_ms / half_life_ms) if half_life_ms > 0 else 1.0
        adjusted = score * (alpha + (1 - alpha) * decay)
        result.append((cid, adjusted))
    return result


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[，。？！,?!]|在.+里|和|以及|跟|与")


def expand_query(query: str) -> list[str]:
    results = [query]
    parts = _SPLIT_RE.split(query)
    parts = [p.strip() for p in parts if len(p.strip()) > 2]
    if len(parts) > 1:
        results.extend(parts)
    return results[:4]


# ---------------------------------------------------------------------------
# MemRecall
# ---------------------------------------------------------------------------

class MemRecall:
    """渐进式瀑布检索引擎。"""

    def __init__(
        self,
        store: MemRecallStore,
        embedder: MemEmbedder,
        config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        self.store = store
        self.embedder = embedder
        cfg = config or {}
        recall = cfg.get("recall", {})
        self.max_task_results: int = recall.get("max_task_results", 5)
        self.min_task_hits: int = recall.get("min_task_hits", 3)
        self.chunks_per_task: int = recall.get("chunks_per_task", 3)
        self.max_orphan_chunks: int = recall.get("max_orphan_chunks", 5)
        from runtime.context_budget import resolve_budget
        self.budget_chars: int = recall.get(
            "budget_chars",
            resolve_budget(agent_id).memory_injection_chars,
        )
        self.min_task_score: float = recall.get("min_task_score", 0.3)
        self.rrf_k: int = recall.get("rrf_k", 60)
        self.recency_half_life_days: float = recall.get("recency_half_life_days", 14)
        self.min_inject_score: float = recall.get("min_inject_score", 0.015)

    # ------------------------------------------------------------------
    # Main search (waterfall)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        owner: str | None = None,
        max_results: int | None = None,
    ) -> RecallResult:
        if not query.strip():
            return RecallResult(note="empty query")

        sub_queries = expand_query(query)

        query_vectors: dict[str, list[float]] = {}
        for sq in sub_queries:
            try:
                query_vectors[sq] = await self.embedder.embed_query(sq)
            except Exception:
                logger.warning("Embed failed for sub-query: %s", sq)

        # ① 搜 Tasks
        task_hits = await self._search_tasks(query, sub_queries, query_vectors, owner)
        logger.debug("Recall: %d task hits", len(task_hits))

        # ② Task 不够 → 补搜 orphan Chunks
        orphan_hits: list[RecallHit] = []
        if len(task_hits) < self.min_task_hits:
            orphan_hits = await self._search_orphan_chunks(query, sub_queries, session_id, query_vectors, owner)
            logger.debug("Recall: %d orphan chunk hits", len(orphan_hits))

        # ③ 对命中的 Tasks 搜下属 Chunks → 组装 TaskGroup
        task_ids = [tid for tid, _, _ in task_hits]
        chunk_map = await self._search_chunks_for_tasks(task_ids, query, sub_queries, query_vectors, owner)

        task_groups: list[TaskGroup] = []
        chars_used = 0
        for tid, score, task_obj in task_hits:
            entry_chars = len(task_obj.summary or "")
            if chars_used + entry_chars > self.budget_chars and task_groups:
                break

            date_str = _format_date(task_obj.started_at)
            group = TaskGroup(
                task_id=tid,
                title=task_obj.title or "",
                summary=(task_obj.summary or "")[:600],
                session_date=date_str,
                task_score=round(score, 3),
                chunks=chunk_map.get(tid, []),
            )
            task_groups.append(group)
            chars_used += entry_chars + sum(
                len(c.content_excerpt) for c in group.chunks
            )

        # ④ 过滤低分结果
        min_s = self.min_inject_score
        orphan_hits = [h for h in orphan_hits if h.score >= min_s]
        for g in task_groups:
            g.chunks = [c for c in g.chunks if c.score >= min_s]

        # ④-b orphan chunk 纳入总预算
        budget_remaining = max(0, self.budget_chars - chars_used)
        trimmed_orphans: list[RecallHit] = []
        orphan_chars = 0
        for h in orphan_hits:
            h_chars = len(h.content_excerpt) + len(h.summary)
            if orphan_chars + h_chars > budget_remaining and trimmed_orphans:
                break
            trimmed_orphans.append(h)
            orphan_chars += h_chars
        orphan_hits = trimmed_orphans

        # ⑤ max_results 截断
        if max_results is not None:
            total_hits = 0
            trimmed_groups: list[TaskGroup] = []
            for g in task_groups:
                group_count = 1 + len(g.chunks)
                if total_hits + group_count > max_results:
                    remaining = max_results - total_hits
                    if remaining > 1:
                        g.chunks = g.chunks[:remaining - 1]
                        trimmed_groups.append(g)
                    break
                trimmed_groups.append(g)
                total_hits += group_count
            task_groups = trimmed_groups

            remaining = max(0, max_results - total_hits)
            orphan_hits = orphan_hits[:remaining]

        total = len(task_hits) + len(orphan_hits)
        note_parts = []
        if task_groups:
            note_parts.append(f"tasks:{len(task_groups)}")
        if orphan_hits:
            note_parts.append(f"orphans:{len(orphan_hits)}")

        return RecallResult(
            task_groups=task_groups,
            orphan_hits=orphan_hits,
            total_candidates=total,
            note=",".join(note_parts) or "empty",
        )

    # ------------------------------------------------------------------
    # ① Task search (FTS + ANN → RRF)
    # ------------------------------------------------------------------

    async def _search_tasks(
        self, query: str, sub_queries: list[str],
        query_vectors: dict[str, list[float]] | None = None,
        owner: str | None = None,
    ) -> list[tuple[str, float, Any]]:
        pool_size = self.max_task_results * 5
        qv = query_vectors or {}

        fts_hits = self.store.fts_search_tasks(query, limit=pool_size, owner=owner)
        fts_ranked = [(h.task_id, h.score) for h in fts_hits]

        vec_scores: dict[str, float] = {}
        for sub_q in sub_queries:
            q_vec = qv.get(sub_q)
            if not q_vec:
                continue
            try:
                ann_hits = self.store.ann_search_tasks(q_vec, top_k=pool_size, owner=owner)
                for h in ann_hits:
                    vec_scores[h.task_id] = max(vec_scores.get(h.task_id, 0.0), h.score)
            except Exception:
                logger.warning("Task vector search failed for: %s", sub_q)

        vec_ranked = sorted(vec_scores.items(), key=lambda x: -x[1])
        rrf_scores = rrf_fuse([fts_ranked, vec_ranked], k=self.rrf_k)
        if not rrf_scores:
            return []

        sorted_tasks = sorted(rrf_scores.items(), key=lambda x: -x[1])
        results: list[tuple[str, float, Any]] = []
        for tid, score in sorted_tasks[:self.max_task_results]:
            if score < self.min_task_score:
                break
            task = self.store.get_task(tid)
            if task and task.status == "completed":
                results.append((tid, score, task))
        return results

    # ------------------------------------------------------------------
    # ② Orphan Chunk search (task_id IS NULL)
    # ------------------------------------------------------------------

    async def _search_orphan_chunks(
        self, query: str, sub_queries: list[str], exclude_session: str | None,
        query_vectors: dict[str, list[float]] | None = None,
        owner: str | None = None,
    ) -> list[RecallHit]:
        pool_size = self.max_orphan_chunks * 5
        qv = query_vectors or {}

        fts_hits = self.store.fts_search_orphan_chunks(
            query, limit=pool_size, exclude_session=exclude_session, owner=owner,
        )
        fts_ranked = [(h.chunk_id, h.score) for h in fts_hits]

        vec_scores: dict[str, float] = {}
        for sub_q in sub_queries:
            q_vec = qv.get(sub_q)
            if not q_vec:
                continue
            try:
                ann_hits = self.store.ann_search_orphan_chunks(
                    q_vec, top_k=pool_size, exclude_session=exclude_session, owner=owner,
                )
                for h in ann_hits:
                    vec_scores[h.chunk_id] = max(vec_scores.get(h.chunk_id, 0.0), h.score)
            except Exception:
                logger.warning("Orphan chunk vector search failed for: %s", sub_q)

        vec_ranked = sorted(vec_scores.items(), key=lambda x: -x[1])
        rrf_scores = rrf_fuse([fts_ranked, vec_ranked], k=self.rrf_k)
        if not rrf_scores:
            return []

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: -x[1])

        with_ts: list[tuple[str, float, int]] = []
        chunk_cache: dict[str, Any] = {}
        for cid, score in sorted_ids[:self.max_orphan_chunks * 2]:
            chunk = self.store.get_chunk(cid)
            if chunk:
                chunk_cache[cid] = chunk
                with_ts.append((cid, score, chunk.created_at))

        decayed = apply_recency_decay(with_ts, half_life_days=self.recency_half_life_days)
        decayed.sort(key=lambda x: -x[1])

        results: list[RecallHit] = []
        for cid, score in decayed[:self.max_orphan_chunks]:
            chunk = chunk_cache.get(cid)
            if chunk:
                results.append(RecallHit(
                    chunk_id=cid,
                    score=round(score, 3),
                    summary=chunk.summary or "",
                    content_excerpt=chunk.content[:300],
                    role=chunk.role,
                    session_key=chunk.session_key,
                    task_id=None,
                    created_at=chunk.created_at,
                ))
        return results

    # ------------------------------------------------------------------
    # ③ Chunks under Tasks (group by task_id, top-N per task)
    # ------------------------------------------------------------------

    async def _search_chunks_for_tasks(
        self,
        task_ids: list[str],
        query: str,
        sub_queries: list[str],
        query_vectors: dict[str, list[float]] | None = None,
        owner: str | None = None,
    ) -> dict[str, list[RecallHit]]:
        if not task_ids:
            return {}

        pool_size = len(task_ids) * self.chunks_per_task * 3
        qv = query_vectors or {}

        fts_hits = self.store.fts_search_chunks_in_tasks(query, task_ids, limit=pool_size, owner=owner)
        fts_ranked = [(h.chunk_id, h.score) for h in fts_hits]

        vec_scores: dict[str, float] = {}
        for sub_q in sub_queries:
            q_vec = qv.get(sub_q)
            if not q_vec:
                continue
            try:
                ann_hits = self.store.ann_search_chunks_in_tasks(q_vec, task_ids, top_k=pool_size, owner=owner)
                for h in ann_hits:
                    vec_scores[h.chunk_id] = max(vec_scores.get(h.chunk_id, 0.0), h.score)
            except Exception:
                logger.warning("Task-chunk vector search failed for: %s", sub_q)

        vec_ranked = sorted(vec_scores.items(), key=lambda x: -x[1])
        rrf_scores = rrf_fuse([fts_ranked, vec_ranked], k=self.rrf_k)

        fts_lookup = {h.chunk_id: h for h in fts_hits}

        # group by task_id
        task_chunks: dict[str, list[tuple[str, float]]] = {tid: [] for tid in task_ids}
        for cid, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            hit = fts_lookup.get(cid)
            tid = hit.task_id if hit else None
            if not tid:
                chunk = self.store.get_chunk(cid)
                tid = chunk.task_id if chunk else None
            if tid and tid in task_chunks:
                task_chunks[tid].append((cid, score))

        result: dict[str, list[RecallHit]] = {}
        for tid, entries in task_chunks.items():
            hits: list[RecallHit] = []
            for cid, score in entries[:self.chunks_per_task]:
                chunk = self.store.get_chunk(cid)
                if chunk:
                    hits.append(RecallHit(
                        chunk_id=cid,
                        score=round(score, 3),
                        summary=chunk.summary or "",
                        content_excerpt=chunk.content[:200],
                        role=chunk.role,
                        session_key=chunk.session_key,
                        task_id=tid,
                        created_at=chunk.created_at,
                    ))
            if hits:
                result[tid] = hits
        return result

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        store: MemRecallStore,
        embedder: MemEmbedder,
        agent_id: str | None = None,
    ) -> MemRecall:
        return cls(store=store, embedder=embedder, config=config, agent_id=agent_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_date(ts_ms: int) -> str:
    if not ts_ms:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromtimestamp(ts_ms / 1000)
        return dt.strftime("%-m/%d")
    except Exception:
        return ""
