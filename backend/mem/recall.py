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
import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mem.embedder import MemEmbedder
from mem.recall_store import MemRecallStore
from mem.reranker import MemReranker

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
    ranked_hits: list[RecallHit] = field(default_factory=list)
    retrieved_task_ids: list[str] = field(default_factory=list)
    orphan_search_triggered: bool = False
    expanded_task_count: int = 0
    candidate_chunk_count: int = 0
    total_candidates: int = 0
    note: str = ""

    @property
    def hits(self) -> list[RecallHit]:
        """Final globally ranked chunks; Task metadata does not consume slots."""
        return list(self.ranked_hits)

    @property
    def has_content(self) -> bool:
        return bool(self.ranked_hits)

    @property
    def max_score(self) -> float:
        scores = [hit.score for hit in self.ranked_hits]
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


def _rank_chunk_candidates(
    candidates: list[RecallHit],
) -> list[RecallHit]:
    """Deduplicate chunks without using their parent Task scores."""
    ranked_by_id: dict[str, RecallHit] = {}
    for hit in candidates:
        candidate = RecallHit(
            chunk_id=hit.chunk_id,
            score=hit.score,
            summary=hit.summary,
            content_excerpt=hit.content_excerpt,
            role=hit.role,
            session_key=hit.session_key,
            task_id=hit.task_id,
            created_at=hit.created_at,
        )
        previous = ranked_by_id.get(hit.chunk_id)
        if previous is None or candidate.score > previous.score:
            ranked_by_id[hit.chunk_id] = candidate
    return sorted(
        ranked_by_id.values(),
        key=lambda hit: (-hit.score, -hit.created_at, hit.chunk_id),
    )


def _trim_ranked_hits(
    ranked_hits: list[RecallHit],
    *,
    max_results: int,
    budget_chars: int,
) -> list[RecallHit]:
    selected: list[RecallHit] = []
    chars_used = 0
    for hit in ranked_hits:
        if len(selected) >= max_results:
            break
        hit_chars = len(hit.summary) + len(hit.content_excerpt)
        if selected and chars_used + hit_chars > budget_chars:
            break
        selected.append(hit)
        chars_used += hit_chars
    return selected


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
        self.chunks_per_task: int = recall.get("chunks_per_task", 5)
        self.max_orphan_chunks: int = recall.get("max_orphan_chunks", 5)
        self.final_chunk_top_k: int = recall.get("final_chunk_top_k", 5)
        from runtime.context_budget import resolve_budget
        self.budget_chars: int = recall.get(
            "budget_chars",
            resolve_budget(agent_id).memory_injection_chars,
        )
        self.min_task_score: float = recall.get("min_task_score", 0.015)
        self.rrf_k: int = recall.get("rrf_k", 60)
        self.recency_half_life_days: float = recall.get("recency_half_life_days", 14)
        self.min_inject_score: float = recall.get("min_inject_score", 0.015)
        self.rerank_enabled: bool = bool(recall.get("rerank_enabled", False))
        self.rerank_model: str = recall.get("rerank_model", "BAAI/bge-reranker-v2-m3")
        self.rerank_device: str = recall.get("rerank_device", "cpu")
        self.reranker = (
            MemReranker(self.rerank_model, device=self.rerank_device)
            if self.rerank_enabled else None
        )

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

        # ③ Expand every matched Task independently.
        task_ids = [tid for tid, _, _ in task_hits]
        chunk_map = await self._search_chunks_for_tasks(task_ids, query, sub_queries, query_vectors, owner)

        # ④ Merge Task chunks and orphan chunks. Task scores are not reused.
        candidates = [
            hit
            for task_id in task_ids
            for hit in chunk_map.get(task_id, [])
        ]
        candidates.extend(orphan_hits)
        ranked_candidates = _rank_chunk_candidates(candidates)
        ranked_candidates = [
            hit for hit in ranked_candidates if hit.score >= self.min_inject_score
        ]

        if self.reranker and ranked_candidates:
            rerank_inputs = [
                (hit.chunk_id, "\n".join(p for p in (hit.summary, hit.content_excerpt) if p))
                for hit in ranked_candidates
            ]
            scores = await asyncio.to_thread(self.reranker.score, query, rerank_inputs)
            reranked: list[RecallHit] = []
            for hit, score in zip(ranked_candidates, scores):
                reranked.append(RecallHit(
                    chunk_id=hit.chunk_id, score=score, summary=hit.summary,
                    content_excerpt=hit.content_excerpt, role=hit.role,
                    session_key=hit.session_key, task_id=hit.task_id,
                    created_at=hit.created_at,
                ))
            ranked_candidates = sorted(
                reranked, key=lambda hit: (-hit.score, -hit.created_at, hit.chunk_id)
            )

        final_limit = self.final_chunk_top_k
        if max_results is not None:
            final_limit = min(final_limit, max_results)
        ranked_hits = _trim_ranked_hits(
            ranked_candidates,
            max_results=max(0, final_limit),
            budget_chars=self.budget_chars,
        )

        # ⑤ Restore grouping metadata without changing global chunk order.
        selected_by_task: dict[str, list[RecallHit]] = {}
        selected_orphans: list[RecallHit] = []
        for hit in ranked_hits:
            if hit.task_id:
                selected_by_task.setdefault(hit.task_id, []).append(hit)
            else:
                selected_orphans.append(hit)

        task_groups: list[TaskGroup] = []
        for tid, score, task_obj in task_hits:
            selected = selected_by_task.get(tid)
            if not selected:
                continue
            task_groups.append(TaskGroup(
                task_id=tid,
                title=task_obj.title or "",
                summary=(task_obj.summary or "")[:600],
                session_date=_format_date(task_obj.started_at),
                task_score=round(score, 3),
                chunks=selected,
            ))

        total = len(ranked_candidates)
        note_parts = []
        if task_groups:
            note_parts.append(f"tasks:{len(task_groups)}")
        if selected_orphans:
            note_parts.append(f"orphans:{len(selected_orphans)}")
        note_parts.append(f"chunks:{len(ranked_hits)}")

        return RecallResult(
            task_groups=task_groups,
            orphan_hits=selected_orphans,
            ranked_hits=ranked_hits,
            retrieved_task_ids=task_ids,
            orphan_search_triggered=len(task_hits) < self.min_task_hits,
            expanded_task_count=len(task_hits),
            candidate_chunk_count=len(ranked_candidates),
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

        qv = query_vectors or {}
        # Task IDs only define the search scope. All scoped chunks share one
        # FTS/ANN ranking, so rank positions are comparable across Tasks.
        fts_hits = self.store.fts_search_chunks_in_tasks(
            query, task_ids, limit=None, owner=owner,
        )
        fts_ranked = [(hit.chunk_id, hit.score) for hit in fts_hits]

        vec_scores: dict[str, float] = {}
        for sub_q in sub_queries:
            q_vec = qv.get(sub_q)
            if not q_vec:
                continue
            try:
                ann_hits = self.store.exact_search_chunks_in_tasks(
                    q_vec, task_ids, top_k=None, owner=owner,
                )
                for hit in ann_hits:
                    vec_scores[hit.chunk_id] = max(
                        vec_scores.get(hit.chunk_id, 0.0), hit.score
                    )
            except Exception:
                logger.warning("Task-chunk vector search failed for: %s", sub_q)

        vec_ranked = sorted(vec_scores.items(), key=lambda item: -item[1])
        rrf_scores = rrf_fuse([fts_ranked, vec_ranked], k=self.rrf_k)
        candidate_limit = len(task_ids) * self.chunks_per_task
        entries = sorted(rrf_scores.items(), key=lambda item: -item[1])[
            :candidate_limit
        ]

        result: dict[str, list[RecallHit]] = {}
        for cid, score in entries:
            chunk = self.store.get_chunk(cid)
            if not chunk or chunk.task_id not in task_ids:
                continue
            result.setdefault(chunk.task_id, []).append(RecallHit(
                chunk_id=cid,
                score=score,
                summary=chunk.summary or "",
                content_excerpt=chunk.content[:300],
                role=chunk.role,
                session_key=chunk.session_key,
                task_id=chunk.task_id,
                created_at=chunk.created_at,
            ))
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
