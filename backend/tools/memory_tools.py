"""Memory tools — mem_search, mem_get

mem_search: Hybrid FTS5 + sqlite-vec ANN search via MemRecall
mem_get:    Read a specific chunk by ID from MemStore
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# mem_search — Hybrid recall (FTS5 + ANN + RRF + recency)
# ---------------------------------------------------------------------------

class MemSearchInput(BaseModel):
    query: str = Field(description="搜索查询（自然语言）")
    max_results: int = Field(default=8, description="最多返回结果数（默认 8）")


class MemSearchTool(BaseTool):
    name: str = "memory_search"
    description: str = (
        "记忆检索：搜索历史对话中的事件、决策、配置等信息。"
        "返回相关片段（含 chunk_id、摘要、角色、时间）。"
        "当你发现当前上下文不足以回答用户问题，且需要回忆过往具体信息时使用。"
        "不要每次对话都调用，只在确实需要时使用。"
    )
    args_schema: type[BaseModel] = MemSearchInput
    agent_id: str = ""

    def _run(self, query: str, max_results: int = 8) -> str:
        raise NotImplementedError("Use _arun for async execution")

    async def _arun(self, query: str, max_results: int = 8) -> str:
        try:
            from runtime.agent import agent_manager
            recall = agent_manager.mem_recalls.get(self.agent_id)
        except Exception:
            recall = None

        if not recall:
            return "记忆系统未初始化。"

        owner = f"agent:{self.agent_id}" if self.agent_id else None
        result = await recall.search(query, owner=owner, max_results=max_results)

        if not result.hits:
            return f"未找到与 '{query}' 相关的记忆。"

        lines: list[str] = []
        for h in result.hits:
            ts = ""
            if h.created_at:
                from datetime import datetime
                ts = datetime.fromtimestamp(h.created_at).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"--- chunk_id: {h.chunk_id} | score: {h.score:.3f}"
                f" | role: {h.role} | time: {ts} ---"
            )
            lines.append(h.summary or "(无摘要)")
            if h.content_excerpt:
                lines.append(h.content_excerpt)
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# mem_get — Read a specific chunk by ID
# ---------------------------------------------------------------------------

class MemGetInput(BaseModel):
    chunk_id: str = Field(description="Chunk ID（由 memory_search 返回）")


class MemGetTool(BaseTool):
    name: str = "memory_get"
    description: str = (
        "根据 chunk_id 读取完整记忆内容。"
        "先用 memory_search 找到相关记忆，再用此工具获取全文。"
    )
    args_schema: type[BaseModel] = MemGetInput
    agent_id: str = ""

    def _run(self, chunk_id: str) -> str:
        try:
            from runtime.agent import agent_manager
            store = agent_manager.mem_stores.get(self.agent_id)
        except Exception:
            store = None

        if not store:
            return "记忆系统未初始化。"

        chunk = store.get_chunk(chunk_id)
        if not chunk:
            return f"未找到 chunk: {chunk_id}"

        from datetime import datetime
        ts = datetime.fromtimestamp(chunk.created_at).strftime("%Y-%m-%d %H:%M")

        parts = [
            f"[chunk_id: {chunk.id}]",
            f"[role: {chunk.role} | session: {chunk.session_key} | time: {ts}]",
        ]
        if chunk.summary:
            parts.append(f"[summary] {chunk.summary}")
        parts.append("")
        parts.append(chunk.content)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_memory_tools(agent_id: str = "", **_kwargs: Any) -> list[BaseTool]:
    return [
        MemSearchTool(agent_id=agent_id),
        MemGetTool(agent_id=agent_id),
    ]
