"""知识库工具 x1: search_knowledge_base"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="搜索查询")
    top_k: int = Field(default=3, description="返回结果数量（默认 3）")


class KnowledgeSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "通过 MCP-RAG 在个人知识库中执行 Dense + FTS5 混合检索，"
        "返回最相关的文档片段与溯源引用。"
    )
    args_schema: type[BaseModel] = KnowledgeSearchInput
    agent_dir: str = ""
    agent_id: str = "main"

    def _run(self, query: str, top_k: int = 3) -> str:
        from tools.rag_mcp_client import get_rag_mcp_client

        try:
            client = get_rag_mcp_client()
            collection_name = self._collection_name(client)
            sync = client.sync_directory_sync(
                Path(self.agent_dir) / "knowledge",
                collection_name=collection_name,
            )
            if sync.file_count == 0:
                return self._empty_message()
            return client.query_sync(
                query,
                collection_name=collection_name,
                top_k=top_k,
            )
        except Exception as exc:
            return self._error_message(exc)

    async def _arun(self, query: str, top_k: int = 3) -> str:
        from tools.rag_mcp_client import get_rag_mcp_client

        try:
            client = get_rag_mcp_client()
            collection_name = self._collection_name(client)
            sync = await client.sync_directory(
                Path(self.agent_dir) / "knowledge",
                collection_name=collection_name,
            )
            if sync.file_count == 0:
                return self._empty_message()
            return await client.query(
                query,
                collection_name=collection_name,
                top_k=top_k,
            )
        except Exception as exc:
            return self._error_message(exc)

    def _collection_name(self, client: object) -> str | None:
        settings = getattr(client, "settings", None)
        template = getattr(settings, "collection_name", None)
        if not template:
            return None
        return str(template).replace("{agent_id}", self.agent_id)

    @staticmethod
    def _empty_message() -> str:
        from config import get_config

        locale = get_config().get("app", {}).get("locale", "zh-CN")
        return (
            "知识库目录为空。请将 PDF、Markdown 或 TXT 文档放入 knowledge/ 目录。"
            if locale == "zh-CN"
            else "Knowledge base is empty. Add PDF, Markdown, or TXT files to knowledge/."
        )

    @staticmethod
    def _error_message(error: Exception) -> str:
        from config import get_config
        locale = get_config().get("app", {}).get("locale", "zh-CN")
        return (
            f"知识库检索失败：{error}"
            if locale == "zh-CN"
            else f"Knowledge base search failed: {error}"
        )


def get_knowledge_tools(agent_dir: str, agent_id: str = "main") -> list[BaseTool]:
    return [
        KnowledgeSearchTool(agent_dir=agent_dir, agent_id=agent_id),
    ]
