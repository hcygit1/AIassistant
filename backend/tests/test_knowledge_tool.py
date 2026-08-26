from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tools.knowledge_tool import KnowledgeSearchTool
from tools.rag_mcp_client import (
    KnowledgeSyncResult,
    RagMcpClient,
    RagMcpSettings,
    resolve_rag_mcp_settings,
)


class KnowledgeSearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_search_uses_persistent_mcp_client(self) -> None:
        client = Mock()
        client.settings = SimpleNamespace(collection_name="pipixia_{agent_id}")
        client.sync_directory = AsyncMock(return_value=KnowledgeSyncResult(1, 1, 0))
        client.query = AsyncMock(return_value="RAG result with citations")
        tool = KnowledgeSearchTool(agent_dir="unused", agent_id="research")

        with patch(
            "tools.rag_mcp_client.get_rag_mcp_client",
            return_value=client,
        ):
            result = await tool._arun("semantic query", top_k=5)

        self.assertEqual(result, "RAG result with citations")
        client.sync_directory.assert_awaited_once_with(
            Path("unused") / "knowledge",
            collection_name="pipixia_research",
        )
        client.query.assert_awaited_once_with(
            "semantic query",
            collection_name="pipixia_research",
            top_k=5,
        )

    async def test_search_returns_readable_error(self) -> None:
        client = Mock()
        client.settings = SimpleNamespace(collection_name="pipixia_{agent_id}")
        client.sync_directory = AsyncMock(return_value=KnowledgeSyncResult(1, 0, 1))
        client.query = AsyncMock(side_effect=RuntimeError("server unavailable"))
        tool = KnowledgeSearchTool(agent_dir="unused")

        with patch(
            "tools.rag_mcp_client.get_rag_mcp_client",
            return_value=client,
        ), patch("config.get_config", return_value={"app": {"locale": "zh-CN"}}):
            result = await tool._arun("query")

        self.assertIn("知识库检索失败", result)
        self.assertIn("server unavailable", result)

    async def test_empty_directory_does_not_query_mcp(self) -> None:
        client = Mock()
        client.settings = SimpleNamespace(collection_name="pipixia_{agent_id}")
        client.sync_directory = AsyncMock(return_value=KnowledgeSyncResult(0, 0, 0))
        client.query = AsyncMock()
        tool = KnowledgeSearchTool(agent_dir="unused")

        with patch(
            "tools.rag_mcp_client.get_rag_mcp_client",
            return_value=client,
        ), patch("config.get_config", return_value={"app": {"locale": "zh-CN"}}):
            result = await tool._arun("query")

        self.assertIn("知识库目录为空", result)
        client.query.assert_not_awaited()


class RagMcpSettingsTests(unittest.TestCase):
    def test_default_paths_point_to_sibling_rag_project(self) -> None:
        settings = resolve_rag_mcp_settings({"tools": {"knowledge": {}}})

        self.assertEqual(settings.project_path.name, "MODULAR-RAG-MCP-SERVER")
        self.assertEqual(settings.config_path.name, "settings.yaml")
        self.assertEqual(settings.collection_name, "pipixia_{agent_id}")
        self.assertEqual(settings.timeout_seconds, 120)

    def test_explicit_collection_and_timeout_are_used(self) -> None:
        settings = resolve_rag_mcp_settings({
            "tools": {
                "knowledge": {
                    "collectionName": "personal_docs",
                    "timeoutSeconds": 45,
                }
            }
        })

        self.assertEqual(settings.collection_name, "personal_docs")
        self.assertEqual(settings.timeout_seconds, 45)


class RagMcpDirectorySyncTests(unittest.TestCase):
    def test_only_supported_changed_files_are_ingested(self) -> None:
        settings = RagMcpSettings(
            project_path=Path("."),
            python_path=Path("python"),
            config_path=Path("settings.yaml"),
            collection_name="pipixia_main",
            timeout_seconds=120,
        )
        client = RagMcpClient(settings)
        response = SimpleNamespace(
            isError=False,
            structuredContent={"ingested_count": 1, "skipped_count": 0},
            content=[],
        )
        client._call_tool_sync = Mock(return_value=response)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("knowledge", encoding="utf-8")
            (root / "ignored.rst").write_text("ignored", encoding="utf-8")

            first = client.sync_directory_sync(root)
            second = client.sync_directory_sync(root)

        self.assertEqual(first, KnowledgeSyncResult(1, 1, 0))
        self.assertEqual(second, KnowledgeSyncResult(1, 0, 1))
        client._call_tool_sync.assert_called_once()
        tool_name, arguments = client._call_tool_sync.call_args.args
        self.assertEqual(tool_name, "ingest_documents_normal")
        self.assertEqual(arguments["collection_name"], "pipixia_main")
        self.assertEqual(len(arguments["file_paths"]), 1)

    def test_query_result_keeps_structured_citations(self) -> None:
        result = SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="text", text="retrieved content")],
            structuredContent={
                "citations": [{
                    "source": "knowledge/design.md",
                    "page": 2,
                    "chunk_id": "chunk-1",
                }]
            },
        )

        text = RagMcpClient._format_result(result)

        self.assertIn("retrieved content", text)
        self.assertIn("### 引用来源", text)
        self.assertIn("knowledge/design.md", text)


if __name__ == "__main__":
    unittest.main()
