"""Persistent MCP client for the external MODULAR-RAG service."""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RagMcpSettings:
    project_path: Path
    python_path: Path
    config_path: Path
    collection_name: str | None
    timeout_seconds: float


def resolve_rag_mcp_settings(config: dict[str, Any]) -> RagMcpSettings:
    """Resolve MCP-RAG paths, using the sibling repository by default."""
    from config import PROJECT_ROOT

    knowledge = ((config.get("tools") or {}).get("knowledge") or {})
    raw_project = str(knowledge.get("projectPath") or "").strip()
    project_path = (
        Path(raw_project)
        if raw_project
        else PROJECT_ROOT.parent / "MODULAR-RAG-MCP-SERVER"
    )
    if not project_path.is_absolute():
        project_path = PROJECT_ROOT / project_path
    project_path = project_path.resolve()

    raw_python = str(knowledge.get("pythonPath") or "").strip()
    if raw_python:
        python_path = Path(raw_python)
        if not python_path.is_absolute():
            python_path = PROJECT_ROOT / python_path
    else:
        executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        venv_python = project_path / ".venv" / executable
        python_path = venv_python if venv_python.exists() else Path(sys.executable)

    raw_config = str(knowledge.get("configPath") or "").strip()
    config_path = (
        Path(raw_config)
        if raw_config
        else project_path / "config" / "settings.yaml"
    )
    if not config_path.is_absolute():
        config_path = project_path / config_path

    collection = str(
        knowledge.get("collectionName", "pipixia_{agent_id}") or ""
    ).strip() or None
    try:
        timeout = max(1.0, float(knowledge.get("timeoutSeconds", 120)))
    except (TypeError, ValueError):
        timeout = 120.0

    return RagMcpSettings(
        project_path=project_path,
        python_path=python_path.resolve(),
        config_path=config_path.resolve(),
        collection_name=collection,
        timeout_seconds=timeout,
    )


@dataclass
class KnowledgeSyncResult:
    file_count: int
    ingested_count: int
    unchanged_count: int


@dataclass
class _ToolRequest:
    tool_name: str
    arguments: dict[str, Any]
    future: Future[Any]


_STOP = object()


class RagMcpClient:
    """Keep one MCP stdio process alive so the BGE model is loaded only once."""

    def __init__(self, settings: RagMcpSettings) -> None:
        self.settings = settings
        self._state_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._requests: queue.Queue[_ToolRequest | object] | None = None
        self._ready: Future[None] | None = None
        self._file_cache: dict[tuple[str, str], tuple[int, int]] = {}

    async def query(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        top_k: int = 3,
    ) -> str:
        return await asyncio.to_thread(
            self.query_sync,
            query,
            collection_name=collection_name,
            top_k=top_k,
        )

    def query_sync(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        top_k: int = 3,
    ) -> str:
        result = self._call_tool_sync(
            "query_knowledge_hub",
            {
                "query": query,
                "collection_name": collection_name or self.settings.collection_name,
                "top_k": top_k,
            },
        )
        return self._format_result(result)

    async def sync_directory(
        self,
        directory: Path,
        *,
        collection_name: str | None = None,
    ) -> KnowledgeSyncResult:
        return await asyncio.to_thread(
            self.sync_directory_sync,
            directory,
            collection_name=collection_name,
        )

    def sync_directory_sync(
        self,
        directory: Path,
        *,
        collection_name: str | None = None,
    ) -> KnowledgeSyncResult:
        collection = collection_name or self.settings.collection_name
        supported = {".pdf", ".md", ".txt"}
        if not directory.is_dir():
            return KnowledgeSyncResult(0, 0, 0)

        files = sorted(
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in supported
        )
        unchanged = 0
        with self._sync_lock:
            changed: list[tuple[Path, tuple[str, str], tuple[int, int]]] = []
            for path in files:
                stat = path.stat()
                fingerprint = (stat.st_mtime_ns, stat.st_size)
                cache_key = (collection or "__default__", str(path))
                if self._file_cache.get(cache_key) == fingerprint:
                    unchanged += 1
                    continue
                changed.append((path, cache_key, fingerprint))

            if changed:
                result = self._call_tool_sync(
                    "ingest_documents_normal",
                    {
                        "file_paths": [str(item[0]) for item in changed],
                        "collection_name": collection,
                    },
                )
                self._raise_for_error(result)
                structured = getattr(result, "structuredContent", None) or {}
                ingested = int(structured.get("ingested_count", 0))
                unchanged += int(structured.get("skipped_count", 0))
                for _, cache_key, fingerprint in changed:
                    self._file_cache[cache_key] = fingerprint
            else:
                ingested = 0

        return KnowledgeSyncResult(len(files), ingested, unchanged)

    def _call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        requests = self._ensure_worker()
        future: Future[Any] = Future()
        requests.put(_ToolRequest(tool_name, arguments, future))
        try:
            return future.result(timeout=self.settings.timeout_seconds)
        except TimeoutError as exc:
            raise RuntimeError(
                f"MCP-RAG 调用超时（{self.settings.timeout_seconds:g} 秒）"
            ) from exc

    def _ensure_worker(self) -> queue.Queue[_ToolRequest | object]:
        with self._state_lock:
            if self._thread is None or not self._thread.is_alive():
                self._validate_paths()
                self._requests = queue.Queue()
                self._ready = Future()
                self._thread = threading.Thread(
                    target=self._thread_main,
                    args=(self._requests, self._ready),
                    name="pipixia-rag-mcp",
                    daemon=True,
                )
                self._thread.start()
            requests = self._requests
            ready = self._ready

        if requests is None or ready is None:
            raise RuntimeError("MCP-RAG 客户端初始化失败")
        try:
            ready.result(timeout=self.settings.timeout_seconds)
        except TimeoutError as exc:
            raise RuntimeError(
                f"MCP-RAG 服务启动超时（{self.settings.timeout_seconds:g} 秒）"
            ) from exc
        return requests

    def _validate_paths(self) -> None:
        if not self.settings.project_path.is_dir():
            raise RuntimeError(f"MCP-RAG 项目不存在：{self.settings.project_path}")
        if not self.settings.python_path.is_file():
            raise RuntimeError(f"MCP-RAG Python 不存在：{self.settings.python_path}")
        if not self.settings.config_path.is_file():
            raise RuntimeError(f"MCP-RAG 配置不存在：{self.settings.config_path}")

    def _thread_main(
        self,
        requests: queue.Queue[_ToolRequest | object],
        ready: Future[None],
    ) -> None:
        try:
            asyncio.run(self._serve(requests, ready))
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            self._fail_pending(requests, exc)

    async def _serve(
        self,
        requests: queue.Queue[_ToolRequest | object],
        ready: Future[None],
    ) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        env["MODULAR_RAG_CONFIG_PATH"] = str(self.settings.config_path)
        # The embedding weights are local.  Prevent Hugging Face from doing
        # a metadata HEAD request on every fresh MCP subprocess, which can
        # block for minutes on networks that cannot reach huggingface.co.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        params = StdioServerParameters(
            command=str(self.settings.python_path),
            args=["-m", "src.mcp_server.server"],
            cwd=str(self.settings.project_path),
            env=env,
            encoding="utf-8",
        )

        log_path = self.settings.project_path / "logs" / "pipixia-mcp.stderr.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as streams, ClientSession(
                *streams
            ) as session:
                await session.initialize()
                ready.set_result(None)
                while True:
                    request = await asyncio.to_thread(requests.get)
                    if request is _STOP:
                        return
                    if not isinstance(request, _ToolRequest):
                        continue
                    try:
                        result = await session.call_tool(
                            request.tool_name,
                            arguments=request.arguments,
                            read_timeout_seconds=timedelta(
                                seconds=self.settings.timeout_seconds
                            ),
                        )
                        request.future.set_result(result)
                    except BaseException as exc:
                        request.future.set_exception(exc)

    @staticmethod
    def _format_result(result: Any) -> str:
        texts = [
            str(block.text)
            for block in (getattr(result, "content", None) or [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        text = "\n\n".join(texts).strip()
        RagMcpClient._raise_for_error(result, text=text)
        structured = getattr(result, "structuredContent", None) or {}
        citations = structured.get("citations") or []
        if citations:
            citation_lines = ["### 引用来源"]
            for citation in citations:
                source = citation.get("source") or "unknown"
                page = citation.get("page")
                chunk_id = citation.get("chunk_id")
                details = [str(source)]
                if page is not None:
                    details.append(f"page={page}")
                if chunk_id:
                    details.append(f"chunk={chunk_id}")
                citation_lines.append(f"- {' | '.join(details)}")
            text = "\n\n".join(part for part in (text, "\n".join(citation_lines)) if part)
        return text or "知识库未返回相关内容。"

    @staticmethod
    def _raise_for_error(result: Any, *, text: str | None = None) -> None:
        if not getattr(result, "isError", False):
            return
        if text is None:
            text = "\n\n".join(
                str(block.text)
                for block in (getattr(result, "content", None) or [])
                if getattr(block, "type", None) == "text"
                and getattr(block, "text", None)
            ).strip()
        raise RuntimeError(text or "MCP-RAG 返回错误")

    @staticmethod
    def _fail_pending(
        requests: queue.Queue[_ToolRequest | object],
        error: BaseException,
    ) -> None:
        while True:
            try:
                request = requests.get_nowait()
            except queue.Empty:
                return
            if isinstance(request, _ToolRequest) and not request.future.done():
                request.future.set_exception(error)

    def close(self, timeout: float = 10.0) -> None:
        with self._state_lock:
            thread = self._thread
            requests = self._requests
            self._thread = None
            self._requests = None
            self._ready = None
        if thread is None or requests is None or not thread.is_alive():
            return
        requests.put(_STOP)
        thread.join(timeout=max(0.0, timeout))


_client_lock = threading.Lock()
_client: RagMcpClient | None = None


def get_rag_mcp_client() -> RagMcpClient:
    from config import get_config

    global _client
    settings = resolve_rag_mcp_settings(get_config())
    with _client_lock:
        if _client is None or _client.settings != settings:
            if _client is not None:
                _client.close()
            _client = RagMcpClient(settings)
        return _client


async def close_rag_mcp_client() -> None:
    global _client
    with _client_lock:
        client = _client
        _client = None
    if client is not None:
        await asyncio.to_thread(client.close)
