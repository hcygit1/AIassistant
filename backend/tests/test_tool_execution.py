from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import PrivateAttr


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.security_context import get_runtime_security_context
from runtime.agent import AgentManager
from runtime.tool_execution import invoke_tool_async
from tool_results.storage import PERSISTED_OUTPUT_OPEN
from tools.persistence_wrapper import wrap_tools_for_persistence


class _AsyncUnavailableTool(BaseTool):
    name: str = "unsafe_fallback"
    description: str = "Test tool whose async path is unavailable."
    sync_called: bool = False

    def _run(self) -> str:
        self.sync_called = True
        return "sync result"

    async def _arun(self) -> str:
        raise NotImplementedError("async execution unavailable")


class _AsyncLargeOutputTool(BaseTool):
    name: str = "read"
    description: str = "Test tool that produces a large async result."

    def _run(self) -> str:
        raise AssertionError("sync execution must not be used")

    async def _arun(self) -> str:
        return "x" * 40050


class _SyncOnlyExecTool(BaseTool):
    name: str = "exec"
    description: str = "Approval-sensitive tool with only a sync implementation."
    sync_called: bool = False

    def _run(self, command: str) -> str:
        self.sync_called = True
        return f"ran {command}"


class _NoPersistSyncOnlyExecTool(_SyncOnlyExecTool):
    no_persist: bool = True


class _HandledErrorTool(BaseTool):
    name: str = "read"
    description: str = "Tool with public ToolException handling."
    handle_tool_error: bool = True

    def _run(self, path: str) -> str:
        raise ToolException(f"cannot read {path}")


class _ArtifactTool(BaseTool):
    name: str = "artifact"
    description: str = "Tool returning content and artifact."
    response_format: str = "content_and_artifact"

    def _run(self) -> tuple[str, dict[str, str]]:
        return "artifact content", {"source": "test"}


class _PublicInvokeTool(BaseTool):
    name: str = "configured"
    description: str = "Tool that observes public ainvoke config."
    _observed_config: object = PrivateAttr(default=None)

    def _run(self) -> str:
        raise AssertionError("protected sync path must not be used")

    async def ainvoke(self, tool_input, config=None, **kwargs):
        self._observed_config = config
        return "configured result"


class ToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_text_fallback_uses_async_entry_and_propagates_untrusted_source(
        self,
    ) -> None:
        manager = AgentManager()
        observed_contexts: dict[str, object] = {}

        class _FallbackTool:
            def __init__(self, name: str, result: str) -> None:
                self.name = name
                self.result = result
                self.sync_called = False

            async def ainvoke(self, tool_input: dict[str, object]) -> str:
                observed_contexts[self.name] = get_runtime_security_context()
                return self.result

            def _run(self, **_kwargs: object) -> str:
                self.sync_called = True
                raise AssertionError("sync execution must not be used")

        read_tool = _FallbackTool("read", "untrusted file content")
        exec_tool = _FallbackTool("exec", "tests passed")
        text_calls = (
            'functions.read:0{"path":"note.txt"}'
            'functions.exec:1{"command":"pytest"}'
        )

        class _FakeReactAgent:
            async def astream_events(self, payload, version="v2", config=None):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {"chunk": SimpleNamespace(content=text_calls)},
                }
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "model-run",
                    "data": {},
                }

        async def _fake_run_with_fallback_stream(
            _candidates,
            run_fn,
            _agent_id,
        ):
            async for item in run_fn("fake", "model"):
                yield item

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = lambda **_kwargs: _FakeReactAgent()
        tracker = SimpleNamespace(
            start_turn=Mock(return_value=SimpleNamespace(run_id="turn-1")),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
            log=Mock(),
        )

        with (
            patch.object(
                manager,
                "_get_or_build_tool_names",
                return_value=("read", "exec"),
            ),
            patch.object(
                manager,
                "_get_or_build_prompt",
                return_value=("system prompt", SimpleNamespace(summary=lambda: ""), 0),
            ),
            patch.object(
                manager,
                "_get_or_build_session_context",
                return_value=SimpleNamespace(
                    pruned_history=[],
                    summary_tokens=0,
                    history_tokens=0,
                ),
            ),
            patch.object(manager, "_build_tools", return_value=[read_tool, exec_tool]),
            patch.object(manager, "_incremental_ingest", new=AsyncMock()),
            patch.object(manager, "_maybe_auto_compact", new=AsyncMock()),
            patch("tools.skills_scanner.write_skills_snapshot"),
            patch("runtime.workspace.has_bootstrap", return_value=False),
            patch(
                "runtime.agent.resolve_fallback_candidates",
                return_value=[SimpleNamespace(provider="fake", model="model")],
            ),
            patch(
                "runtime.agent.run_with_fallback_stream",
                side_effect=_fake_run_with_fallback_stream,
            ),
            patch(
                "runtime.agent.resolve_agent_config",
                return_value={"recursion_limit": 10},
            ),
            patch("runtime.agent.create_llm", return_value=object()),
            patch("runtime.agent.run_tracker", tracker),
            patch("runtime.agent.audit_logger", audit),
            patch("runtime.agent.session_manager.save_message"),
            patch(
                "runtime.context_budget.resolve_budget",
                return_value=SimpleNamespace(active_tokens=10000),
            ),
            patch("runtime.agent.count_tokens", return_value=0),
            patch.dict(
                sys.modules,
                {
                    "langgraph": fake_langgraph,
                    "langgraph.prebuilt": fake_prebuilt,
                },
            ),
        ):
            events = [
                event
                async for event in manager.astream(
                    "检查文件后运行测试",
                    "s1",
                    agent_id="main",
                )
            ]

        self.assertTrue(any(event.get("type") == "done" for event in events))
        self.assertFalse(read_tool.sync_called)
        self.assertFalse(exec_tool.sync_called)
        self.assertFalse(observed_contexts["read"].recent_untrusted_content)
        self.assertTrue(observed_contexts["exec"].recent_untrusted_content)

    async def test_direct_execution_uses_public_async_entry_with_security_context(
        self,
    ) -> None:
        observed: dict[str, object] = {}

        class _Tool:
            async def ainvoke(self, tool_input: dict[str, object]) -> str:
                observed["input"] = tool_input
                observed["security"] = get_runtime_security_context()
                return "async result"

            def _run(self, **_kwargs: object) -> str:
                raise AssertionError("sync execution must not be used")

        result = await invoke_tool_async(
            _Tool(),
            {"command": "pytest"},
            user_message="检查文件后运行测试",
            recent_untrusted_content=True,
        )

        self.assertEqual(result, "async result")
        self.assertEqual(observed["input"], {"command": "pytest"})
        security = observed["security"]
        self.assertEqual(security.user_message, "检查文件后运行测试")
        self.assertTrue(security.recent_untrusted_content)

    async def test_persistence_wrapper_does_not_fallback_to_sync_execution(
        self,
    ) -> None:
        inner = _AsyncUnavailableTool()
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [inner],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            with self.assertRaisesRegex(
                NotImplementedError,
                "async execution unavailable",
            ):
                await wrapped.ainvoke({})

        self.assertFalse(inner.sync_called)

    async def test_persistence_wrapper_keeps_async_output_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [_AsyncLargeOutputTool()],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            result = await wrapped.ainvoke({})

            self.assertIn(PERSISTED_OUTPUT_OPEN, result)
            persisted_files = list(
                (Path(tmp) / "main" / "sessions" / "s1" / "tool-results").glob(
                    "*.txt"
                )
            )
            self.assertEqual(len(persisted_files), 1)
            self.assertEqual(
                persisted_files[0].read_text(encoding="utf-8"),
                "x" * 40050,
            )

    async def test_wrapper_rejects_approval_sensitive_sync_only_tool(self) -> None:
        inner = _SyncOnlyExecTool()
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [inner],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            with self.assertRaisesRegex(RuntimeError, "async approval"):
                await wrapped.ainvoke({"command": "pytest"})

        self.assertFalse(inner.sync_called)

    async def test_safety_wrapper_remains_when_persistence_is_disabled(self) -> None:
        inner = _SyncOnlyExecTool()
        wrapped = wrap_tools_for_persistence(
            [inner],
            data_dir="",
            agent_id="main",
            session_id="s1",
        )[0]

        self.assertIsNot(wrapped, inner)
        with self.assertRaisesRegex(RuntimeError, "async approval"):
            await wrapped.ainvoke({"command": "pytest"})
        self.assertFalse(inner.sync_called)

    async def test_no_persist_skips_storage_but_not_safety_wrapper(self) -> None:
        inner = _NoPersistSyncOnlyExecTool()
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [inner],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            self.assertIsNot(wrapped, inner)
            with self.assertRaisesRegex(RuntimeError, "async approval"):
                await wrapped.ainvoke({"command": "pytest"})

        self.assertFalse(inner.sync_called)

    async def test_wrapper_preserves_public_ainvoke_config(self) -> None:
        inner = _PublicInvokeTool()
        config = {"metadata": {"request_id": "req-1"}}
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [inner],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            result = await wrapped.ainvoke({}, config=config)

        self.assertEqual(result, "configured result")
        self.assertIs(inner._observed_config, config)

    async def test_wrapper_preserves_inner_tool_error_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [_HandledErrorTool()],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            result = await wrapped.ainvoke({"path": "missing.txt"})

        self.assertEqual(result, "cannot read missing.txt")

    async def test_wrapper_preserves_tool_message_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = wrap_tools_for_persistence(
                [_ArtifactTool()],
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )[0]

            result = await wrapped.ainvoke(
                {
                    "name": "artifact",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            )

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.content, "artifact content")
        self.assertEqual(result.artifact, {"source": "test"})


if __name__ == "__main__":
    unittest.main()
