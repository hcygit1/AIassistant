"""Thin wrapper that applies tool-result persistence to any LangChain BaseTool.

Usage (inside AgentManager._build_tools):

    from tools.persistence_wrapper import wrap_tools_for_persistence
    tools = wrap_tools_for_persistence(tools, data_dir, agent_id, session_id)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from runtime.tool_execution import (
    ensure_async_execution_safe,
    ensure_sync_execution_allowed,
)
from tool_results.pipeline import maybe_persist_tool_output

logger = logging.getLogger(__name__)


class _PersistenceWrapper(BaseTool):
    """Delegates all calls to the inner tool and post-processes the string output."""

    _inner: BaseTool = PrivateAttr()
    _data_dir: str = PrivateAttr()
    _agent_id: str = PrivateAttr()
    _session_id: str = PrivateAttr()

    def __init__(
        self,
        inner: BaseTool,
        data_dir: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        super().__init__(
            name=inner.name,
            description=inner.description,
            args_schema=inner.args_schema,
            return_direct=inner.return_direct,
            tags=inner.tags,
            metadata=inner.metadata,
        )
        self._inner = inner
        self._data_dir = data_dir
        self._agent_id = agent_id
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def wrapped_tool(self) -> BaseTool:
        return self._inner

    def _persist(self, raw: Any) -> str:
        text = raw if isinstance(raw, str) else str(raw)
        if not self._data_dir or getattr(self._inner, "no_persist", False):
            return text
        return maybe_persist_tool_output(
            text,
            tool_name=self._inner.name,
            data_dir=self._data_dir,
            agent_id=self._agent_id,
            session_id=self._session_id,
        )

    def _persist_result(self, raw: Any) -> Any:
        if isinstance(raw, str):
            return self._persist(raw)
        if isinstance(raw, ToolMessage) and isinstance(raw.content, str):
            persisted_content = self._persist(raw.content)
            if persisted_content == raw.content:
                return raw
            return raw.model_copy(update={"content": persisted_content})
        return raw

    @staticmethod
    def _rebuild_tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if args and kwargs:
            raise TypeError("Tool invocation cannot mix positional and keyword inputs")
        if len(args) > 1:
            raise TypeError("Tool invocation accepts at most one positional input")
        if args:
            return args[0]
        return kwargs

    # ------------------------------------------------------------------
    # Public delegation keeps validation, callbacks, config and artifacts.
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        ensure_sync_execution_allowed(self)
        result = self._inner.invoke(input, config=config, **kwargs)
        return self._persist_result(result)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        ensure_async_execution_safe(self)
        result = await self._inner.ainvoke(input, config=config, **kwargs)
        return self._persist_result(result)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        ensure_sync_execution_allowed(self)
        tool_input = self._rebuild_tool_input(args, kwargs)
        return self._persist_result(self._inner.invoke(tool_input))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        ensure_async_execution_safe(self)
        tool_input = self._rebuild_tool_input(args, kwargs)
        return self._persist_result(await self._inner.ainvoke(tool_input))


def wrap_tools_for_persistence(
    tools: list[BaseTool],
    data_dir: str,
    agent_id: str,
    session_id: str,
) -> list[BaseTool]:
    """Return a new list where every tool is wrapped with persistence logic.

    If *data_dir* is empty, persistence is skipped but the wrapper remains so
    approval-sensitive tools still use the guarded public async path.

    Tools that define ``no_persist = True`` skip storage only; the safety
    wrapper remains active.
    """
    wrapped: list[BaseTool] = []
    for tool in tools:
        wrapped.append(
            _PersistenceWrapper(
                inner=tool,
                data_dir=data_dir,
                agent_id=agent_id,
                session_id=session_id,
            )
        )
    return wrapped
