"""Thin wrapper that applies tool-result persistence to any LangChain BaseTool.

Usage (inside AgentManager._build_tools):

    from tools.persistence_wrapper import wrap_tools_for_persistence
    tools = wrap_tools_for_persistence(tools, data_dir, agent_id, session_id)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

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
        )
        self._inner = inner
        self._data_dir = data_dir
        self._agent_id = agent_id
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist(self, raw: Any) -> str:
        text = raw if isinstance(raw, str) else str(raw)
        return maybe_persist_tool_output(
            text,
            tool_name=self._inner.name,
            data_dir=self._data_dir,
            agent_id=self._agent_id,
            session_id=self._session_id,
        )

    # ------------------------------------------------------------------
    # Sync / async delegation
    # ------------------------------------------------------------------

    def _run(self, *args: Any, **kwargs: Any) -> str:
        result = self._inner._run(*args, **kwargs)
        return self._persist(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        try:
            result = await self._inner._arun(*args, **kwargs)
            return self._persist(result)
        except NotImplementedError:
            result = self._inner._run(*args, **kwargs)
            return self._persist(result)


def wrap_tools_for_persistence(
    tools: list[BaseTool],
    data_dir: str,
    agent_id: str,
    session_id: str,
) -> list[BaseTool]:
    """Return a new list where every tool is wrapped with persistence logic.

    If *data_dir* is empty (e.g. AgentManager not yet initialised) the list
    is returned unchanged so startup is not broken.

    Tools that define ``no_persist = True`` as a class attribute are left
    unwrapped (useful for structural / tiny-output tools if needed).
    """
    if not data_dir:
        return tools

    wrapped: list[BaseTool] = []
    for tool in tools:
        if getattr(tool, "no_persist", False):
            wrapped.append(tool)
        else:
            wrapped.append(
                _PersistenceWrapper(
                    inner=tool,
                    data_dir=data_dir,
                    agent_id=agent_id,
                    session_id=session_id,
                )
            )
    return wrapped
