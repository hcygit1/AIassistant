"""Agent runtime startup, shutdown, and background-task coordination."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, MutableMapping
from typing import Any, Awaitable

logger = logging.getLogger("runtime.agent")


class AgentLifecycle:
    """Own the lifecycle boundary for Agent runtime resources."""

    def __init__(
        self,
        *,
        state_runtime: Any,
        memory_runtime: Any,
        model_runtime: Any,
        list_agents: Callable[[], list[dict[str, Any]]],
        ensure_workspace: Callable[[str], None],
        prompt_cache: MutableMapping[Any, Any],
        session_context_cache: MutableMapping[Any, Any],
        tool_name_cache: MutableMapping[Any, Any],
    ) -> None:
        self._state_runtime = state_runtime
        self._memory_runtime = memory_runtime
        self._model_runtime = model_runtime
        self._list_agents = list_agents
        self._ensure_workspace = ensure_workspace
        self._prompt_cache = prompt_cache
        self._session_context_cache = session_context_cache
        self._tool_name_cache = tool_name_cache
        self.data_dir = ""
        self.initialized = False

    async def initialize(self, data_dir: str) -> None:
        self.data_dir = data_dir

        for agent in self._list_agents():
            self._initialize_agent_runtime(agent["id"])

        self.initialized = True

    async def register_agent(self, agent_id: str) -> None:
        self._initialize_agent_runtime(agent_id)

    def _initialize_agent_runtime(self, agent_id: str) -> None:
        self._ensure_workspace(agent_id)
        self._memory_runtime.initialize_agent(agent_id)
        self._state_runtime.initialize_agent(agent_id)

    async def wait_for_pending_tasks(
        self,
        *,
        pending_tasks: set[asyncio.Task],
        timeout: float,
        save_all_states: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        try:
            await (
                save_all_states()
                if save_all_states is not None
                else self._state_runtime.save_all_states()
            )
        except Exception as error:
            logger.error("关闭前保存 Agent 状态失败: %s", error)

        await self._state_runtime.stop_periodic_saves()

        if not pending_tasks:
            return
        logger.info("等待 %s 个后台任务完成...", len(pending_tasks))
        pending = list(pending_tasks)
        pending_tasks.clear()
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
            logger.info("所有后台任务已完成")
        except asyncio.TimeoutError:
            logger.warning(
                "等待后台任务超时（%s秒），部分任务可能未完成",
                timeout,
            )
        except Exception as error:
            logger.error("等待后台任务时出错: %s", error)

    async def close(
        self,
        *,
        pending_tasks: set[asyncio.Task],
        timeout: float,
        save_all_states: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.initialized = False
        try:
            await self.wait_for_pending_tasks(
                pending_tasks=pending_tasks,
                timeout=timeout,
                save_all_states=save_all_states,
            )
        finally:
            self._memory_runtime.close()
            self._model_runtime.clear()
            self._state_runtime.clear()
            pending_tasks.clear()
            self._prompt_cache.clear()
            self._session_context_cache.clear()
            self._tool_name_cache.clear()
            self.data_dir = ""
