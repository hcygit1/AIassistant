"""Agent 状态的加载、持久化和定时保存。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from runtime.agent_state import AgentState


logger = logging.getLogger("runtime.agent")


class AgentStateRuntime:
    """管理 AgentState 生命周期，不负责 Agent 其他运行时资源。"""

    def __init__(
        self,
        *,
        resolve_persist_config: Callable[[str], tuple[bool, int]],
        resolve_state_path: Callable[[str], Path],
        resolve_think_level: Callable[[str], Any],
        is_initialized: Callable[[], bool],
    ) -> None:
        self._resolve_persist_config = resolve_persist_config
        self._resolve_state_path = resolve_state_path
        self._resolve_think_level = resolve_think_level
        self._is_initialized = is_initialized
        self.states: dict[str, AgentState] = {}
        self.save_tasks: dict[str, asyncio.Task] = {}

    def initialize_agent(self, agent_id: str) -> None:
        enabled, _ = self._resolve_persist_config(agent_id)
        if enabled:
            state_path = self._resolve_state_path(agent_id)
            self.states[agent_id] = AgentState.load_from_disk(
                state_path,
                agent_id,
            )
            self.save_tasks[agent_id] = asyncio.create_task(
                self._periodic_state_save(agent_id)
            )
            return

        think_level = self._resolve_think_level(agent_id)
        self.states[agent_id] = AgentState(
            agent_id=agent_id,
            think_level=getattr(think_level, "value", think_level),
        )

    def get_state(self, agent_id: str) -> AgentState:
        if agent_id not in self.states:
            self.states[agent_id] = AgentState(agent_id=agent_id)
        return self.states[agent_id]

    async def _periodic_state_save(self, agent_id: str) -> None:
        enabled, interval = self._resolve_persist_config(agent_id)
        if not enabled:
            return

        interval_seconds = max(60, interval * 60)
        while self._is_initialized():
            try:
                await asyncio.sleep(interval_seconds)
                state = self.states.get(agent_id)
                if state is not None:
                    state.save_to_disk(self._resolve_state_path(agent_id))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "Periodic state save error for %s: %s",
                    agent_id,
                    e,
                )

    async def save_all_states(self) -> None:
        for agent_id, state in self.states.items():
            try:
                enabled, _ = self._resolve_persist_config(agent_id)
                if enabled:
                    state.save_to_disk(self._resolve_state_path(agent_id))
            except Exception as e:
                logger.warning(
                    "Failed to save state for %s: %s",
                    agent_id,
                    e,
                )

    async def stop_periodic_saves(self) -> None:
        tasks = list(self.save_tasks.values())
        for task in tasks:
            task.cancel()
        self.save_tasks.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self) -> None:
        self.states.clear()
        self.save_tasks.clear()
