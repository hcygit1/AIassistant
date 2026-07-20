"""Agent 引擎核心 — AgentManager, AgentState, 生命周期, 自动压缩, 命令处理"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import (
    DATA_DIR,
    get_heartbeat_config,
    resolve_agent_config,
    resolve_agent_workspace,
    resolve_agent_dir,
    list_agents,
)
from runtime.prompt_builder import prompt_builder
from sessions.session_manager import session_manager
from infra.run_tracker import run_tracker
from infra.audit_log import audit_logger
from infra.token_counter import (
    count_messages_tokens,
    count_tokens,
    detect_compaction_level,
)
from sessions.session_pruning import prune_messages
from runtime.command_parser import parse_command, execute_command
from llm.model_selection import (
    get_model_display_name,
    resolve_agent_model,
    resolve_fallback_candidates,
    run_with_fallback_stream,
)
from llm.llm_factory import create_llm, llm_cache
from llm.models_config import models_config
from runtime.agent_state import AgentState
from runtime.agent_compat import AgentManagerCompatibilityMixin
from runtime.agent_turn_compat import (
    AgentManagerTurnPreparationCompatibilityMixin,
)
from runtime.agent_runtime_assembly import AgentRuntimeAssembler
from runtime.agent_turn_preparation import AgentTurnPreparationAdapter
from runtime.agent_state_runtime import AgentStateRuntime
from runtime.agent_lifecycle import AgentLifecycle
from runtime.memory_runtime import MemoryRuntime
from runtime.model_runtime import ModelRuntime
from runtime.session_commands import SessionCommands
from runtime.session_compactor import SessionCompactor
from runtime.session_lifecycle import SessionLifecycle
from runtime.tool_registry import ToolRegistry
from subagents.subagent_runner import SubagentRunner
from subagents.subagent_service import SubagentService
from runtime.turn_recovery import TurnRecovery
from runtime.turn_executor import (
    TurnExecutor,
    should_persist_input_message as _should_persist_input_message,
)
from runtime.turn_context import (
    PromptCacheEntry,
    SessionContextCacheEntry,
    TurnContext,
)
from runtime.turn_preparation import TurnPreparation
from runtime.turn_service import (
    BARE_SESSION_RESET_PROMPT,
    TurnService,
    TurnServicePorts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 生命周期钩子
# ---------------------------------------------------------------------------

@dataclass
class LifecycleHooks:
    """显式生命周期钩子，用于审计、确认、记录等扩展"""

    async def on_before_tool_call(
        self, agent_id: str, run_id: str, tool_name: str, tool_input: dict[str, Any]
    ) -> None:
        """工具调用前（可在此拦截/确认）"""
        pass

    async def on_after_tool_call(
        self, agent_id: str, run_id: str, tool_name: str, tool_input: Any, tool_output: str
    ) -> None:
        """工具调用后（审计、记录）"""
        pass


from infra.event_bus import event_bus


# ---------------------------------------------------------------------------
# AgentManager — 核心引擎
# ---------------------------------------------------------------------------

class AgentManager(
    AgentManagerTurnPreparationCompatibilityMixin,
    AgentManagerCompatibilityMixin,
):

    @staticmethod
    def _log_compress(
        agent_id: str,
        session_id: str,
        archived_count: int,
        remaining_count: int,
    ) -> None:
        audit_logger.log_compress(
            agent_id,
            session_id,
            archived_count,
            remaining_count,
        )

    @staticmethod
    def _emit_runtime_event(
        agent_id: str,
        event: dict[str, Any],
    ) -> None:
        event_bus.emit(agent_id, event)

    @staticmethod
    def _audit_runtime_event(
        agent_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        audit_logger.log(agent_id, event_type, data)

    @staticmethod
    def _write_skills_snapshot(agent_id: str) -> None:
        from tools.skills_scanner import write_skills_snapshot

        write_skills_snapshot(agent_id)

    async def _ingest_completed_turn(
        self,
        agent_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        await self._incremental_ingest(
            agent_id,
            session_id,
            user_content,
            assistant_content,
        )

    async def _run_auto_compaction(
        self,
        session_id: str,
        agent_id: str,
        **kwargs: Any,
    ) -> None:
        await self._maybe_auto_compact(
            session_id,
            agent_id,
            **kwargs,
        )

    async def _compress_for_recovery(
        self,
        session_id: str,
        agent_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self.compress_session(
            session_id,
            agent_id,
            **kwargs,
        )

    @staticmethod
    def _has_bootstrap(agent_id: str) -> bool:
        from runtime.workspace import has_bootstrap

        return has_bootstrap(agent_id)

    @staticmethod
    def _get_locale() -> str:
        from config import get_config

        return get_config().get(
            "app",
            {},
        ).get("locale", "zh-CN")

    @staticmethod
    def _resolve_context_budget(agent_id: str) -> Any:
        from runtime.context_budget import resolve_budget

        return resolve_budget(agent_id)

    def __init__(self):
        self.lifecycle_hooks: LifecycleHooks | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        AgentRuntimeAssembler(self, globals()).build().install_on(self)

    def _get_state_persist_config(self, agent_id: str) -> tuple[bool, int]:
        """获取状态持久化配置 (enabled, interval_minutes)"""
        try:
            from config import resolve_agent_config
            cfg = resolve_agent_config(agent_id)
            persist_cfg = cfg.get("statePersist", {})
            return (
                persist_cfg.get("enabled", True),
                persist_cfg.get("autoSaveIntervalMinutes", 5),
            )
        except Exception:
            return True, 5

    def _get_state_path(self, agent_id: str) -> Path:
        """获取状态文件路径"""
        agent_dir = resolve_agent_dir(agent_id)
        return agent_dir / "agent_state.json"

    @staticmethod
    def _resolve_think_level(agent_id: str) -> Any:
        from llm.thinking import resolve_agent_think_default

        return resolve_agent_think_default(agent_id)

    def _init_mem_system(self, agent_id: str) -> None:
        self._memory_runtime.initialize_agent(agent_id)

    @staticmethod
    def _ensure_agent_workspace(agent_id: str) -> None:
        from runtime.workspace import ensure_agent_workspace

        ensure_agent_workspace(agent_id)

    async def initialize(self, data_dir: str) -> None:
        await self._lifecycle.initialize(data_dir)

    def get_llm(self, agent_id: str = "main"):
        """获取指定 Agent 的 LLM 实例（per-agent 动态创建，按 Provider 配置路由）"""
        return self._model_runtime.get_llm(agent_id)

    def get_current_model_ref(self, agent_id: str = "main"):
        """获取 Agent 当前使用的 ModelRef"""
        return self._model_runtime.resolve_current(agent_id)

    def switch_model(self, agent_id: str, model_raw: str) -> str:
        """运行时切换 Agent 模型，返回新模型描述"""
        return self._model_runtime.switch(
            agent_id,
            model_raw,
        )

    def get_model_override(
        self,
        agent_id: str,
    ):
        return self._model_runtime.get_override(agent_id)

    def restore_model_override(
        self,
        agent_id: str,
        override,
    ) -> None:
        self._model_runtime.restore_override(
            agent_id,
            override,
        )

    def clear_model_overrides(
        self,
        agent_id: str | None = None,
    ) -> None:
        self._model_runtime.clear(agent_id)

    def get_state(self, agent_id: str) -> AgentState:
        return self._state_runtime.get_state(agent_id)

    async def wait_for_pending_tasks(self, timeout: float = 30.0) -> None:
        """等待所有后台任务完成，用于应用关闭前确保数据不丢失"""
        await self._lifecycle.wait_for_pending_tasks(
            pending_tasks=self._pending_tasks,
            timeout=timeout,
            save_all_states=self._save_all_states,
        )

    async def close(self, timeout: float = 30.0) -> None:
        """停止后台任务、关闭持久化资源，并将管理器恢复为未初始化状态。"""
        try:
            await self._lifecycle.close(
                pending_tasks=self._pending_tasks,
                timeout=timeout,
                save_all_states=self._save_all_states,
            )
        finally:
            self.lifecycle_hooks = None

    async def _save_all_states(self) -> None:
        """保存所有 Agent 状态到磁盘"""
        await self._state_runtime.save_all_states()

    # ------------------------------------------------------------------
    # 核心流式方法
    # ------------------------------------------------------------------

    async def astream(
        self,
        message: str,
        session_id: str,
        agent_id: str = "main",
        prompt_mode: str = "full",
        persist_input_role: str = "user",
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._turn_service.stream(
            message,
            session_id,
            agent_id=agent_id,
            prompt_mode=prompt_mode,
            persist_input_role=persist_input_role,
        ):
            yield event

    # ------------------------------------------------------------------
    # 每轮增量入库 (每轮结束后异步触发, hash 去重保证幂等)
    # ------------------------------------------------------------------

    async def _incremental_ingest(
        self,
        agent_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        await self._memory_runtime.ingest_turn(
            agent_id,
            session_id,
            user_content,
            assistant_content,
        )

    # ------------------------------------------------------------------
    # Mem Worker 批量入库 (压缩 / session 结束时触发)
    # ------------------------------------------------------------------

    async def _batch_ingest_messages(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        session_end: bool = False,
    ) -> None:
        await self._memory_runtime.ingest_messages(
            agent_id,
            session_id,
            messages,
            session_end=session_end,
        )

    # ------------------------------------------------------------------
    # 自动压缩
    # ------------------------------------------------------------------

    async def _maybe_auto_compact(self, session_id: str, agent_id: str, overhead_tokens: int = 0) -> None:
        await self._session_lifecycle.maybe_auto_compact(
            session_id,
            agent_id,
            overhead_tokens=overhead_tokens,
            compress_session=self.compress_session,
        )

    # ------------------------------------------------------------------
    # 会话重置命令处理：/new 与 /reset
    # ------------------------------------------------------------------

    async def _handle_reset(
        self, session_id: str, agent_id: str, model_override: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._session_commands.handle_reset(
            session_id,
            agent_id,
            model_override=model_override,
            get_store=lambda target_agent_id: self.mem_stores.get(
                target_agent_id
            ),
            get_state=self.get_state,
            batch_ingest_messages=self._batch_ingest_messages,
            switch_model=self.switch_model,
        ):
            yield event

    async def _handle_reset_noflush(
        self, session_id: str, agent_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in (
            self._session_commands.handle_reset_noflush(
                session_id,
                agent_id,
                get_store=lambda target_agent_id: self.mem_stores.get(
                    target_agent_id
                ),
                get_state=self.get_state,
            )
        ):
            yield event

    # ------------------------------------------------------------------
    # /compact 命令处理
    # ------------------------------------------------------------------

    async def _handle_compact(
        self, session_id: str, agent_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._session_commands.handle_compact(
            session_id,
            agent_id,
            compress_session=self.compress_session,
            calc_compress_count_by_turns=(
                self._calc_compress_count_by_turns
            ),
        ):
            yield event

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    async def _generate_structured_summary(
        self,
        agent_id: str,
        session_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        return await self._session_lifecycle.generate_structured_summary(
            agent_id,
            session_id,
            to_compress,
            text_to_summarize,
            plain_fallback=self._summarize_plain_fallback,
        )

    async def _summarize_plain_fallback(
        self,
        agent_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        return await self._session_lifecycle.summarize_plain_fallback(
            agent_id,
            to_compress,
            text_to_summarize,
        )

    async def compress_session(
        self, session_id: str, agent_id: str, level: str = "sliding",
    ) -> dict[str, Any]:
        return await self._session_lifecycle.compress_session(
            session_id,
            agent_id,
            level=level,
            generate_summary=self._generate_structured_summary,
            batch_ingest_messages=self._batch_ingest_messages,
            pending_tasks=self._pending_tasks,
        )

    @staticmethod
    def _calc_compress_count_by_turns(messages: list[dict[str, Any]], keep_turns: int) -> int:
        return SessionLifecycle.calc_compress_count_by_turns(
            messages,
            keep_turns,
        )

    # ------------------------------------------------------------------
    # Agent 注册
    # ------------------------------------------------------------------

    async def register_agent(self, agent_id: str) -> None:
        await self._lifecycle.register_agent(agent_id)


agent_manager = AgentManager()
