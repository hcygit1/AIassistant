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
from infra.token_counter import count_messages_tokens, count_tokens
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
from runtime.memory_runtime import MemoryRuntime
from runtime.model_runtime import ModelRuntime
from runtime.session_commands import SessionCommands
from runtime.session_compactor import SessionCompactor
from runtime.tool_registry import ToolRegistry
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


from infra.event_bus import EventBus, Events, event_bus


# ---------------------------------------------------------------------------
# AgentManager — 核心引擎
# ---------------------------------------------------------------------------

class AgentManager:
    @property
    def _prompt_cache(self) -> dict[tuple[Any, ...], PromptCacheEntry]:
        return self._turn_context.prompt_cache

    @_prompt_cache.setter
    def _prompt_cache(
        self,
        value: dict[tuple[Any, ...], PromptCacheEntry],
    ) -> None:
        self._turn_context.prompt_cache = value

    @property
    def _session_context_cache(
        self,
    ) -> dict[tuple[str, str], SessionContextCacheEntry]:
        return self._turn_context.session_context_cache

    @_session_context_cache.setter
    def _session_context_cache(
        self,
        value: dict[tuple[str, str], SessionContextCacheEntry],
    ) -> None:
        self._turn_context.session_context_cache = value

    @property
    def mem_stores(self) -> dict[str, Any]:
        return self._memory_runtime.stores

    @mem_stores.setter
    def mem_stores(self, value: dict[str, Any]) -> None:
        self._memory_runtime.stores = value

    @property
    def mem_embedders(self) -> dict[str, Any]:
        return self._memory_runtime.embedders

    @mem_embedders.setter
    def mem_embedders(self, value: dict[str, Any]) -> None:
        self._memory_runtime.embedders = value

    @property
    def mem_workers(self) -> dict[str, Any]:
        return self._memory_runtime.workers

    @mem_workers.setter
    def mem_workers(self, value: dict[str, Any]) -> None:
        self._memory_runtime.workers = value

    @property
    def mem_recalls(self) -> dict[str, Any]:
        return self._memory_runtime.recalls

    @mem_recalls.setter
    def mem_recalls(self, value: dict[str, Any]) -> None:
        self._memory_runtime.recalls = value

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
        self.data_dir: str = ""
        self._memory_runtime = MemoryRuntime()
        self._model_runtime = ModelRuntime(
            resolve_configured_model=lambda agent_id: (
                resolve_agent_model(agent_id)
            ),
            resolve_configured_candidates=lambda agent_id: (
                resolve_fallback_candidates(agent_id)
            ),
            find_model=lambda model_id: (
                models_config.find_model_by_id(model_id)
            ),
            get_model=lambda ref: models_config.get_model(ref),
            invalidate_llm=lambda agent_id: (
                llm_cache.invalidate(agent_id)
            ),
            get_or_create_llm=lambda agent_id, ref: (
                llm_cache.get_or_create(agent_id, ref)
            ),
            get_display_name=lambda ref: (
                get_model_display_name(ref)
            ),
        )
        self._tool_registry = ToolRegistry(self)
        self._turn_context = TurnContext()
        self._session_compactor = SessionCompactor(
            resolve_agent_config=lambda agent_id: resolve_agent_config(
                agent_id
            ),
            load_session=lambda session_id, agent_id: (
                session_manager.load_session(session_id, agent_id)
            ),
            compress_history=lambda session_id, agent_id, count: (
                session_manager.compress_history(
                    session_id,
                    agent_id,
                    count,
                )
            ),
            get_llm=lambda agent_id: self.get_llm(agent_id),
            log_compress=self._log_compress,
        )
        self._session_commands = SessionCommands(
            load_session=lambda session_id, agent_id: (
                session_manager.load_session(session_id, agent_id)
            ),
            reset_session=lambda session_id, agent_id: (
                session_manager.reset_session(session_id, agent_id)
            ),
            resolve_agent_config=lambda agent_id: (
                resolve_agent_config(agent_id)
            ),
            emit_event=self._emit_runtime_event,
            audit_log=self._audit_runtime_event,
        )
        self._turn_recovery = TurnRecovery(
            reset_session=lambda session_id, agent_id: (
                session_manager.reset_session(
                    session_id,
                    agent_id,
                )
            ),
            compress_session=self._compress_for_recovery,
            audit_log=self._audit_runtime_event,
            sleep=lambda seconds: asyncio.sleep(seconds),
        )
        self._turn_executor = TurnExecutor(
            create_llm=lambda ref: create_llm(ref),
            build_messages=lambda history, message: (
                self._build_messages(history, message)
            ),
            get_lifecycle_hooks=lambda: self.lifecycle_hooks,
            get_run_tracker=lambda: run_tracker,
            get_audit_logger=lambda: audit_logger,
            save_message=lambda *args, **kwargs: (
                session_manager.save_message(*args, **kwargs)
            ),
            write_skills_snapshot=self._write_skills_snapshot,
            emit_event=self._emit_runtime_event,
            count_tokens=lambda text: count_tokens(text),
            incremental_ingest=self._ingest_completed_turn,
            get_pending_tasks=lambda: self._pending_tasks,
            maybe_auto_compact=self._run_auto_compaction,
        )
        self._turn_service = TurnService(
            TurnServicePorts(
                get_state=lambda agent_id: (
                    self.get_state(agent_id)
                ),
                parse_command=lambda message: (
                    parse_command(message)
                ),
                execute_command=lambda *args, **kwargs: (
                    execute_command(*args, **kwargs)
                ),
                switch_model=lambda agent_id, model: (
                    self.switch_model(agent_id, model)
                ),
                get_current_model=lambda agent_id: (
                    self.get_current_model_ref(agent_id)
                ),
                get_model_override=lambda agent_id: (
                    self.get_model_override(agent_id)
                ),
                handle_reset=lambda *args, **kwargs: (
                    self._handle_reset(*args, **kwargs)
                ),
                handle_reset_noflush=lambda *args, **kwargs: (
                    self._handle_reset_noflush(
                        *args,
                        **kwargs,
                    )
                ),
                handle_compact=lambda *args, **kwargs: (
                    self._handle_compact(*args, **kwargs)
                ),
                write_skills_snapshot=lambda agent_id: (
                    self._write_skills_snapshot(agent_id)
                ),
                has_bootstrap=lambda agent_id: (
                    self._has_bootstrap(agent_id)
                ),
                resolve_workspace=lambda agent_id: (
                    resolve_agent_workspace(agent_id)
                ),
                get_locale=lambda: self._get_locale(),
                get_tool_names=lambda agent_id: (
                    self._get_or_build_tool_names(agent_id)
                ),
                build_prompt=lambda **kwargs: (
                    self._get_or_build_prompt(**kwargs)
                ),
                get_session_context=lambda **kwargs: (
                    self._get_or_build_session_context(
                        **kwargs
                    )
                ),
                build_tools=lambda agent_id, session_id: (
                    self._build_tools(agent_id, session_id)
                ),
                resolve_budget=lambda agent_id: (
                    self._resolve_context_budget(agent_id)
                ),
                resolve_agent_config=lambda agent_id: (
                    resolve_agent_config(agent_id)
                ),
                resolve_candidates=lambda agent_id: (
                    self._model_runtime.resolve_candidates(
                        agent_id
                    )
                ),
                execute_turn=lambda request: (
                    self._turn_executor.execute(request)
                ),
                run_fallback_stream=(
                    lambda candidates, run_model, agent_id: (
                        run_with_fallback_stream(
                            candidates,
                            run_model,
                            agent_id,
                        )
                    )
                ),
                recover_turn=lambda **kwargs: (
                    self._turn_recovery.run(**kwargs)
                ),
            )
        )
        self._states: dict[str, AgentState] = {}
        self._initialized = False
        self.lifecycle_hooks: LifecycleHooks | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._state_save_tasks: dict[str, asyncio.Task] = {}
        self._tool_name_cache = self._tool_registry.name_cache

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

    async def _periodic_state_save(self, agent_id: str) -> None:
        """定期保存 Agent 状态"""
        enabled, interval = self._get_state_persist_config(agent_id)
        if not enabled:
            return

        interval_seconds = max(60, interval * 60)  # 至少1分钟
        while self._initialized:
            try:
                await asyncio.sleep(interval_seconds)
                if agent_id in self._states:
                    state = self._states[agent_id]
                    state_path = self._get_state_path(agent_id)
                    state.save_to_disk(state_path)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Periodic state save error for {agent_id}: {e}")

    def _init_mem_system(self, agent_id: str) -> None:
        self._memory_runtime.initialize_agent(agent_id)

    async def initialize(self, data_dir: str) -> None:
        self.data_dir = data_dir

        from runtime.workspace import ensure_agent_workspace

        for agent in list_agents():
            agent_id = agent["id"]
            ensure_agent_workspace(agent_id)
            self._init_mem_system(agent_id)

            # 从磁盘加载状态或创建新状态
            enabled, _ = self._get_state_persist_config(agent_id)
            if enabled:
                state_path = self._get_state_path(agent_id)
                self._states[agent_id] = AgentState.load_from_disk(state_path, agent_id)
                # 启动定期保存任务
                save_task = asyncio.create_task(self._periodic_state_save(agent_id))
                self._state_save_tasks[agent_id] = save_task
            else:
                from llm.thinking import resolve_agent_think_default
                think_level = resolve_agent_think_default(agent_id)
                self._states[agent_id] = AgentState(agent_id=agent_id, think_level=think_level.value)

        self._initialized = True

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
        if agent_id not in self._states:
            self._states[agent_id] = AgentState(agent_id=agent_id)
        return self._states[agent_id]

    async def wait_for_pending_tasks(self, timeout: float = 30.0) -> None:
        """等待所有后台任务完成，用于应用关闭前确保数据不丢失"""
        # 先保存所有 Agent 状态
        try:
            await self._save_all_states()
        except Exception as e:
            logger.error("关闭前保存 Agent 状态失败: %s", e)

        # 取消状态保存任务
        state_save_tasks = list(self._state_save_tasks.values())
        for task in state_save_tasks:
            task.cancel()
        self._state_save_tasks.clear()
        if state_save_tasks:
            await asyncio.gather(*state_save_tasks, return_exceptions=True)

        if not self._pending_tasks:
            return
        logger.info(f"等待 {len(self._pending_tasks)} 个后台任务完成...")
        # 创建所有任务的副本
        pending = list(self._pending_tasks)
        self._pending_tasks.clear()
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
            logger.info("所有后台任务已完成")
        except asyncio.TimeoutError:
            logger.warning(f"等待后台任务超时（{timeout}秒），部分任务可能未完成")
        except Exception as e:
            logger.error(f"等待后台任务时出错: {e}")

    async def close(self, timeout: float = 30.0) -> None:
        """停止后台任务、关闭持久化资源，并将管理器恢复为未初始化状态。"""
        self._initialized = False
        try:
            await self.wait_for_pending_tasks(timeout=timeout)
        finally:
            self._memory_runtime.close()
            self._model_runtime.clear()
            self._states.clear()
            self._state_save_tasks.clear()
            self._pending_tasks.clear()
            self._prompt_cache.clear()
            self._session_context_cache.clear()
            self._tool_name_cache.clear()
            self.lifecycle_hooks = None
            self.data_dir = ""

    async def _save_all_states(self) -> None:
        """保存所有 Agent 状态到磁盘"""
        for agent_id, state in self._states.items():
            try:
                enabled, _ = self._get_state_persist_config(agent_id)
                if enabled:
                    state_path = self._get_state_path(agent_id)
                    state.save_to_disk(state_path)
            except Exception as e:
                logger.warning(f"Failed to save state for {agent_id}: {e}")

    def _collect_tools(self, agent_id: str, session_id: str = "") -> list:
        return self._tool_registry.collect_tools(agent_id, session_id)

    def _wrap_tools_for_session(self, agent_id: str, session_id: str, tools: list) -> list:
        return self._tool_registry.wrap_tools(
            self.data_dir,
            agent_id,
            session_id,
            tools,
        )

    def _build_tools(self, agent_id: str, session_id: str = "") -> list:
        return self._tool_registry.build_tools(
            self.data_dir,
            agent_id,
            session_id,
            collect_tools=self._collect_tools,
            filter_tools=self._filter_tools_by_policy,
            wrap_tools=self._wrap_tools_for_session,
        )

    def _resolve_tool_policy(self, agent_id: str) -> tuple[list[str], list[str]]:
        return self._tool_registry.resolve_policy(agent_id)

    def _filter_tools_by_policy(self, agent_id: str, tools: list) -> list:
        return self._tool_registry.filter_tools(
            agent_id,
            tools,
            resolve_policy=self._resolve_tool_policy,
        )

    def _build_messages(
        self, history: list[dict[str, Any]], new_message: str
    ) -> list:
        return self._turn_context.build_messages(
            history,
            new_message,
            human_message=HumanMessage,
            ai_message=AIMessage,
            system_message=SystemMessage,
        )

    @staticmethod
    def _safe_mtime(path: Path) -> float | None:
        return TurnContext.safe_mtime(path)

    def _project_context_signature(self, agent_id: str, prompt_mode: str) -> tuple[Any, ...]:
        return self._turn_context.project_context_signature(
            agent_id,
            prompt_mode,
            resolve_workspace=resolve_agent_workspace,
            resolve_agent_dir=resolve_agent_dir,
            safe_mtime=self._safe_mtime,
        )

    def _prompt_runtime_signature(
        self,
        agent_id: str,
    ) -> tuple[Any, ...]:
        base_signature = (
            self._turn_context.prompt_runtime_signature(
                agent_id,
                resolve_agent_config_fn=resolve_agent_config,
                get_heartbeat_config_fn=get_heartbeat_config,
            )
        )
        return (
            *base_signature,
            str(self.get_current_model_ref(agent_id)),
        )

    def _pruning_signature(self, agent_id: str) -> str:
        return self._turn_context.pruning_signature(
            agent_id,
            resolve_agent_config_fn=resolve_agent_config,
        )

    def _tool_policy_signature(self, agent_id: str) -> tuple[Any, ...]:
        return self._tool_registry.policy_signature(
            agent_id,
            resolve_policy=self._resolve_tool_policy,
        )

    def _get_or_build_tool_names(self, agent_id: str) -> tuple[str, ...]:
        return self._tool_registry.get_or_build_tool_names(
            agent_id,
            collect_tools=self._collect_tools,
            filter_tools=self._filter_tools_by_policy,
            policy_signature=self._tool_policy_signature,
        )

    def _get_or_build_prompt(
        self,
        *,
        agent_id: str,
        prompt_mode: str,
        available_tool_names: list[str] | None,
        extra_system_prompt: str | None,
        locale: str,
    ) -> tuple[str, Any, int]:
        return self._turn_context.get_or_build_prompt(
            agent_id=agent_id,
            prompt_mode=prompt_mode,
            available_tool_names=available_tool_names,
            extra_system_prompt=extra_system_prompt or None,
            locale=locale,
            static_signature=self._project_context_signature(
                agent_id,
                prompt_mode,
            ),
            runtime_signature=self._prompt_runtime_signature(agent_id),
            build_prompt=prompt_builder.build_system_prompt_with_report,
            count_tokens=count_tokens,
        )

    @staticmethod
    def _session_summary_fingerprint(summary: Any) -> str | None:
        return TurnContext.session_summary_fingerprint(summary)

    def _get_or_build_session_context(
        self,
        *,
        agent_id: str,
        session_id: str,
    ) -> SessionContextCacheEntry:
        session_path = resolve_agent_dir(agent_id) / "sessions" / f"{session_id}.json"
        store = self.mem_stores.get(agent_id)
        return self._turn_context.get_or_build_session_context(
            agent_id=agent_id,
            session_id=session_id,
            session_path=session_path,
            store=store,
            safe_mtime=self._safe_mtime,
            summary_fingerprint=self._session_summary_fingerprint,
            load_history=session_manager.load_session_for_agent,
            format_summary=prompt_builder.format_session_summary,
            prune_history=prune_messages,
            count_tokens=count_tokens,
            count_messages_tokens=count_messages_tokens,
            pruning_signature=self._pruning_signature(agent_id),
        )

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
        agent_cfg = resolve_agent_config(agent_id)
        compaction_cfg = agent_cfg.get("compaction", {})
        if not compaction_cfg.get("enabled", True):
            return

        data = session_manager.load_session(session_id, agent_id)
        if not data:
            return

        messages = data.get("messages", [])

        from infra.token_counter import detect_compaction_level
        level = detect_compaction_level(messages, agent_id=agent_id, overhead_tokens=overhead_tokens)

        if level == "none":
            return

        logger.info("Auto-compaction triggered: level=%s agent=%s session=%s", level, agent_id, session_id)
        audit_logger.log(agent_id, "auto_compact_trigger", {"session_id": session_id, "level": level})
        event_bus.emit(agent_id, Events.auto_compact_start(session_id=session_id, level=level))
        try:
            await self.compress_session(session_id, agent_id, level=level)
            event_bus.emit(agent_id, Events.auto_compact_done(session_id=session_id))
        except Exception as e:
            logger.error(f"Auto-compaction failed: {e}")
            audit_logger.log(agent_id, "auto_compact_error", {"error": str(e)})

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
        return await self._session_compactor.generate_structured_summary(
            agent_id,
            session_id,
            to_compress,
            text_to_summarize,
            store=self.mem_stores.get(agent_id),
            plain_fallback=self._summarize_plain_fallback,
        )

    async def _summarize_plain_fallback(
        self,
        agent_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        return await self._session_compactor.summarize_plain_fallback(
            agent_id,
            to_compress,
            text_to_summarize,
        )

    async def compress_session(
        self, session_id: str, agent_id: str, level: str = "sliding",
    ) -> dict[str, Any]:
        return await self._session_compactor.compress_session(
            session_id,
            agent_id,
            level=level,
            get_store=lambda target_agent_id: self.mem_stores.get(
                target_agent_id
            ),
            get_state=self.get_state,
            generate_summary=self._generate_structured_summary,
            batch_ingest_messages=self._batch_ingest_messages,
            pending_tasks=self._pending_tasks,
        )

    @staticmethod
    def _calc_compress_count_by_turns(messages: list[dict[str, Any]], keep_turns: int) -> int:
        return SessionCompactor.calc_compress_count_by_turns(
            messages,
            keep_turns,
        )

    # ------------------------------------------------------------------
    # Agent 注册
    # ------------------------------------------------------------------

    async def register_agent(self, agent_id: str) -> None:
        from runtime.workspace import ensure_agent_workspace

        ensure_agent_workspace(agent_id)
        self._init_mem_system(agent_id)
        self._states[agent_id] = AgentState(agent_id=agent_id)


agent_manager = AgentManager()
