"""Agent 引擎核心 — AgentManager, AgentState, 生命周期, 自动压缩, 命令处理"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    DATA_DIR,
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
from runtime.tool_call_parser import parse_text_tool_calls, strip_tool_call_patterns
from infra.errors import (
    is_compaction_failure_error,
    is_likely_context_overflow_error,
    is_role_ordering_error,
    is_session_corruption_error,
    is_transient_http_error,
)
from llm.model_selection import (
    resolve_fallback_candidates,
    run_with_fallback_stream,
)
from llm.models_config import ModelRef
from llm.llm_factory import create_llm
from runtime.source_sink_guard import (
    contains_untrusted_marker,
    is_untrusted_source_tool,
)
from runtime.security_context import mark_recent_untrusted_content, runtime_security_context
from runtime.tool_execution import invoke_tool_async
from runtime.agent_state import AgentState
from runtime.memory_runtime import MemoryRuntime

logger = logging.getLogger(__name__)

TRANSIENT_HTTP_RETRY_DELAY_MS = 2500

# 裸 /new 或 /reset 后作为首条用户消息注入，触发 Session Startup + 问候
BARE_SESSION_RESET_PROMPT = (
    "A new session was started via /new or /reset. "
    "Greet the user in your configured persona (IDENTITY.md is already in your system prompt). "
    "Be yourself - use your defined voice, mannerisms, and mood. "
    "Keep it to 1-3 sentences and ask what they want to do. "
    "If the runtime model differs from default_model in the system prompt, mention the default model. "
    "Do not mention internal files, tools, memory status, or reasoning."
)


def _should_persist_input_message(persist_input_role: str) -> bool:
    return bool((persist_input_role or "").strip())


def _new_tool_call_id() -> str:
    return f"tc_{uuid.uuid4().hex[:12]}"


def _infer_tool_result_status(output: str) -> tuple[str, str | None]:
    text = output or ""
    lowered = text.lower()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            status = str(parsed.get("status") or "").lower()
            if status == "error":
                return "error", str(parsed.get("error") or "")[:500] or None
    except Exception:
        pass
    if "命令被拒绝" in text or "command rejected" in lowered or "已拒绝" in text:
        return "denied", text[:500]
    if "timed out" in lowered or "超时" in text:
        return "timeout", text[:500]
    if "execution error" in lowered or "执行错误" in text or "执行出错" in text:
        return "error", text[:500]
    return "success", None


def _loop_warning_is_breaker(warning: str) -> bool:
    return "全局熔断" in warning or "circuit" in warning.lower()


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


@dataclass
class PromptCacheEntry:
    key: tuple[Any, ...]
    system_prompt: str
    prompt_report: Any
    prompt_tokens: int


@dataclass
class SessionContextCacheEntry:
    agent_id: str
    session_id: str
    session_file_mtime: float | None
    summary_fingerprint: str | None
    raw_history: list[dict[str, Any]]
    summary_text: str
    history_with_summary: list[dict[str, Any]]
    pruned_history: list[dict[str, Any]]
    summary_tokens: int
    history_tokens: int


@dataclass
class ToolNameCacheEntry:
    key: tuple[Any, ...]
    tool_names: tuple[str, ...]



from infra.event_bus import EventBus, Events, event_bus


# ---------------------------------------------------------------------------
# AgentManager — 核心引擎
# ---------------------------------------------------------------------------

class AgentManager:
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

    def __init__(self):
        self.data_dir: str = ""
        self._memory_runtime = MemoryRuntime()
        self._states: dict[str, AgentState] = {}
        self._initialized = False
        self.lifecycle_hooks: LifecycleHooks | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._state_save_tasks: dict[str, asyncio.Task] = {}
        self._prompt_cache: dict[tuple[Any, ...], PromptCacheEntry] = {}
        self._session_context_cache: dict[tuple[str, str], SessionContextCacheEntry] = {}
        self._tool_name_cache: dict[tuple[Any, ...], ToolNameCacheEntry] = {}

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
        from llm.llm_factory import llm_cache
        from llm.model_selection import resolve_agent_model

        ref = resolve_agent_model(agent_id)
        return llm_cache.get_or_create(agent_id, ref)

    def get_current_model_ref(self, agent_id: str = "main"):
        """获取 Agent 当前使用的 ModelRef"""
        from llm.model_selection import resolve_agent_model
        return resolve_agent_model(agent_id)

    def switch_model(self, agent_id: str, model_raw: str) -> str:
        """运行时切换 Agent 模型，返回新模型描述"""
        from llm.llm_factory import llm_cache
        from llm.model_selection import resolve_agent_model, get_model_display_name
        from llm.models_config import parse_model_ref

        ref = parse_model_ref(model_raw)
        if not ref:
            raise ValueError(f"Invalid model reference: {model_raw}")

        if not ref.provider:
            from llm.models_config import models_config
            found = models_config.find_model_by_id(ref.model)
            if found:
                provider, model_def = found
                ref.provider = provider.id
            else:
                raise ValueError(f"Model '{ref.model}' not found in any provider")

        llm_cache.invalidate(agent_id)
        llm_cache.get_or_create(agent_id, ref)

        return get_model_display_name(ref)

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
        workspace = str(resolve_agent_workspace(agent_id))
        agent_dir = str(resolve_agent_dir(agent_id))

        from tools.file_tools import get_file_tools
        from tools.exec_tools import get_exec_tools
        from tools.web_tools import get_web_tools
        from tools.memory_tools import get_memory_tools
        from tools.knowledge_tool import get_knowledge_tools
        from tools.agent_tools import get_agent_tools
        from tools.cron_tools import get_cron_tools
        from tools.status_tool import get_status_tools

        tools = []
        tools.extend(get_file_tools(workspace, agent_id=agent_id))
        tools.extend(get_exec_tools(workspace, agent_id))
        tools.extend(get_web_tools())
        tools.extend(get_memory_tools(agent_id=agent_id))
        tools.extend(get_knowledge_tools(agent_dir))
        tools.extend(get_agent_tools(agent_id, self, session_id))
        tools.extend(get_cron_tools(agent_id))
        tools.extend(get_status_tools(agent_id, session_id))
        return tools

    def _wrap_tools_for_session(self, agent_id: str, session_id: str, tools: list) -> list:
        from tools.persistence_wrapper import wrap_tools_for_persistence

        return wrap_tools_for_persistence(
            tools,
            data_dir=self.data_dir,
            agent_id=agent_id,
            session_id=session_id,
        )

    def _build_tools(self, agent_id: str, session_id: str = "") -> list:
        tools = self._collect_tools(agent_id, session_id)

        tools = self._filter_tools_by_policy(agent_id, tools)
        tools = self._wrap_tools_for_session(agent_id, session_id, tools)
        return tools

    def _resolve_tool_policy(self, agent_id: str) -> tuple[list[str], list[str]]:
        """解析 agent 的工具 allow/deny 策略。"""
        from config import get_config

        cfg = get_config()
        agent_entry = None
        for a in (cfg.get("agents", {}).get("list") or []):
            if a.get("id") == agent_id:
                agent_entry = a
                break
        policy = (agent_entry or {}).get("tools") or {}
        defaults_policy = (cfg.get("agents", {}).get("defaults", {}).get("tools")) or {}
        deny = list(policy.get("deny") or defaults_policy.get("deny") or [])
        allow = list(policy.get("allow") or defaults_policy.get("allow") or [])
        return allow, deny

    def _filter_tools_by_policy(self, agent_id: str, tools: list) -> list:
        """按 agents.list[].tools.allow/deny 过滤工具"""
        allow, deny = self._resolve_tool_policy(agent_id)

        def _normalize(name: str) -> str:
            return name.replace("-", "_").lower().strip()

        deny_set = {_normalize(d) for d in deny if d}
        allow_set = {_normalize(a) for a in allow if a} if allow else None

        def _is_allowed(tool_name: str) -> bool:
            n = _normalize(tool_name)
            if n in deny_set:
                return False
            if allow_set is None:
                return True
            if n in allow_set:
                return True
            if n == "apply_patch" and "exec" in allow_set:
                return True
            return False

        return [t for t in tools if _is_allowed(t.name)]

    def _build_messages(
        self, history: list[dict[str, Any]], new_message: str
    ) -> list:
        messages = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                if not messages:
                    messages.append(SystemMessage(content=content))
                else:
                    # Anthropic 协议不允许 system 消息穿插在 user/assistant 之间，
                    # 降级为 HumanMessage，内容保留 [System Message] 前缀供 LLM 识别来源
                    messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=new_message))
        return messages

    @staticmethod
    def _safe_mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime if path.exists() else None
        except Exception:
            return None

    def _project_context_signature(self, agent_id: str, prompt_mode: str) -> tuple[Any, ...]:
        workspace = resolve_agent_workspace(agent_id)
        files: list[Path] = [workspace / "AGENTS.md"]
        if prompt_mode == "full":
            files.extend([workspace / "IDENTITY.md", workspace / "USER.md"])
        snapshot = resolve_agent_dir(agent_id) / "SKILLS_SNAPSHOT.md"
        bootstrap = workspace / "BOOTSTRAP.md"
        return (
            tuple((str(p), self._safe_mtime(p)) for p in files),
            self._safe_mtime(snapshot),
            bootstrap.exists(),
        )

    def _tool_policy_signature(self, agent_id: str) -> tuple[Any, ...]:
        allow_list, deny_list = self._resolve_tool_policy(agent_id)
        allow = tuple(sorted(str(x) for x in allow_list))
        deny = tuple(sorted(str(x) for x in deny_list))
        return allow, deny

    def _get_or_build_tool_names(self, agent_id: str) -> tuple[str, ...]:
        cache_key = (agent_id, self._tool_policy_signature(agent_id))
        cached = self._tool_name_cache.get(cache_key)
        if cached is not None:
            return cached.tool_names

        tools = self._collect_tools(agent_id, session_id="")
        tools = self._filter_tools_by_policy(agent_id, tools)
        tool_names = tuple(sorted(t.name for t in tools))
        self._tool_name_cache[cache_key] = ToolNameCacheEntry(
            key=cache_key,
            tool_names=tool_names,
        )
        return tool_names

    def _get_or_build_prompt(
        self,
        *,
        agent_id: str,
        prompt_mode: str,
        available_tool_names: list[str] | None,
        extra_system_prompt: str | None,
        locale: str,
    ) -> tuple[str, Any, int]:
        from runtime.prompt_builder import PromptParams
        from infra.token_counter import count_tokens

        tool_key = tuple(sorted(available_tool_names or []))
        static_sig = self._project_context_signature(agent_id, prompt_mode)
        cache_key = (
            agent_id,
            prompt_mode,
            tool_key,
            extra_system_prompt or "",
            locale,
            static_sig,
        )
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached.system_prompt, cached.prompt_report, cached.prompt_tokens

        prompt_params = PromptParams(
            agent_id=agent_id,
            mode=prompt_mode,
            available_tools=available_tool_names,
            extra_system_prompt=extra_system_prompt or None,
            locale=locale,
        )
        system_prompt, prompt_report = prompt_builder.build_system_prompt_with_report(prompt_params)
        prompt_tokens = count_tokens(system_prompt)

        self._prompt_cache[cache_key] = PromptCacheEntry(
            key=cache_key,
            system_prompt=system_prompt,
            prompt_report=prompt_report,
            prompt_tokens=prompt_tokens,
        )
        return system_prompt, prompt_report, prompt_tokens

    @staticmethod
    def _session_summary_fingerprint(summary: Any) -> str | None:
        if not summary:
            return None
        try:
            if isinstance(summary, dict):
                payload = summary
            else:
                payload = {
                    "goal": getattr(summary, "goal", None),
                    "decisions": getattr(summary, "decisions", None),
                    "progress": getattr(summary, "progress", None),
                    "open_items": getattr(summary, "open_items", None),
                    "entities": getattr(summary, "entities", None),
                    "user_preferences": getattr(summary, "user_preferences", None),
                    "raw_summary": getattr(summary, "raw_summary", None),
                }
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(summary)

    def _get_or_build_session_context(
        self,
        *,
        agent_id: str,
        session_id: str,
    ) -> SessionContextCacheEntry:
        from infra.token_counter import count_tokens

        cache_key = (agent_id, session_id)
        session_path = resolve_agent_dir(agent_id) / "sessions" / f"{session_id}.json"
        session_file_mtime = self._safe_mtime(session_path)

        store = self.mem_stores.get(agent_id)
        session_summary = store.get_session_summary(session_id, agent_id) if store else None
        summary_fingerprint = self._session_summary_fingerprint(session_summary)

        cached = self._session_context_cache.get(cache_key)
        if (
            cached is not None
            and cached.session_file_mtime == session_file_mtime
            and cached.summary_fingerprint == summary_fingerprint
        ):
            return cached

        raw_history = session_manager.load_session_for_agent(session_id, agent_id)
        summary_text = ""
        history_with_summary = list(raw_history)
        if session_summary:
            summary_text = prompt_builder.format_session_summary(session_summary)
            if summary_text:
                history_with_summary = [{"role": "system", "content": summary_text}, *history_with_summary]

        pruned_history = prune_messages(history_with_summary, agent_id=agent_id)
        summary_tokens = count_tokens(summary_text) if summary_text else 0
        history_tokens = count_messages_tokens(pruned_history)

        entry = SessionContextCacheEntry(
            agent_id=agent_id,
            session_id=session_id,
            session_file_mtime=session_file_mtime,
            summary_fingerprint=summary_fingerprint,
            raw_history=raw_history,
            summary_text=summary_text,
            history_with_summary=history_with_summary,
            pruned_history=pruned_history,
            summary_tokens=summary_tokens,
            history_tokens=history_tokens,
        )
        self._session_context_cache[cache_key] = entry
        return entry

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
        from tools.skills_scanner import write_skills_snapshot

        state = self.get_state(agent_id)

        # 命令处理
        parsed = parse_command(message)
        if parsed:
            result = await execute_command(parsed, agent_id, session_id, state)
            if result.get("handled"):
                action = result.get("action", "")

                if action == "reset":
                    # /new：保存 session-memory 后重置，再注入 BARE_SESSION_RESET_PROMPT 跑一轮问候
                    model_override = result.get("model_override")
                    async for evt in self._handle_reset(
                        session_id, agent_id, model_override=model_override
                    ):
                        yield evt
                    persist_input_role = ""
                    message = BARE_SESSION_RESET_PROMPT
                elif action == "reset_noflush":
                    # /reset：不写入 session-memory 的轻量重置，再注入 BARE_SESSION_RESET_PROMPT 跑一轮问候
                    async for evt in self._handle_reset_noflush(session_id, agent_id):
                        yield evt
                    persist_input_role = ""
                    message = BARE_SESSION_RESET_PROMPT
                else:
                    if action == "compact":
                        async for evt in self._handle_compact(session_id, agent_id):
                            yield evt
                        return
                    if action == "stop":
                        yield {"type": "command_response", "response": result["response"]}
                        yield {"type": "done", "content": result["response"], "session_id": session_id}
                        return
                    yield {"type": "command_response", "response": result["response"]}
                    yield {"type": "done", "content": result["response"], "session_id": session_id}
                    return

        write_skills_snapshot(agent_id)

        # 检测 BOOTSTRAP.md
        from runtime.workspace import has_bootstrap
        extra_prompt = ""
        if has_bootstrap(agent_id):
            bootstrap_path = resolve_agent_workspace(agent_id) / "BOOTSTRAP.md"
            try:
                extra_prompt = (
                    "\n\n## 首次运行引导\n\n"
                    "检测到 BOOTSTRAP.md，请先读取并执行其中的引导步骤。"
                    "完成后删除该文件。\n"
                )
            except Exception:
                pass

        available_tool_names = list(self._get_or_build_tool_names(agent_id))

        from config import get_config
        _locale = get_config().get("app", {}).get("locale", "zh-CN")
        system_prompt, prompt_report, _sp_tokens = self._get_or_build_prompt(
            agent_id=agent_id,
            prompt_mode=prompt_mode,
            available_tool_names=available_tool_names,
            extra_system_prompt=extra_prompt or None,
            locale=_locale,
        )
        logger.info(prompt_report.summary())

        context_entry = self._get_or_build_session_context(agent_id=agent_id, session_id=session_id)
        history = context_entry.pruned_history
        tools = self._build_tools(agent_id, session_id)

        from runtime.context_budget import resolve_budget
        _budget = resolve_budget(agent_id)
        _summary_tokens = context_entry.summary_tokens
        _history_tokens = context_entry.history_tokens

        agent_cfg = resolve_agent_config(agent_id)
        recursion_limit = agent_cfg.get("recursion_limit", 50)

        candidates = resolve_fallback_candidates(agent_id)
        did_retry_transient = False
        did_reset_compaction = False
        did_retry_forced_compaction = False

        async def run_for_model(provider: str, model: str):
            ref = ModelRef(provider=provider, model=model)
            try:
                llm = create_llm(ref)
            except Exception as e:
                yield {"type": "error", "error": f"LLM 初始化失败: {e}"}
                return

            try:
                from langgraph.prebuilt import create_react_agent
                agent = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt=system_prompt,
                )
            except ImportError:
                yield {"type": "error", "error": "langgraph 未安装"}
                return

            lc_messages = self._build_messages(history, message)

            turn = run_tracker.start_turn(agent_id, session_id)
            audit_logger.log_turn_start(agent_id, turn.run_id, session_id)
            yield Events.turn_start(run_id=turn.run_id, model=str(ref))

            full_response = ""
            tool_calls_log: list[dict[str, Any]] = []
            tool_input_by_run_id: dict[str, Any] = {}
            tool_call_id_by_run_id: dict[str, str] = {}
            _streaming_model_run_id: str | None = None
            step_count = 0
            _content_refresh_sent = False
            recent_untrusted_content = any(
                contains_untrusted_marker(str(msg.get("content", ""))) for msg in history[-4:]
            )
            from sandbox.loop_detection import LoopDetector
            loop_detector = LoopDetector()

            try:
                with runtime_security_context(
                    message,
                    recent_untrusted_content=recent_untrusted_content,
                ):
                    async for event in agent.astream_events(
                        {"messages": lc_messages},
                        version="v2",
                        config={"recursion_limit": recursion_limit},
                    ):
                        kind = event.get("event", "")

                        if kind == "on_chat_model_stream":
                            evt_run_id = event.get("run_id", "")
                            if _streaming_model_run_id is None:
                                _streaming_model_run_id = evt_run_id
                            elif evt_run_id != _streaming_model_run_id:
                                continue

                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                content = chunk.content
                                if isinstance(content, str):
                                    full_response += content
                                    yield {"type": "token", "content": content}
                                elif isinstance(content, list):
                                    for block in content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text = block.get("text", "")
                                            if text:
                                                full_response += text
                                                yield {"type": "token", "content": text}

                            if chunk and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                                usage = chunk.usage_metadata
                                run_tracker.record_tokens(
                                    turn.run_id,
                                    input_tokens=getattr(usage, "input_tokens", 0),
                                    output_tokens=getattr(usage, "output_tokens", 0),
                                    cache_read=getattr(usage, "input_token_details", {}).get("cache_read", 0) if hasattr(usage, "input_token_details") else 0,
                                )

                        elif kind == "on_chat_model_end":
                            if event.get("run_id") == _streaming_model_run_id:
                                _streaming_model_run_id = None

                        elif kind == "on_tool_start":
                            if not _content_refresh_sent and full_response and parse_text_tool_calls(full_response):
                                cleaned = strip_tool_call_patterns(full_response)
                                yield {"type": "content_refresh", "content": cleaned}
                                _content_refresh_sent = True

                            tool_name = event.get("name", "")
                            tool_input = event.get("data", {}).get("input") or {}
                            if not isinstance(tool_input, dict):
                                tool_input = {}
                            if self.lifecycle_hooks:
                                await self.lifecycle_hooks.on_before_tool_call(
                                    agent_id, turn.run_id, tool_name, tool_input
                                )
                            step_count += 1
                            evt_run_id = str(event.get("run_id", ""))
                            tool_call_id = _new_tool_call_id()
                            if evt_run_id:
                                tool_input_by_run_id[evt_run_id] = tool_input
                                tool_call_id_by_run_id[evt_run_id] = tool_call_id
                            run_tracker.record_tool_start(
                                turn.run_id,
                                tool_name,
                                tool_input,
                                tool_call_id=tool_call_id,
                            )
                            yield {
                                "type": "tool_start",
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "input": tool_input,
                                "step": step_count,
                                "max_steps": recursion_limit,
                            }

                        elif kind == "on_tool_end":
                            tool_output = event.get("data", {}).get("output", "")
                            if isinstance(tool_output, str):
                                output_str = tool_output
                            elif hasattr(tool_output, "content") and tool_output.content is not None:
                                output_str = str(tool_output.content)
                            else:
                                output_str = str(tool_output)

                            evt_run_id = str(event.get("run_id", ""))
                            tool_input = tool_input_by_run_id.pop(evt_run_id, None)
                            tool_call_id = tool_call_id_by_run_id.pop(evt_run_id, None) or _new_tool_call_id()
                            tool_input_for_log = tool_input if tool_input is not None else ""
                            tool_name = event.get("name", "")
                            status, error = _infer_tool_result_status(output_str)
                            run_tracker.record_tool_end(
                                turn.run_id,
                                tool_name,
                                output_str,
                                error=error,
                                tool_call_id=tool_call_id,
                            )
                            audit_logger.log_tool_call(
                                agent_id,
                                turn.run_id,
                                tool_name,
                                tool_input_for_log,
                                output_str,
                                tool_call_id=tool_call_id,
                                status=status,
                                error=error,
                            )

                            tool_calls_log.append(
                                {
                                    "tool_call_id": tool_call_id,
                                    "tool": tool_name,
                                    "status": status,
                                    "input": tool_input_for_log,
                                    "output": output_str,
                                    "error": error,
                                }
                            )
                            if self.lifecycle_hooks:
                                await self.lifecycle_hooks.on_after_tool_call(
                                    agent_id, turn.run_id, tool_name, tool_input_for_log, output_str
                                )
                            if is_untrusted_source_tool(tool_name):
                                recent_untrusted_content = True
                                mark_recent_untrusted_content(True)
                            yield {
                                "type": "tool_end",
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "status": status,
                                "error": error,
                                "output": output_str[:2000],
                            }

                            loop_warning = loop_detector.record(
                                tool_name,
                                tool_input_for_log,
                                output_str,
                            )
                            if loop_warning:
                                audit_logger.log_tool_loop_warning(
                                    agent_id,
                                    turn.run_id,
                                    tool_name,
                                    loop_warning,
                                    tool_call_id=tool_call_id,
                                )
                                loop_event = Events.tool_loop_warning(
                                    run_id=turn.run_id,
                                    tool=tool_name,
                                    warning=loop_warning,
                                    tool_call_id=tool_call_id,
                                )
                                event_bus.emit(agent_id, loop_event)
                                yield loop_event
                                if _loop_warning_is_breaker(loop_warning):
                                    raise RuntimeError(loop_warning)

                            if tool_name in ("exec", "process_kill"):
                                safe_input = str(tool_input_for_log)[:200] if tool_input_for_log else ""
                                event_bus.emit(
                                    agent_id,
                                    Events.tool_dangerous_executed(tool=tool_name, input_preview=safe_input),
                                )

            except Exception as e:
                error_str = str(e)
                is_recursion = "recursion" in error_str.lower() or "GraphRecursionError" in type(e).__name__
                run_tracker.error_turn(turn.run_id, error_str)
                audit_logger.log_turn_error(agent_id, turn.run_id, error_str)
                if is_recursion:
                    yield Events.recursion_limit_reached(step=step_count, max_steps=recursion_limit)
                    yield {
                        "type": "error",
                        "error": f"Agent 达到最大迭代次数 ({recursion_limit})，已自动停止。已执行 {step_count} 步工具调用。",
                    }
                else:
                    yield Events.turn_error(error=error_str)
                    yield {"type": "error", "error": error_str}
                return

            # 生命周期: Turn 完成
            completed = run_tracker.complete_turn(turn.run_id)
            if completed:
                state.record_turn(completed.input_tokens, completed.output_tokens)
                audit_logger.log_turn_end(
                    agent_id, turn.run_id, session_id,
                    tokens={"input": completed.input_tokens, "output": completed.output_tokens},
                    tool_calls=len(tool_calls_log),
                    duration_ms=completed.duration_ms,
                )

            # Fallback: 模型以文本形式输出 tool call 时，解析并执行（Kimi K2 等）
            parsed_calls = parse_text_tool_calls(full_response)
            if parsed_calls and not tool_calls_log:
                if not _content_refresh_sent:
                    cleaned = strip_tool_call_patterns(full_response)
                    yield {"type": "content_refresh", "content": cleaned}
                    _content_refresh_sent = True
                tool_names = {getattr(t, "name", ""): t for t in tools}
                for fallback_tool_name, fallback_tool_args in parsed_calls:
                    matched_tool = tool_names.get(fallback_tool_name)
                    if matched_tool:
                        step_count += 1
                        args_to_use = dict(fallback_tool_args) if fallback_tool_args else {}
                        if fallback_tool_name == "read" and not args_to_use.get("path"):
                            args_to_use["path"] = "IDENTITY.md"
                            logger.info(f"Fallback read: 无 path 参数，使用默认 IDENTITY.md")
                        tool_call_id = _new_tool_call_id()
                        run_tracker.record_tool_start(
                            turn.run_id,
                            fallback_tool_name,
                            args_to_use,
                            tool_call_id=tool_call_id,
                        )
                        logger.info(f"Fallback tool call: {fallback_tool_name}({args_to_use})")
                        yield {
                            "type": "tool_start",
                            "tool_call_id": tool_call_id,
                            "tool": fallback_tool_name,
                            "input": args_to_use,
                            "step": step_count, "max_steps": recursion_limit,
                        }
                        try:
                            result_str = (
                                await invoke_tool_async(
                                    matched_tool,
                                    args_to_use,
                                    user_message=message,
                                    recent_untrusted_content=recent_untrusted_content,
                                )
                            )[:2000]
                        except Exception as te:
                            from tools.error_utils import format_tool_error
                            result_str = format_tool_error(fallback_tool_name, te)
                        if is_untrusted_source_tool(fallback_tool_name):
                            recent_untrusted_content = True
                        status, error = _infer_tool_result_status(result_str)
                        run_tracker.record_tool_end(
                            turn.run_id,
                            fallback_tool_name,
                            result_str,
                            error=error,
                            tool_call_id=tool_call_id,
                        )
                        audit_logger.log_tool_call(
                            agent_id,
                            turn.run_id,
                            fallback_tool_name,
                            args_to_use,
                            result_str,
                            tool_call_id=tool_call_id,
                            status=status,
                            error=error,
                        )
                        yield {
                            "type": "tool_end",
                            "tool_call_id": tool_call_id,
                            "tool": fallback_tool_name,
                            "status": status,
                            "error": error,
                            "output": result_str,
                        }
                        tool_calls_log.append({
                            "tool_call_id": tool_call_id,
                            "tool": fallback_tool_name,
                            "status": status,
                            "input": args_to_use,
                            "output": result_str,
                            "error": error,
                        })
                        loop_warning = loop_detector.record(
                            fallback_tool_name,
                            args_to_use,
                            result_str,
                        )
                        if loop_warning:
                            audit_logger.log_tool_loop_warning(
                                agent_id,
                                turn.run_id,
                                fallback_tool_name,
                                loop_warning,
                                tool_call_id=tool_call_id,
                            )
                            loop_event = Events.tool_loop_warning(
                                run_id=turn.run_id,
                                tool=fallback_tool_name,
                                warning=loop_warning,
                                tool_call_id=tool_call_id,
                            )
                            event_bus.emit(agent_id, loop_event)
                            yield loop_event
                            if _loop_warning_is_breaker(loop_warning):
                                yield Events.turn_error(error=loop_warning)
                                yield {"type": "error", "error": loop_warning}
                                return
                full_response = strip_tool_call_patterns(full_response)

            # 保存消息（若含文本形式工具调用则保存清理后的 content）
            if _should_persist_input_message(persist_input_role):
                session_manager.save_message(session_id, agent_id, persist_input_role, message)
            content_to_save = strip_tool_call_patterns(full_response) if parse_text_tool_calls(full_response) else full_response
            session_manager.save_message(
                session_id, agent_id, "assistant", content_to_save,
                tool_calls=tool_calls_log if tool_calls_log else None,
            )

            write_skills_snapshot(agent_id)

            # 发送完成事件 (含 token 使用信息)
            usage_info = {}
            if completed:
                usage_info = {
                    "input_tokens": completed.input_tokens,
                    "output_tokens": completed.output_tokens,
                    "total_tokens": completed.total_tokens,
                    "duration_ms": completed.duration_ms,
                    "model": str(ref),
                }

            yield Events.turn_end(run_id=turn.run_id, usage=usage_info)
            done_content = strip_tool_call_patterns(full_response) if parse_text_tool_calls(full_response) else full_response

            _turn_tokens = count_tokens(message) + count_tokens(full_response)
            for _tc in tool_calls_log:
                _turn_tokens += count_tokens(str(_tc.get("output", "")))
            _total_ctx = _sp_tokens + _history_tokens + _turn_tokens
            _ctx_utilization = round(_total_ctx / _budget.active_tokens, 3) if _budget.active_tokens else 0

            yield {
                "type": "done",
                "content": done_content,
                "session_id": session_id,
                "usage": usage_info,
                "context_utilization": _ctx_utilization,
            }

            # 每轮异步入库 (非阻塞, hash 去重保证幂等)
            ingested_user_content = message if persist_input_role == "user" else ""
            task = asyncio.create_task(
                self._incremental_ingest(
                    agent_id, session_id, ingested_user_content, done_content,
                )
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

            # 自动压缩检测
            await self._maybe_auto_compact(session_id, agent_id, overhead_tokens=_sp_tokens + _summary_tokens)

        # 外层循环：瞬时 HTTP 重试、压缩失败/role ordering/session 损坏恢复
        while True:
            try:
                async for evt in run_with_fallback_stream(candidates, run_for_model, agent_id):
                    yield evt
                break
            except Exception as e:
                msg = str(e)
                if is_transient_http_error(msg) and not did_retry_transient:
                    did_retry_transient = True
                    logger.warning(
                        f"Transient HTTP error ({msg[:150]}). Retrying in {TRANSIENT_HTTP_RETRY_DELAY_MS}ms."
                    )
                    await asyncio.sleep(TRANSIENT_HTTP_RETRY_DELAY_MS / 1000)
                    continue

                if is_compaction_failure_error(msg) and not did_reset_compaction:
                    did_reset_compaction = True
                    session_manager.reset_session(session_id, agent_id)
                    state.compaction_count = 0
                    audit_logger.log(agent_id, "session_reset_compaction_failure", {"error": msg[:200]})
                    yield {
                        "type": "session_reset",
                        "session_id": session_id,
                        "memory": {"saved": False, "reason": "compaction_failure"},
                    }
                    yield {
                        "type": "done",
                        "content": (
                            "⚠️ 上下文超出限制，压缩失败。已重置会话，请重试。\n\n"
                            "建议在 config 中提高 agents.defaults.compaction.reserveTokensFloor（如 20000）以降低此问题。"
                        ),
                        "session_id": session_id,
                    }
                    return

                if is_role_ordering_error(msg):
                    session_manager.reset_session(session_id, agent_id)
                    state.compaction_count = 0
                    yield {"type": "session_reset", "session_id": session_id, "memory": {"saved": False}}
                    yield {
                        "type": "done",
                        "content": "⚠️ 消息顺序冲突，已重置会话，请重试。",
                        "session_id": session_id,
                    }
                    return

                if is_session_corruption_error(msg):
                    session_manager.reset_session(session_id, agent_id)
                    state.compaction_count = 0
                    yield {"type": "session_reset", "session_id": session_id, "memory": {"saved": False}}
                    yield {
                        "type": "done",
                        "content": "⚠️ 会话历史损坏，已重置，请重试。",
                        "session_id": session_id,
                    }
                    return

                if is_likely_context_overflow_error(msg):
                    if not did_retry_forced_compaction:
                        did_retry_forced_compaction = True
                        logger.warning(
                            "Context overflow detected for agent=%s session=%s. "
                            "Attempting forced compaction retry.",
                            agent_id,
                            session_id,
                        )
                        try:
                            forced_result = await self.compress_session(
                                session_id, agent_id, level="forced"
                            )
                            if "error" not in forced_result:
                                audit_logger.log(
                                    agent_id,
                                    "forced_compaction_retry",
                                    {"session_id": session_id, "reason": msg[:200]},
                                )
                                continue
                            logger.warning(
                                "Forced compaction retry skipped for agent=%s session=%s: %s",
                                agent_id,
                                session_id,
                                forced_result.get("error", "unknown"),
                            )
                        except Exception as forced_err:
                            logger.warning(
                                "Forced compaction retry failed for agent=%s session=%s: %s",
                                agent_id,
                                session_id,
                                forced_err,
                            )

                    yield {
                        "type": "error",
                        "error": "⚠️ 上下文溢出，已尝试紧急压缩但仍失败。请缩短消息或使用更大 context 的模型。",
                    }
                    return

                yield Events.turn_error(error=msg)
                yield {"type": "error", "error": msg}
                return

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
        """/new：重置会话，先将当前 session 消息批量入库（session_end=True）"""
        yield {"type": "command_response", "response": "正在重置会话..."}

        data = session_manager.load_session(session_id, agent_id)
        if data:
            all_msgs = data.get("messages", [])
            if all_msgs:
                await self._batch_ingest_messages(agent_id, session_id, all_msgs, session_end=True)

        session_manager.reset_session(session_id, agent_id)

        store = self.mem_stores.get(agent_id)
        if store:
            store.delete_session_summary(session_id, agent_id)

        state = self.get_state(agent_id)
        state.compaction_count = 0

        model_msg = ""
        if model_override:
            try:
                new_name = self.switch_model(agent_id, model_override)
                model_msg = f" 模型已切换到 {new_name}。"
            except Exception as e:
                model_msg = f" 模型切换失败: {e}"

        audit_logger.log(
            agent_id,
            "session_reset",
            {
                "session_id": session_id,
                "model_override": model_override,
            },
        )

        msg = "会话已重置。" + model_msg

        yield {"type": "command_response", "response": msg}
        yield {"type": "session_reset", "session_id": session_id}
        # 不 yield done：主流程会接着用 BARE_SESSION_RESET_PROMPT 跑问候，由 agent 流产出 done

    async def _handle_reset_noflush(
        self, session_id: str, agent_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """/reset：不写入 session-memory 的轻量重置，仅归档会话文件。"""
        yield {"type": "command_response", "response": "正在重置会话（不写入长期记忆）..."}

        session_manager.reset_session(session_id, agent_id)

        store = self.mem_stores.get(agent_id)
        if store:
            store.delete_session_summary(session_id, agent_id)

        state = self.get_state(agent_id)
        state.compaction_count = 0

        audit_logger.log(
            agent_id,
            "session_reset",
            {
                "session_id": session_id,
                "memory_saved": False,
                "mode": "no_memory",
            },
        )

        msg = "会话已重置（本轮对话未写入长期记忆）。"
        yield {"type": "command_response", "response": msg}
        yield {
            "type": "session_reset",
            "session_id": session_id,
            "memory": {"saved": False, "reason": "no-flush"},
        }
        # 不 yield done：主流程会接着用 BARE_SESSION_RESET_PROMPT 跑问候，由 agent 流产出 done

    # ------------------------------------------------------------------
    # /compact 命令处理
    # ------------------------------------------------------------------

    async def _handle_compact(
        self, session_id: str, agent_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "command_response", "response": "正在执行压缩..."}
        event_bus.emit(agent_id, Events.manual_compact_start(session_id=session_id))

        try:
            result = await self.compress_session(session_id, agent_id)
            if "error" in result:
                reason = str(result.get("error") or "未知原因")
                session_data = session_manager.load_session(session_id, agent_id) or {}
                messages = session_data.get("messages", []) or []
                msg_tokens = 0
                total_tokens = 0
                threshold = 0
                keep_recent_tokens = 8000
                compressible_count = 0
                try:
                    from infra.token_counter import (
                        count_messages_tokens,
                        resolve_compaction_threshold,
                    )
                    msg_tokens = count_messages_tokens(messages)
                    total_tokens = msg_tokens
                    threshold = resolve_compaction_threshold(agent_id)
                except Exception:
                    pass
                try:
                    compaction_cfg = resolve_agent_config(agent_id).get("compaction", {})
                    keep_recent_turns = int(compaction_cfg.get("keepRecentTurns", 12) or 12)
                except Exception:
                    keep_recent_turns = 8
                try:
                    compressible_count = self._calc_compress_count_by_turns(messages, keep_recent_turns)
                except Exception:
                    compressible_count = 0

                suggestion = "建议：继续对话累积上下文，或降低 compaction.keepRecentTurns。"
                if reason == "消息过少，无需压缩":
                    suggestion = "建议：至少累积到 4 条以上消息后再尝试。"
                elif reason == "无足够消息可压缩":
                    suggestion = (
                        f"建议：当前轮次不足 keepRecentTurns({keep_recent_turns})，"
                        "可继续对话后重试，或调低 compaction.keepRecentTurns。"
                    )
                elif reason == "会话不存在":
                    suggestion = "建议：先发送一条消息创建会话，再执行 /compact。"

                msg = (
                    f"压缩未执行：{reason}\n"
                    f"\n当前状态（动态）:\n"
                    f"- 消息数: {len(messages)}\n"
                    f"- 消息 tokens: {msg_tokens}\n"
                    f"- 总 tokens: {total_tokens}\n"
                    f"- 压缩阈值(sliding threshold): {threshold}\n"
                    f"- 保留轮次(compaction.keepRecentTurns): {keep_recent_turns}\n"
                    f"- 当前可压缩消息数: {compressible_count}\n"
                    f"\n{suggestion}"
                )
                yield {"type": "command_response", "response": msg}
                event_bus.emit(agent_id, Events.manual_compact_skipped(session_id=session_id, reason=result.get("error", "")))
                yield {"type": "done", "content": msg, "session_id": session_id}
                return

            c = result.get("compress", {}) or {}

            msg = (
                f"压缩完成。\n"
                f"- 归档消息：{c.get('archived_count', 0)} 条\n"
                f"- 剩余消息：{c.get('remaining_count', 0)} 条"
            )
            yield {"type": "command_response", "response": msg}
            yield {"type": "session_compacted", "result": result}
            event_bus.emit(agent_id, Events.manual_compact_done(
                session_id=session_id,
                data={"archived_count": c.get("archived_count", 0), "remaining_count": c.get("remaining_count", 0)},
            ))
            yield {"type": "done", "content": msg, "session_id": session_id}
        except Exception as e:
            event_bus.emit(agent_id, Events.manual_compact_error(session_id=session_id, error=str(e)[:200]))
            yield {"type": "error", "error": f"压缩失败: {e}"}

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
        """生成结构化摘要 dict，失败时降级为 raw_summary。"""
        from llm.retry import retry_async
        from infra.token_counter import count_tokens
        from llm.model_selection import resolve_agent_model, get_model_context_window

        store = self.mem_stores.get(agent_id)
        prev_summary: dict[str, Any] = {}
        if store:
            existing = store.get_session_summary(session_id, agent_id)
            if existing:
                prev_summary = {
                    "goal": existing.goal,
                    "decisions": json.loads(existing.decisions) if existing.decisions else [],
                    "progress": existing.progress,
                    "open_items": json.loads(existing.open_items) if existing.open_items else [],
                    "entities": json.loads(existing.entities) if existing.entities else [],
                    "user_preferences": json.loads(existing.user_preferences) if existing.user_preferences else [],
                }

        prev_block = ""
        if prev_summary:
            prev_block = (
                "\n\n## 上一版摘要（请在此基础上更新，而非重新生成）\n"
                f"{json.dumps(prev_summary, ensure_ascii=False, indent=2)}"
            )

        system_prompt = (
            "你是一个对话摘要生成器。将对话历史压缩为结构化 JSON 摘要。\n\n"
            "关键语言规则：使用与用户消息相同的语言输出。中文输入→中文输出。英文输入→英文输出。\n\n"
            "输出严格的 JSON（无 markdown 代码块包裹）：\n"
            "{\n"
            '  "goal": "用户的总体目标（1 句话）",\n'
            '  "decisions": ["关键决策1", "关键决策2"],\n'
            '  "progress": "当前进展到哪一步",\n'
            '  "open_items": ["待办事项1", "未解决问题1"],\n'
            '  "entities": ["关键实体: 版本号/路径/配置值等"],\n'
            '  "user_preferences": ["用户偏好1"]\n'
            "}\n\n"
            "规则：\n"
            "- 保留所有关键信息：命令、文件路径、配置值、版本号、错误信息\n"
            "- 丢弃寒暄、填充词\n"
            "- 如果有上一版摘要，在其基础上更新（合并/覆盖），不要丢弃仍然有效的信息\n"
            "- 敏感信息替换为 [REDACTED]\n"
            "- 只输出 JSON，不要输出其他内容"
        )

        from runtime.context_budget import resolve_budget
        budget = resolve_budget(agent_id)
        summary_max_tokens = budget.session_summary_tokens

        async def _do_structured(text: str) -> dict[str, Any]:
            llm = self.get_llm(agent_id)
            resp = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=text + prev_block),
                ],
                max_tokens=summary_max_tokens,
            )
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            parsed["raw_summary"] = raw
            parsed["token_count"] = count_tokens(raw)
            return parsed

        try:
            return await retry_async(
                lambda: _do_structured(text_to_summarize),
                attempts=3,
                min_delay_ms=500,
                max_delay_ms=5000,
                jitter=0.2,
                should_retry=lambda e, _: "AbortError" not in type(e).__name__,
            )
        except Exception as e:
            logger.warning("Structured summary failed, falling back to plain text: %s", e)

        return await self._summarize_plain_fallback(
            agent_id, to_compress, text_to_summarize,
        )

    async def _summarize_plain_fallback(
        self,
        agent_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        """纯文本摘要降级：返回只含 raw_summary 的 dict。"""
        from llm.retry import retry_async
        from infra.token_counter import count_tokens
        from llm.model_selection import resolve_agent_model, get_model_context_window

        from runtime.context_budget import resolve_budget
        budget = resolve_budget(agent_id)
        summary_max_tokens = budget.session_summary_tokens

        async def _do_summarize(text: str) -> str:
            llm = self.get_llm(agent_id)
            resp = await llm.ainvoke(
                [
                    SystemMessage(content=(
                        "你是一个对话摘要生成器。请将以下对话历史压缩为简洁的摘要，不超过500字。"
                        "使用与用户消息相同的语言。保留关键信息、决定、上下文和待办事项。"
                    )),
                    HumanMessage(content=text),
                ],
                max_tokens=summary_max_tokens,
            )
            return resp.content.strip()

        try:
            text = await retry_async(
                lambda: _do_summarize(text_to_summarize),
                attempts=3,
                min_delay_ms=500,
                max_delay_ms=5000,
                jitter=0.2,
                should_retry=lambda e, _: "AbortError" not in type(e).__name__,
            )
            return {"raw_summary": text, "token_count": count_tokens(text)}
        except Exception as full_err:
            logger.warning(f"Full summarization failed, trying partial: {full_err}")

        try:
            ref = resolve_agent_model(agent_id)
            context_window = get_model_context_window(ref)
            small_msgs: list[dict[str, Any]] = []
            oversized_notes: list[str] = []
            for m in to_compress:
                content = m.get("content", "")
                tokens = count_tokens(content) + 4
                if tokens > context_window * 0.5:
                    role = m.get("role", "message")
                    oversized_notes.append(
                        f"[Large {role} (~{tokens // 1000}K tokens) omitted from summary]"
                    )
                else:
                    small_msgs.append(m)

            if small_msgs:
                partial_text = "\n".join(
                    f"[{x.get('role', '?')}] {x.get('content', '')}"
                    for x in small_msgs
                )
                partial = await retry_async(
                    lambda: _do_summarize(partial_text),
                    attempts=2,
                    min_delay_ms=500,
                    max_delay_ms=3000,
                    jitter=0.2,
                )
                notes = "\n\n" + "\n".join(oversized_notes) if oversized_notes else ""
                text = partial + notes
                return {"raw_summary": text, "token_count": count_tokens(text)}
        except Exception as partial_err:
            logger.warning(f"Partial summarization failed: {partial_err}")

        fallback = (
            f"Context contained {len(to_compress)} messages. "
            "Summary unavailable due to size limits."
        )
        return {"raw_summary": fallback, "token_count": count_tokens(fallback)}

    async def compress_session(
        self, session_id: str, agent_id: str, level: str = "sliding",
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "compress": None,
            "post_compaction": None,
        }

        agent_cfg = resolve_agent_config(agent_id)
        compaction_cfg = agent_cfg.get("compaction", {})

        if level == "forced":
            keep_turns = int(compaction_cfg.get("forcedKeepRecentTurns", 4))
        else:
            keep_turns = int(compaction_cfg.get("keepRecentTurns", 12))

        data = session_manager.load_session(session_id, agent_id)
        if not data:
            return {**result, "error": "会话不存在"}

        messages = data.get("messages", [])
        if len(messages) < 4:
            return {**result, "error": "消息过少，无需压缩"}

        n = self._calc_compress_count_by_turns(messages, keep_turns)
        if n < 2:
            return {**result, "error": "无足够消息可压缩"}

        to_compress = messages[:n]

        text_to_summarize = "\n".join(
            f"[{m.get('role', '?')}] {m.get('content', '')}"
            for m in to_compress
        )

        summary_dict = await self._generate_structured_summary(
            agent_id, session_id, to_compress, text_to_summarize,
        )

        store = self.mem_stores.get(agent_id)
        if store:
            store.upsert_session_summary(session_id, agent_id, summary_dict)

        compress_result = session_manager.compress_history(
            session_id, agent_id, n,
        )
        result["compress"] = {"summary": summary_dict, "level": level, **compress_result}

        state = self.get_state(agent_id)
        state.compaction_count += 1

        audit_logger.log_compress(
            agent_id, session_id,
            compress_result.get("archived_count", 0),
            compress_result.get("remaining_count", 0),
        )

        task = asyncio.create_task(
            self._batch_ingest_messages(agent_id, session_id, to_compress, session_end=False)
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

        return result

    @staticmethod
    def _calc_compress_count_by_turns(messages: list[dict[str, Any]], keep_turns: int) -> int:
        """按轮次保留：从尾部数 keep_turns 轮（每轮 = 连续 user+assistant），前面全部压缩。

        token 预算兜底：如果保留的轮次 token 数超过 summary_max_tokens * 5，
        则减少保留轮数。
        """
        if not messages:
            return 0

        turn_boundaries: list[int] = []
        i = len(messages) - 1
        while i >= 0:
            if messages[i].get("role") == "assistant" and i > 0 and messages[i - 1].get("role") == "user":
                turn_boundaries.append(i - 1)
                i -= 2
            else:
                turn_boundaries.append(i)
                i -= 1

        turn_boundaries.reverse()

        if len(turn_boundaries) <= keep_turns:
            return 0

        keep_from = turn_boundaries[-keep_turns]
        compress_count = keep_from
        return max(compress_count, 2) if compress_count >= 2 else 0

    # ------------------------------------------------------------------
    # Agent 注册
    # ------------------------------------------------------------------

    async def register_agent(self, agent_id: str) -> None:
        from runtime.workspace import ensure_agent_workspace

        ensure_agent_workspace(agent_id)
        self._init_mem_system(agent_id)
        self._states[agent_id] = AgentState(agent_id=agent_id)


agent_manager = AgentManager()
