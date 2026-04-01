"""Agent 引擎核心 — AgentManager, AgentState, 生命周期, 自动压缩, 命令处理"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    DATA_DIR,
    resolve_agent_config,
    resolve_agent_workspace,
    resolve_agent_dir,
    resolve_mem_config,
    list_agents,
)
from graph.prompt_builder import prompt_builder
from graph.session_manager import session_manager
from graph.run_tracker import run_tracker
from graph.audit_log import audit_logger
from graph.token_counter import count_messages_tokens
from graph.session_pruning import prune_messages
from graph.command_parser import parse_command, execute_command
from graph.tool_call_parser import parse_text_tool_calls, strip_tool_call_patterns
from graph.errors import (
    is_compaction_failure_error,
    is_likely_context_overflow_error,
    is_role_ordering_error,
    is_session_corruption_error,
    is_transient_http_error,
)
from graph.model_selection import (
    resolve_fallback_candidates,
    run_with_fallback_stream,
)
from graph.models_config import ModelRef
from graph.llm_factory import create_llm

logger = logging.getLogger(__name__)

TRANSIENT_HTTP_RETRY_DELAY_MS = 2500

# 裸 /new 或 /reset 后作为首条用户消息注入，触发 Session Startup + 问候
BARE_SESSION_RESET_PROMPT = (
    "A new session was started via /new or /reset. "
    "Greet the user in your configured persona (SOUL.md / IDENTITY.md are already in your system prompt). "
    "Be yourself - use your defined voice, mannerisms, and mood. "
    "Keep it to 1-3 sentences and ask what they want to do. "
    "If the runtime model differs from default_model in the system prompt, mention the default model. "
    "Do not mention internal files, tools, memory status, or reasoning."
)


# ---------------------------------------------------------------------------
# AgentState — 每个 Agent 实例的运行时状态
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    agent_id: str
    compaction_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_turns: int = 0
    think_level: int = 0
    verbose: bool = False
    reasoning: bool = False
    last_active: float = 0.0
    _tools_cache: list | None = field(default=None, repr=False)

    @property
    def thinking(self) -> bool:
        return self.think_level > 0

    def record_turn(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_turns += 1
        import time
        self.last_active = time.time()

    def invalidate_tools(self) -> None:
        self._tools_cache = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，用于持久化"""
        return {
            "agent_id": self.agent_id,
            "compaction_count": self.compaction_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_turns": self.total_turns,
            "think_level": self.think_level,
            "verbose": self.verbose,
            "reasoning": self.reasoning,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """从字典恢复状态"""
        return cls(
            agent_id=data.get("agent_id", ""),
            compaction_count=data.get("compaction_count", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_turns=data.get("total_turns", 0),
            think_level=data.get("think_level", 0),
            verbose=data.get("verbose", False),
            reasoning=data.get("reasoning", False),
            last_active=data.get("last_active", 0.0),
        )

    def save_to_disk(self, path: Path) -> None:
        """保存状态到磁盘"""
        try:
            data = self.to_dict()
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"AgentState saved for {self.agent_id}")
        except Exception as e:
            logger.warning(f"Failed to save AgentState for {self.agent_id}: {e}")

    @classmethod
    def load_from_disk(cls, path: Path, agent_id: str) -> "AgentState":
        """从磁盘加载状态"""
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                state = cls.from_dict(data)
                state.agent_id = agent_id  # 确保 agent_id 正确
                logger.debug(f"AgentState loaded for {agent_id}")
                return state
        except Exception as e:
            logger.warning(f"Failed to load AgentState for {agent_id}: {e}")
        return cls(agent_id=agent_id)


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




# ---------------------------------------------------------------------------
# SSE 事件队列 — 用于前端实时更新
# ---------------------------------------------------------------------------

class EventBus:
    """简易事件总线，支持 SSE 订阅"""

    def __init__(self):
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, agent_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(agent_id, []).append(queue)
        return queue

    def unsubscribe(self, agent_id: str, queue: asyncio.Queue) -> None:
        queues = self._queues.get(agent_id, [])
        if queue in queues:
            queues.remove(queue)

    def emit(self, agent_id: str, event: dict[str, Any]) -> None:
        for queue in self._queues.get(agent_id, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


event_bus = EventBus()


# ---------------------------------------------------------------------------
# AgentManager — 核心引擎
# ---------------------------------------------------------------------------

class AgentManager:
    def __init__(self):
        self.data_dir: str = ""
        # New mem-system singletons (per agent_id)
        self.mem_stores: dict[str, Any] = {}
        self.mem_embedders: dict[str, Any] = {}
        self.mem_workers: dict[str, Any] = {}
        self.mem_recalls: dict[str, Any] = {}
        self._states: dict[str, AgentState] = {}
        self._initialized = False
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self.lifecycle_hooks: LifecycleHooks | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._state_save_tasks: dict[str, asyncio.Task] = {}

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
        """Initialize the new memory system (MemStore / MemEmbedder / MemWorker / MemRecall)."""
        try:
            mem_cfg = resolve_mem_config()
            if not mem_cfg.get("enabled", True):
                logger.info("Mem system disabled for %s", agent_id)
                return

            from mem.store import MemStore
            from mem.embedder import MemEmbedder
            from mem.worker import MemWorker
            from mem.recall import MemRecall
            from mem.task_processor import MemTaskProcessor
            from mem.skill_evolver import MemSkillEvolver

            store = MemStore(
                db_path=mem_cfg["storage"]["db_path"],
                dimensions=mem_cfg.get("embedding", {}).get("dimensions", 1536),
            )
            self.mem_stores[agent_id] = store

            embedder = MemEmbedder.from_config(mem_cfg.get("embedding", {}))
            self.mem_embedders[agent_id] = embedder

            skill_store_dir = str(Path(mem_cfg["storage"]["db_path"]).parent / "skills-store")
            skill_evolver = MemSkillEvolver.from_config(
                mem_cfg, store=store, embedder=embedder,
                skill_store_dir=skill_store_dir,
            )

            async def _on_task_completed(task: Any) -> None:
                await skill_evolver.on_task_completed(task)

            task_proc = MemTaskProcessor.from_config(
                mem_cfg, store=store, embedder=embedder,
                on_task_completed=_on_task_completed,
            )

            async def _on_chunks_ingested(session_key: str, session_end: bool) -> None:
                await task_proc.on_chunks_ingested(session_key, session_end, owner=agent_id)

            worker = MemWorker.from_config(
                mem_cfg, store=store, embedder=embedder,
                on_chunks_ingested=_on_chunks_ingested,
            )
            self.mem_workers[agent_id] = worker

            recall = MemRecall.from_config(mem_cfg, store=store, embedder=embedder)
            self.mem_recalls[agent_id] = recall

            logger.info("Mem system initialized for agent %s", agent_id)
        except Exception as e:
            logger.error("Failed to initialize mem system for %s: %s", agent_id, e)

    async def initialize(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._main_loop = asyncio.get_running_loop()

        from graph.workspace import ensure_agent_workspace

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
                from graph.thinking import resolve_agent_think_default
                think_level = resolve_agent_think_default(agent_id)
                self._states[agent_id] = AgentState(agent_id=agent_id, think_level=think_level.value)

        self._initialized = True

    def get_llm(self, agent_id: str = "main"):
        """获取指定 Agent 的 LLM 实例（per-agent 动态创建，按 Provider 配置路由）"""
        from graph.llm_factory import llm_cache
        from graph.model_selection import resolve_agent_model

        ref = resolve_agent_model(agent_id)
        return llm_cache.get_or_create(agent_id, ref)

    def get_current_model_ref(self, agent_id: str = "main"):
        """获取 Agent 当前使用的 ModelRef"""
        from graph.model_selection import resolve_agent_model
        return resolve_agent_model(agent_id)

    def switch_model(self, agent_id: str, model_raw: str) -> str:
        """运行时切换 Agent 模型，返回新模型描述"""
        from graph.llm_factory import llm_cache
        from graph.model_selection import resolve_agent_model, get_model_display_name
        from graph.models_config import parse_model_ref

        ref = parse_model_ref(model_raw)
        if not ref:
            raise ValueError(f"Invalid model reference: {model_raw}")

        if not ref.provider:
            from graph.models_config import models_config
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
        await self._save_all_states()

        # 取消状态保存任务
        for task in self._state_save_tasks.values():
            task.cancel()
        self._state_save_tasks.clear()

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

    def _build_tools(self, agent_id: str, session_id: str = "") -> list:
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
        tools.extend(get_agent_tools(agent_id, self, session_id, main_loop=self._main_loop))
        tools.extend(get_cron_tools(agent_id))
        tools.extend(get_status_tools(agent_id, session_id))

        tools = self._filter_tools_by_policy(agent_id, tools)
        return tools

    def _filter_tools_by_policy(self, agent_id: str, tools: list) -> list:
        """按 agents.list[].tools.allow/deny 过滤工具"""
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
                    message = BARE_SESSION_RESET_PROMPT
                elif action == "reset_noflush":
                    # /reset：不写入 session-memory 的轻量重置，再注入 BARE_SESSION_RESET_PROMPT 跑一轮问候
                    async for evt in self._handle_reset_noflush(session_id, agent_id):
                        yield evt
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
        from graph.workspace import has_bootstrap
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

        tools = self._build_tools(agent_id, session_id)
        available_tool_names = [t.name for t in tools] if tools else None

        from graph.prompt_builder import PromptParams
        from config import get_config
        _locale = get_config().get("app", {}).get("locale", "zh-CN")
        prompt_params = PromptParams(
            agent_id=agent_id,
            mode=prompt_mode,
            available_tools=available_tool_names,
            extra_system_prompt=extra_prompt or None,
            locale=_locale,
        )
        system_prompt, prompt_report = prompt_builder.build_system_prompt_with_report(prompt_params)
        logger.info(prompt_report.summary())

        history = session_manager.load_session_for_agent(session_id, agent_id)

        # 注入结构化会话摘要（压缩后的上下文续接）
        store = self.mem_stores.get(agent_id)
        if store:
            session_summary = store.get_session_summary(session_id, agent_id)
            if session_summary:
                summary_text = prompt_builder.format_session_summary(session_summary)
                if summary_text:
                    history.insert(0, {"role": "system", "content": summary_text})

        # 会话修剪
        history = prune_messages(history, agent_id=agent_id)

        from graph.context_budget import resolve_budget
        from graph.token_counter import count_tokens, count_messages_tokens
        _budget = resolve_budget(agent_id)
        _sp_tokens = count_tokens(system_prompt)
        _summary_tokens = count_tokens(history[0].get("content", "")) if history and history[0].get("role") == "system" else 0
        _history_tokens = count_messages_tokens(history)

        # 输入预算硬检查：总输入不超过 active_tokens，超出时从 history 头部裁对话消息
        _msg_tokens = count_tokens(message)
        _total_input = _sp_tokens + _history_tokens + _msg_tokens
        if _total_input > _budget.active_tokens:
            while _total_input > _budget.active_tokens and len(history) > 1:
                _trim_idx = 1 if history and history[0].get("role") == "system" else 0
                if _trim_idx >= len(history):
                    break
                _removed = history.pop(_trim_idx)
                _removed_tokens = count_tokens(_removed.get("content", "")) + 4
                for _tc in _removed.get("tool_calls", []):
                    _removed_tokens += count_tokens(str(_tc.get("input", "")))
                    _removed_tokens += count_tokens(str(_tc.get("output", "")))
                _total_input -= _removed_tokens
            _history_tokens = count_messages_tokens(history)
            logger.info("Input budget enforced: trimmed to %d history tokens (active_budget=%d)",
                        _history_tokens, _budget.active_tokens)

        agent_cfg = resolve_agent_config(agent_id)
        recursion_limit = agent_cfg.get("recursion_limit", 50)

        candidates = resolve_fallback_candidates(agent_id)
        did_retry_transient = False
        did_reset_compaction = False

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
            yield {"type": "lifecycle", "event": "turn_start", "run_id": turn.run_id, "model": str(ref)}

            full_response = ""
            tool_calls_log: list[dict[str, Any]] = []
            tool_input_by_run_id: dict[str, Any] = {}
            _streaming_model_run_id: str | None = None
            step_count = 0
            _content_refresh_sent = False

            try:
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
                        # 若 full_response 含文本形式工具调用，首次 tool_start 时刷新前端
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
                        if evt_run_id:
                            tool_input_by_run_id[evt_run_id] = tool_input
                        run_tracker.record_tool_start(turn.run_id, tool_name, tool_input)
                        yield {
                            "type": "tool_start", "tool": tool_name, "input": tool_input,
                            "step": step_count, "max_steps": recursion_limit,
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
                        tool_input_for_log = tool_input if tool_input is not None else ""
                        tool_name = event.get("name", "")
                        run_tracker.record_tool_end(turn.run_id, tool_name, output_str)
                        audit_logger.log_tool_call(
                            agent_id, turn.run_id, tool_name,
                            tool_input_for_log,
                            output_str,
                        )

                        tool_calls_log.append({
                            "tool": tool_name,
                            "input": tool_input_for_log,
                            "output": output_str[:2000],
                        })
                        if self.lifecycle_hooks:
                            await self.lifecycle_hooks.on_after_tool_call(
                                agent_id, turn.run_id, tool_name, tool_input_for_log, output_str
                            )
                        yield {"type": "tool_end", "tool": tool_name, "output": output_str[:2000]}

                        # 危险工具执行后通知前端（用于审计/确认提示）
                        if tool_name in ("exec", "process_kill"):
                            safe_input = str(tool_input_for_log)[:200] if tool_input_for_log else ""
                            event_bus.emit(agent_id, {
                                "type": "lifecycle",
                                "event": "tool_dangerous_executed",
                                "tool": tool_name,
                                "input_preview": safe_input,
                            })

            except Exception as e:
                error_str = str(e)
                is_recursion = "recursion" in error_str.lower() or "GraphRecursionError" in type(e).__name__
                run_tracker.error_turn(turn.run_id, error_str)
                audit_logger.log_turn_error(agent_id, turn.run_id, error_str)
                if is_recursion:
                    yield {
                        "type": "lifecycle", "event": "recursion_limit_reached",
                        "step": step_count, "max_steps": recursion_limit,
                    }
                    yield {
                        "type": "error",
                        "error": f"Agent 达到最大迭代次数 ({recursion_limit})，已自动停止。已执行 {step_count} 步工具调用。",
                    }
                else:
                    yield {"type": "lifecycle", "event": "turn_error", "error": error_str}
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
                        run_tracker.record_tool_start(turn.run_id, fallback_tool_name, args_to_use)
                        logger.info(f"Fallback tool call: {fallback_tool_name}({args_to_use})")
                        yield {
                            "type": "tool_start", "tool": fallback_tool_name, "input": args_to_use,
                            "step": step_count, "max_steps": recursion_limit,
                        }
                        try:
                            result_str = str(matched_tool._run(**args_to_use))[:2000]
                        except Exception as te:
                            from tools.error_utils import format_tool_error
                            result_str = format_tool_error(fallback_tool_name, te)
                        run_tracker.record_tool_end(turn.run_id, fallback_tool_name, result_str)
                        audit_logger.log_tool_call(agent_id, turn.run_id, fallback_tool_name, args_to_use, result_str)
                        yield {"type": "tool_end", "tool": fallback_tool_name, "output": result_str}
                        tool_calls_log.append({
                            "tool": fallback_tool_name,
                            "input": args_to_use,
                            "output": result_str,
                        })
                full_response = strip_tool_call_patterns(full_response)

            # 保存消息（若含文本形式工具调用则保存清理后的 content）
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

            yield {
                "type": "lifecycle",
                "event": "turn_end",
                "run_id": turn.run_id,
                "usage": usage_info,
            }
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
            task = asyncio.create_task(
                self._incremental_ingest(
                    agent_id, session_id, message, done_content,
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
                    yield {
                        "type": "error",
                        "error": "⚠️ 上下文溢出 — 提示过长。请缩短消息或使用更大 context 的模型。",
                    }
                    return

                yield {"type": "lifecycle", "event": "turn_error", "error": msg}
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
        worker = self.mem_workers.get(agent_id)
        if not worker:
            return
        try:
            import uuid as _uuid
            from mem.worker import IngestMessage
            turn_id = str(_uuid.uuid4())
            batch: list[IngestMessage] = []
            if user_content.strip():
                batch.append(IngestMessage(
                    role="user",
                    content=user_content.strip(),
                    session_key=session_id,
                    turn_id=turn_id,
                    owner=agent_id,
                ))
            if assistant_content.strip():
                batch.append(IngestMessage(
                    role="assistant",
                    content=assistant_content.strip(),
                    session_key=session_id,
                    turn_id=turn_id,
                    owner=agent_id,
                ))
            if batch:
                await worker.enqueue(batch, session_end=False)
        except Exception as e:
            logger.warning("incremental_ingest failed for %s: %s", agent_id, e)

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
        worker = self.mem_workers.get(agent_id)
        if not worker or not messages:
            return
        try:
            import uuid as _uuid
            from mem.worker import IngestMessage
            batch: list[IngestMessage] = []
            for msg in messages:
                content = msg.get("content", "").strip()
                if not content:
                    continue
                role = msg.get("role", "user")
                if role == "system":
                    continue
                batch.append(IngestMessage(
                    role=role,
                    content=content,
                    session_key=session_id,
                    turn_id=str(_uuid.uuid4()),
                    owner=agent_id,
                ))
            if batch:
                await worker.enqueue(batch, session_end=session_end)
        except Exception as e:
            logger.warning("batch_ingest failed for %s: %s", agent_id, e)

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

        from graph.token_counter import detect_compaction_level
        level = detect_compaction_level(messages, agent_id=agent_id, overhead_tokens=overhead_tokens)

        if level == "none":
            return

        logger.info("Auto-compaction triggered: level=%s agent=%s session=%s", level, agent_id, session_id)
        audit_logger.log(agent_id, "auto_compact_trigger", {"session_id": session_id, "level": level})
        event_bus.emit(agent_id, {
            "type": "lifecycle",
            "event": "auto_compact_start",
            "session_id": session_id,
            "level": level,
        })
        try:
            await self.compress_session(session_id, agent_id, level=level)
            event_bus.emit(agent_id, {
                "type": "lifecycle",
                "event": "auto_compact_done",
                "session_id": session_id,
            })
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
        event_bus.emit(agent_id, {
            "type": "lifecycle",
            "event": "manual_compact_start",
            "session_id": session_id,
        })

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
                    from graph.token_counter import (
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
                event_bus.emit(agent_id, {
                    "type": "lifecycle",
                    "event": "manual_compact_skipped",
                    "session_id": session_id,
                    "reason": result.get("error"),
                })
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
            event_bus.emit(agent_id, {
                "type": "lifecycle",
                "event": "manual_compact_done",
                "session_id": session_id,
                "data": {
                    "archived_count": c.get("archived_count", 0),
                    "remaining_count": c.get("remaining_count", 0),
                },
            })
            yield {"type": "done", "content": msg, "session_id": session_id}
        except Exception as e:
            event_bus.emit(agent_id, {
                "type": "lifecycle",
                "event": "manual_compact_error",
                "session_id": session_id,
                "error": str(e)[:200],
            })
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
        from graph.retry import retry_async
        from graph.token_counter import count_tokens
        from graph.model_selection import resolve_agent_model, get_model_context_window

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

        from graph.context_budget import resolve_budget
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
        from graph.retry import retry_async
        from graph.token_counter import count_tokens
        from graph.model_selection import resolve_agent_model, get_model_context_window

        from graph.context_budget import resolve_budget
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
        from graph.workspace import ensure_agent_workspace

        ensure_agent_workspace(agent_id)
        self._init_mem_system(agent_id)
        self._states[agent_id] = AgentState(agent_id=agent_id)


agent_manager = AgentManager()
