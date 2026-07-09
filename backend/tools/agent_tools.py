"""Agent/Session tools: agents_list, sessions_list,
sessions_history, sessions_send, sessions_spawn, subagents"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import (
    list_agents,
    resolve_agent_config,
    get_config,
)

ANNOUNCE_ACQUIRE_TIMEOUT_SEC = 10
ANNOUNCE_RUN_TIMEOUT_SEC = 30

# #region agent log
def _debug_log(location: str, message: str, data: dict[str, Any], hypothesis_id: str) -> None:
    try:
        import json as _json
        import time as _time
        payload = {
            "sessionId": "e404f0",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        log_path = Path(__file__).resolve().parents[2] / ".cursor" / "debug-e404f0.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion


# ---------------------------------------------------------------------------
# agents_list
# ---------------------------------------------------------------------------

class AgentsListTool(BaseTool):
    name: str = "agents_list"
    description: str = (
        "列出当前 Agent 允许协作的 Agent（ID、名称、描述）。"
        "结果根据配置中的 subagents.allow_agents 过滤。"
    )
    current_agent_id: str = "main"

    def _run(self, **kwargs) -> str:
        """Only list agents allowed for collaboration to prevent arbitrary agent calls.

        Rules:
        - Always include self (current_agent_id)
        - If subagents.allow_agents contains \"*\", show all configured agents
        - Otherwise, only show agents in the allow_agents list
        """
        agents = list_agents()
        if not agents:
            return "No agents configured."

        from config import resolve_agent_config

        requester_id = self.current_agent_id or "main"
        cfg = resolve_agent_config(requester_id) or {}
        subagents_cfg = cfg.get("subagents") or {}
        allow = subagents_cfg.get("allow_agents") or []

        # 归一化允许列表
        allow_any = "*" in allow
        allow_set = {a for a in allow if a and a != "*"}

        visible: list[dict[str, Any]] = []
        for a in agents:
            aid = a.get("id")
            if not aid:
                continue
            if aid == requester_id:
                visible.append(a)
                continue
            if allow_any or aid in allow_set:
                visible.append(a)

        if not visible:
            return "The current agent is not configured to collaborate with any other agents."

        lines = []
        for a in visible:
            lines.append(f"- {a['id']}: {a.get('name', 'Unnamed')} — {a.get('description', '')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sessions_list
# ---------------------------------------------------------------------------

class SessionsListInput(BaseModel):
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    spawned_by: str = Field(
        default="",
        description="按此 session_key 过滤由该会话创建的子会话（可选）",
    )


class SessionsListTool(BaseTool):
    name: str = "sessions_list"
    description: str = "列出指定 Agent 的所有会话。spawned_by 可过滤由特定会话创建的子 Agent 会话。"
    args_schema: type[BaseModel] = SessionsListInput
    current_agent_id: str = "main"
    current_session_id: str = ""

    def _run(self, agent_id: str = "", spawned_by: str = "") -> str:
        target_id = agent_id or self.current_agent_id
        from sessions.session_manager import session_manager

        spawned_by_key: str | None = None
        if spawned_by and (spawned_by or "").strip():
            sk = (spawned_by or "").strip()
            if sk.startswith("agent:") and ":" in sk[6:]:
                spawned_by_key = sk
            else:
                spawned_by_key = session_manager.session_key_from_session_id(
                    target_id, sk
                )
        sessions = session_manager.list_sessions(target_id, spawned_by_session_key=spawned_by_key)
        if not sessions:
            return f"No sessions found for Agent '{target_id}'."
        lines = []
        for s in sessions:
            title = s.get("title", "No Title")
            sid = s.get("session_id", "?")
            msg_count = s.get("message_count", 0)
            lines.append(f"- {sid}: {title} ({msg_count} messages)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sessions_history
# ---------------------------------------------------------------------------

class SessionsHistoryInput(BaseModel):
    session_id: str = Field(default="", description="会话 ID。为空则使用当前会话（来自 sessions_list）")
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    limit: int = Field(default=20, description="最多获取的消息条数")


class SessionsHistoryTool(BaseTool):
    name: str = "sessions_history"
    description: str = "获取指定会话的消息历史。session_id 为空则用当前会话；否则使用 sessions_list 中的 session_id。"
    args_schema: type[BaseModel] = SessionsHistoryInput
    current_agent_id: str = "main"
    current_session_id: str = ""

    def _run(self, session_id: str = "", agent_id: str = "", limit: int = 20) -> str:
        target_id = agent_id or self.current_agent_id
        from sessions.session_manager import session_manager
        effective_sid = (session_id or "").strip() or self.current_session_id
        if not effective_sid:
            effective_sid = session_manager.resolve_main_session_id(target_id)
        data = session_manager.load_session(effective_sid, target_id)
        if data is None:
            return f"Session '{effective_sid}' does not exist."

        messages = data.get("messages", [])[-limit:]
        if not messages:
            return "Session has no messages."

        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sessions_send
# ---------------------------------------------------------------------------

class SessionsSendInput(BaseModel):
    session_id: str = Field(default="", description="目标会话 ID。为空则使用当前会话（来自 sessions_list）")
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    message: str = Field(description="要发送的消息")


class SessionsSendTool(BaseTool):
    name: str = "sessions_send"
    description: str = "向另一会话发送消息。session_id 为空则用当前会话；否则使用 sessions_list 中的 session_id。"
    args_schema: type[BaseModel] = SessionsSendInput
    current_agent_id: str = "main"
    current_session_id: str = ""

    def _run(self, session_id: str = "", message: str = "", agent_id: str = "") -> str:
        target_id = agent_id or self.current_agent_id
        from sessions.session_manager import session_manager
        effective_sid = (session_id or "").strip() or self.current_session_id
        if not effective_sid:
            effective_sid = session_manager.resolve_main_session_id(target_id)
        session_manager.save_message(effective_sid, target_id, "user", message)
        return f"Message sent to agent:{target_id}:{effective_sid}"


# ---------------------------------------------------------------------------
# sessions_spawn
# ---------------------------------------------------------------------------

class SessionsSpawnInput(BaseModel):
    task: str = Field(description="子 Agent 要执行的任务描述")
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    label: str | None = Field(default=None, description="子 Agent 标签（可选）")
    model: str | None = Field(default=None, description="模型覆盖（可选）")


class SessionsSpawnTool(BaseTool):
    name: str = "sessions_spawn"
    description: str = (
        "在后台启动一个独立子 Agent 并行执行任务，不阻塞当前对话。"
        "子 Agent 拥有独立上下文和工具，完成后自动将结果通知你。"
        "适用场景：耗时操作（网络搜索、大量文件处理）、可并行的独立子任务、"
        "需要独立上下文的调研工作。可同时 spawn 多个子 Agent 并行执行。"
        "不适用：简单问答、需要与用户多轮交互的任务。"
    )
    args_schema: type[BaseModel] = SessionsSpawnInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    _agent_manager: Any = None

    _FAILURE_HINTS = (
        "failure",
        "error",
        "exception",
        "timeout",
        "not found",
        "return none",
        "no results",
        "cannot",
        "failed",
        "无结果",
        "无法",
        "失败",
    )

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use _arun for async execution")

    async def _arun(
        self,
        task: str,
        agent_id: str = "",
        label: str | None = None,
        model: str | None = None,
    ) -> str:
        target_id = agent_id or self.current_agent_id

        # 基于 subagents.allow_agents 做目标 Agent 允许性校验
        requester_id = self.current_agent_id or "main"
        requester_cfg = resolve_agent_config(requester_id) or {}
        subagents_cfg = requester_cfg.get("subagents") or {}
        allow = subagents_cfg.get("allow_agents") or []

        allow_any = "*" in allow
        allow_set = {a for a in allow if a and a != "*"}

        if not allow_any and target_id != requester_id and target_id not in allow_set:
            return (
                f"Error: Current agent is not allowed to spawn tasks for '{target_id}'. "
                f"Please explicitly add this agent to agents.list[].subagents.allow_agents in the configuration."
            )
        from sessions.session_manager import session_manager
        from subagents.subagent_registry import registry

        requester_key = session_manager.session_key_from_session_id(
            self.current_agent_id,
            self.current_session_id or session_manager.resolve_main_session_id(self.current_agent_id),
        )
        requester_depth = registry.get_requester_depth(requester_key)
        child_depth = requester_depth + 1
        cfg = resolve_agent_config(target_id)
        subagent_cfg = cfg.get("subagents", {})
        max_spawn_depth = subagent_cfg.get("max_spawn_depth", 1)
        max_children = subagent_cfg.get("max_children_per_agent", 5)

        if child_depth > max_spawn_depth:
            return (
                f"Error: Current depth limit reached (maxSpawnDepth={max_spawn_depth}), "
                f"cannot spawn more sub-agents."
            )

        child_session_id = f"subagent-{uuid.uuid4().hex[:12]}"
        child_session_key = f"agent:{target_id}:subagent:{child_session_id}"
        run_id = uuid.uuid4().hex[:12]

        active = registry.count_active_for_requester(requester_key)
        if active >= max_children:
            return f"Error: Active sub-agents limit reached ({max_children})"

        registry.register_run(
            run_id=run_id,
            child_session_key=child_session_key,
            requester_session_key=requester_key,
            requester_agent_id=self.current_agent_id,
            target_agent_id=target_id,
            task=task,
            label=label,
            model=model,
            spawn_depth=child_depth,
        )

        session_manager.ensure_session(
            child_session_id,
            target_id,
            spawned_by=requester_key,
            label=(label or task[:60] or "Sub-agent task"),
        )

        if self._agent_manager:
            try:
                raw = subagent_cfg.get("run_timeout_seconds", 0)
                run_timeout_seconds = int(raw) if raw is not None else 0
                if run_timeout_seconds < 0:
                    run_timeout_seconds = 0
                coro = self._run_subagent(
                    run_id,
                    child_session_id,
                    target_id,
                    task,
                    requester_key,
                    run_timeout_seconds=run_timeout_seconds,
                )
                t = asyncio.create_task(coro)
                registry.set_task(run_id, t)
            except Exception as e:
                return f"Failed to start sub-agent: {e}"

        return (
            f"Sub-agent spawned:\n"
            f"  run_id: {run_id}\n"
            f"  session_key: {child_session_key}\n"
            f"  Task: {task}"
        )

    def _looks_like_failure_output(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(h in normalized for h in self._FAILURE_HINTS)

    def _collect_latest_subagent_output(
        self,
        session_id: str,
        agent_id: str,
        streamed_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        """优先使用流式文本；若为空则回读会话最后一条 assistant 消息与工具输出。"""
        from sessions.session_manager import session_manager

        if (streamed_text or "").strip():
            return streamed_text.strip(), self._looks_like_failure_output(streamed_text)

        data = session_manager.load_session(session_id, agent_id) or {}
        messages = data.get("messages", []) if isinstance(data, dict) else []
        latest_assistant: dict[str, Any] | None = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                latest_assistant = msg
                break

        content = (latest_assistant or {}).get("content", "") if latest_assistant else ""
        tool_calls_payload = (latest_assistant or {}).get("tool_calls", []) if latest_assistant else []
        merged_tool_calls = [*tool_calls, *tool_calls_payload]

        snippets: list[str] = []
        failure_count = 0
        for tc in merged_tool_calls:
            tool = str(tc.get("tool", "")).strip() or "tool"
            output = str(tc.get("output", "")).strip()
            if not output:
                continue
            if self._looks_like_failure_output(output):
                failure_count += 1
            snippets.append(f"[{tool}] {output[:280]}")

        parts = []
        if (content or "").strip():
            parts.append(content.strip())
        if snippets:
            parts.append("Summary of tool outputs:\n" + "\n".join(snippets[:4]))

        merged = "\n\n".join(parts).strip()
        has_failure = failure_count > 0 and failure_count >= max(1, len(snippets))
        return merged, has_failure

    async def _run_subagent(
        self,
        run_id: str,
        session_id: str,
        agent_id: str,
        task: str,
        requester_key: str,
        run_timeout_seconds: int = 0,
    ) -> None:
        from subagents.subagent_registry import registry
        from infra.event_bus import Events, event_bus

        started_at: float | None = None
        result_parts: list[str] = []
        tool_calls_log: list[dict[str, Any]] = []
        child_session_key = f"agent:{agent_id}:subagent:{session_id}"
        try:
            registry.mark_started(run_id)
            started_at = __import__("time").time()
            event_bus.emit(
                self.current_agent_id,
                Events.subagent_start(run_id=run_id, agent_id=agent_id, task=task[:200]),
            )

            async def _stream_child() -> None:
                import time as _time
                last_progress_emit = 0.0
                async for event in self._agent_manager.astream(
                    message=task,
                    session_id=session_id,
                    agent_id=agent_id,
                    prompt_mode="minimal",
                ):
                    if event.get("type") == "token":
                        token = event.get("content", "") or ""
                        result_parts.append(token)
                        now_ts = _time.time()
                        if now_ts - last_progress_emit >= 1.0:
                            last_progress_emit = now_ts
                            event_bus.emit(
                                self.current_agent_id,
                                Events.subagent_progress(
                                    run_id=run_id,
                                    chars=len("".join(result_parts)),
                                    elapsed_s=int(now_ts - (started_at or now_ts)),
                                ),
                            )
                    elif event.get("type") == "tool_start":
                        event_bus.emit(
                            self.current_agent_id,
                            Events.subagent_tool(run_id=run_id, tool=event.get("tool", "")),
                        )
                    elif event.get("type") == "tool_end":
                        output = event.get("output", "") or ""
                        tool_calls_log.append({
                            "tool": event.get("tool", ""),
                            "output": output,
                        })
                        event_bus.emit(
                            self.current_agent_id,
                            Events.subagent_tool_end(
                                run_id=run_id,
                                tool=event.get("tool", ""),
                                output_preview=str(output)[:160],
                            ),
                        )

            if run_timeout_seconds > 0:
                await asyncio.wait_for(_stream_child(), timeout=run_timeout_seconds)
            else:
                await _stream_child()

            streamed = "".join(result_parts).strip()
            result, all_failed = self._collect_latest_subagent_output(
                session_id=session_id,
                agent_id=agent_id,
                streamed_text=streamed,
                tool_calls=tool_calls_log,
            )
            ended_at = __import__("time").time()
            outcome_key = "completed-empty" if not result else "completed-with-errors" if all_failed else "completed"
            registry.mark_completed(
                run_id,
                result,
                outcome=outcome_key,
                terminal_reason="all-tools-failed" if all_failed else None,
            )
            # #region agent log
            _debug_log(
                "backend/tools/agent_tools.py:_run_subagent",
                "subagent_done_emit_pre_announce",
                {"run_id": run_id, "requester_key": requester_key},
                "H2B",
            )
            # #endregion
            event_bus.emit(
                self.current_agent_id,
                Events.subagent_done(run_id=run_id, result=result[:300]),
            )

            record = registry.get_run(run_id)
            label = record.label if record else None
            announce_outcome = "completed successfully"
            if outcome_key == "completed-empty":
                announce_outcome = "completed with empty output"
            elif outcome_key == "completed-with-errors":
                announce_outcome = "completed with tool errors"
            # #region agent log
            _debug_log(
                "backend/tools/agent_tools.py:_run_subagent",
                "deliver_announce_call",
                {"run_id": run_id, "requester_key": requester_key},
                "H2B",
            )
            # #endregion
            from subagents.subagent_delivery import subagent_announce_delivery

            await subagent_announce_delivery.deliver_to_requester(
                requester_key=requester_key,
                child_session_key=child_session_key,
                run_id=run_id,
                task=task,
                result=result,
                outcome=announce_outcome,
                label=label,
                started_at=started_at,
                ended_at=ended_at,
                debug_log=_debug_log,
            )
        except asyncio.TimeoutError:
            timeout_secs = run_timeout_seconds
            partial_stream = "".join(result_parts).strip()
            result, _ = self._collect_latest_subagent_output(
                session_id=session_id,
                agent_id=agent_id,
                streamed_text=partial_stream,
                tool_calls=tool_calls_log,
            )
            fallback_result = result or f"Sub-agent execution timed out ({timeout_secs}s)"
            registry.mark_terminated(run_id, "timeout")
            event_bus.emit(
                self.current_agent_id,
                Events.subagent_error(run_id=run_id, error=f"timeout after {timeout_secs}s"),
            )
            record = registry.get_run(run_id)
            label = record.label if record else None
            from subagents.subagent_delivery import subagent_announce_delivery

            await subagent_announce_delivery.deliver_to_requester(
                requester_key=requester_key,
                child_session_key=child_session_key,
                run_id=run_id,
                task=task,
                result=fallback_result,
                outcome="timed out",
                label=label,
                started_at=started_at,
                ended_at=__import__("time").time(),
                debug_log=_debug_log,
            )
        except asyncio.CancelledError:
            registry.mark_terminated(run_id, "killed")
            event_bus.emit(self.current_agent_id, Events.subagent_killed(run_id=run_id))
        except Exception as e:
            registry.mark_terminated(run_id, f"error: {e}")
            event_bus.emit(
                self.current_agent_id,
                Events.subagent_error(run_id=run_id, error=str(e)[:200]),
            )


# ---------------------------------------------------------------------------
# subagents
# ---------------------------------------------------------------------------

class SubagentsInput(BaseModel):
    action: Literal["list", "kill", "steer"] = Field(description="操作类型")
    target: str | None = Field(default=None, description="目标 run_id（kill/steer 必填，或填 'all'）")
    message: str | None = Field(default=None, description="新指令（steer 时必填）")
    recent_minutes: int | None = Field(
        default=None,
        description="仅列出最近 N 分钟内完成的子 Agent（默认 30）",
    )


class SubagentsTool(BaseTool):
    name: str = "subagents"
    description: str = (
        "管理子 Agent。操作：list（列出所有子 Agent）、"
        "kill（终止子 Agent，target='all' 表示全部）、"
        "steer（向子 Agent 发送新指令并立即执行，中断当前任务）。"
    )
    args_schema: type[BaseModel] = SubagentsInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    _agent_manager: Any = None
    _spawn_tool: Any = None

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use _arun for async execution")

    async def _arun(
        self,
        action: str,
        target: str | None = None,
        message: str | None = None,
        recent_minutes: int | None = None,
    ) -> str:
        from config import get_config
        from sessions.session_manager import session_manager
        from subagents.subagent_registry import registry

        requester_key = session_manager.session_key_from_session_id(
            self.current_agent_id,
            self.current_session_id or session_manager.resolve_main_session_id(self.current_agent_id),
        )

        if action == "list":
            cfg = get_config()
            default_recent = (
                cfg.get("agents", {}).get("defaults", {}).get("subagents", {}).get("recent_minutes")
            )
            default_recent = default_recent if isinstance(default_recent, (int, float)) else 30
            default_recent = max(1, min(24 * 60, int(default_recent)))
            minutes = recent_minutes if recent_minutes is not None and recent_minutes > 0 else default_recent
            minutes = max(1, min(24 * 60, minutes))
            runs = registry.list_runs_for_requester(requester_key, include_recent_minutes=minutes)
            if not runs:
                return "No sub-agents found."
            lines = []
            for r in runs:
                status = "Running" if r.ended_at is None else f"Completed({r.outcome})"
                elapsed = ""
                if r.started_at and not r.ended_at:
                    import time
                    elapsed = f" {int(time.time() - r.started_at)}s"
                lines.append(
                    f"- [{r.run_id}] {r.label or 'No Label'} | "
                    f"agent:{r.target_agent_id} | {status}{elapsed}\n"
                    f"  Task: {r.task[:100]}"
                )
            return "\n".join(lines)

        elif action == "kill":
            if not target:
                return "Error: kill action requires target parameter (run_id or 'all')"
            if target in ("all", "*"):
                runs = registry.list_runs_for_requester(requester_key)
                killed = 0
                for r in runs:
                    if r.ended_at is None:
                        registry.kill(r.run_id)
                        killed += 1
                return f"Terminated {killed} sub-agent(s)."
            else:
                registry.kill(target)
                return f"Terminated sub-agent: {target}"

        elif action == "steer":
            if not target or not message:
                return "Error: steer action requires target (run_id) and message parameters"
            MAX_STEER_MESSAGE_CHARS = 4000
            if len(message) > MAX_STEER_MESSAGE_CHARS:
                return f"Error: steer message is too long ({len(message)} chars, limit {MAX_STEER_MESSAGE_CHARS})."
            entry = registry.get_run(target)
            if not entry:
                return f"Error: Sub-agent run_id={target} not found"
            if entry.ended_at is not None:
                return f"Sub-agent {target} has already ended, no need to steer."
            # 禁止子 Agent steer 自身
            if requester_key == entry.child_session_key:
                return "Error: Sub-agent cannot steer itself."
            parsed = session_manager.session_id_from_session_key(entry.child_session_key)
            if not parsed:
                return f"Error: Unable to parse child_session_key: {entry.child_session_key}"
            target_agent_id, target_session_id = parsed
            session_manager.save_message(target_session_id, target_agent_id, "user", message)
            registry.kill(target)
            new_run_id = uuid.uuid4().hex[:12]
            new_record = registry.replace_run_after_steer(
                previous_run_id=target,
                next_run_id=new_run_id,
                task=message,
                fallback=entry,
            )
            if not new_record:
                return "Steer failed: unable to replace run."
            if self._spawn_tool and self._agent_manager:
                try:
                    coro = self._spawn_tool._run_subagent(
                        new_run_id, target_session_id, target_agent_id, message, requester_key
                    )
                    t = asyncio.create_task(coro)
                    registry.set_task(new_run_id, t)
                except Exception as e:
                    return f"New instruction saved, but failed to start new run: {e}"
            label = entry.label or entry.task[:50] or "No Label"
            return f"Steered sub-agent [{label}]: new instruction sent and execution started (run_id={new_run_id})."

        return f"Unknown action: {action}"


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def get_agent_tools(
    agent_id: str,
    agent_manager: Any = None,
    session_id: str = "",
) -> list[BaseTool]:
    spawn_tool = SessionsSpawnTool(
        current_agent_id=agent_id,
        current_session_id=session_id,
    )
    spawn_tool._agent_manager = agent_manager

    # 注入 agentSessionKey/current_session_id，工具从上下文获取当前会话
    effective_session_id = session_id or ""
    if not effective_session_id:
        from sessions.session_manager import session_manager
        effective_session_id = session_manager.resolve_main_session_id(agent_id)
    subagents_tool = SubagentsTool(current_agent_id=agent_id, current_session_id=effective_session_id)
    subagents_tool._agent_manager = agent_manager
    subagents_tool._spawn_tool = spawn_tool
    return [
        AgentsListTool(current_agent_id=agent_id),
        SessionsListTool(current_agent_id=agent_id),
        SessionsHistoryTool(current_agent_id=agent_id, current_session_id=effective_session_id),
        SessionsSendTool(current_agent_id=agent_id, current_session_id=effective_session_id),
        spawn_tool,
        subagents_tool,
    ]
