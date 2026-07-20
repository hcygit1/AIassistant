"""Subagent announce delivery.

负责将子 agent 结果包装为 announce，并投递回 requester 会话。
"""

from __future__ import annotations

from typing import Any

from sessions.session_identity import session_key_from_session_id
from sessions.session_work_policy import deliver_system_work

from subagents.subagent_registry import SubagentRunRecord


class _SubagentDeliveryState:
    def __init__(
        self,
        *,
        registry: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus

    @property
    def registry(self) -> Any:
        if self._registry is not None:
            return self._registry
        from subagents.subagent_registry import registry

        return registry

    @property
    def event_bus(self) -> Any:
        if self._event_bus is not None:
            return self._event_bus
        from infra.event_bus import event_bus

        return event_bus

    def emit(self, agent_id: str, run_id: str, result_delivery_state: str) -> None:
        from infra.event_bus import Events

        self.event_bus.emit(
            agent_id,
            Events.subagent_announce(
                run_id=run_id,
                result_delivery_state=result_delivery_state,
            ),
        )

    def queued(self, agent_id: str, run_id: str) -> None:
        self.registry.set_result_delivery_state(run_id, "queued")
        self.emit(agent_id, run_id, "queued")

    def delivered(self, agent_id: str, run_id: str) -> None:
        self.registry.mark_result_delivery_delivered(run_id)
        self.emit(agent_id, run_id, "delivered")

    def dropped(self, agent_id: str, run_id: str) -> None:
        self.registry.mark_result_delivery_dropped(run_id)
        self.emit(agent_id, run_id, "dropped")

    def bind_work(self, run_id: str, work_id: str) -> None:
        self.registry.set_delivery_work_id(run_id, work_id)


class SubagentAnnounceDelivery:
    def __init__(
        self,
        *,
        session_manager: Any | None = None,
        work_delivery: Any | None = None,
        registry: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._work_delivery = work_delivery
        self._state = _SubagentDeliveryState(
            registry=registry,
            event_bus=event_bus,
        )

    @property
    def session_manager(self) -> Any:
        if self._session_manager is not None:
            return self._session_manager
        from sessions.session_manager import session_manager

        return session_manager

    @property
    def work_delivery(self) -> Any:
        if self._work_delivery is not None:
            return self._work_delivery
        from sessions.session_work_delivery import session_work_delivery

        return session_work_delivery

    @property
    def registry(self) -> Any:
        return self._state.registry

    @property
    def event_bus(self) -> Any:
        return self._state.event_bus

    def _save_announce_and_mark_dropped(
        self,
        session_id: str,
        agent_id: str,
        run_id: str,
        announce_msg: str,
    ) -> None:
        self.session_manager.save_message(
            session_id,
            agent_id,
            "system",
            announce_msg,
        )
        self._state.dropped(agent_id, run_id)

    def _emit_subagent_done(self, agent_id: str, run_id: str, result_preview: str) -> None:
        from infra.event_bus import Events

        self.event_bus.emit(
            agent_id,
            Events.subagent_done(run_id=run_id, result=result_preview),
        )

    def parse_requester_key(self, requester_key: str) -> tuple[str, str] | None:
        """requester_key (session_key) -> (agent_id, session_id)."""
        return self.session_manager.session_id_from_session_key(requester_key)

    def build_announce_message(
        self,
        run_id: str,
        task: str,
        result: str,
        outcome: str = "completed successfully",
        label: str | None = None,
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> str:
        """构建 announce 消息（支持 i18n）。"""
        import time as _time

        from config import get_config

        locale = get_config().get("app", {}).get("locale", "zh-CN")

        task_label = label or task[:50] or "task"
        findings = (result or ("(无输出)" if locale == "zh-CN" else "(no output)"))[:500]
        end = ended_at or _time.time()
        start = started_at or end
        runtime_s = int(end - start) if start else 0

        outcome_map = {
            "completed successfully": {"zh": "成功完成", "en": "completed successfully"},
            "completed with empty output": {"zh": "完成但无输出", "en": "completed with empty output"},
            "completed with tool errors": {"zh": "完成但工具执行出错", "en": "completed with tool errors"},
            "timed out": {"zh": "执行超时", "en": "timed out"},
            "error": {"zh": "执行出错", "en": "error"},
        }
        res_outcome = outcome_map.get(outcome, {"zh": outcome, "en": outcome})
        outcome_text = res_outcome.get(
            locale if locale in ("zh", "en", "zh-CN", "en-US") else "en",
            res_outcome["en"],
        )
        if locale in ("zh-CN", "zh"):
            lines = [
                f"[系统消息] [会话ID: {run_id}] 子任务 \"{task_label}\" {outcome_text}。",
                "",
                "结果:",
                findings,
                "",
                f"统计: 运行耗时 {runtime_s}秒",
            ]
        else:
            lines = [
                f"[System Message] [sessionId: {run_id}] A subagent task \"{task_label}\" just {outcome_text}.",
                "",
                "Result:",
                findings,
                "",
                f"Stats: runtime {runtime_s}s",
            ]
        return "\n".join(lines)

    async def deliver_to_requester(
        self,
        requester_key: str,
        child_session_key: str,
        run_id: str,
        task: str,
        result: str,
        outcome: str = "completed successfully",
        label: str | None = None,
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> None:
        """向 requester 交付 announce；若 requester 是子会话则触发其新 run 并递归向上。"""
        parsed = self.parse_requester_key(requester_key)
        if not parsed:
            return
        req_agent, req_session = parsed
        main_session_id = self.session_manager.resolve_main_session_id(req_agent)

        announce_msg = self.build_announce_message(
            run_id=run_id,
            task=task,
            result=result,
            outcome=outcome,
            label=label,
            started_at=started_at,
            ended_at=ended_at,
        )

        is_main = req_session == main_session_id
        if is_main:
            self._state.queued(req_agent, run_id)

            deliver_system_work(
                self.work_delivery,
                kind="announce",
                content=announce_msg,
                agent_id=req_agent,
                session_id=main_session_id,
                run_id=run_id,
                on_record_created=lambda record: self._state.bind_work(run_id, record.id),
                on_success=lambda: self._state.delivered(req_agent, run_id),
                on_cancel=lambda: self._state.dropped(req_agent, run_id),
                on_failure=lambda: self._save_announce_and_mark_dropped(
                    main_session_id,
                    req_agent,
                    run_id,
                    announce_msg,
                ),
            )
            return

        self._state.queued(req_agent, run_id)

        parent_child_key = session_key_from_session_id(
            req_agent,
            req_session,
        )

        async def _sub_session_announce_result(parent_reply: str) -> None:
            grandparent = self.registry.resolve_requester_for_child_session(
                parent_child_key
            )
            if grandparent:
                g_req_key, _ = grandparent
                await self.deliver_to_requester(
                    requester_key=g_req_key,
                    child_session_key=parent_child_key,
                    run_id=run_id,
                    task=task,
                    result=parent_reply or result,
                    outcome=outcome,
                    label=label,
                    started_at=started_at,
                    ended_at=ended_at,
                )

        async def _sub_session_announce_fail(exc: Exception) -> None:
            self.session_manager.save_message(
                req_session,
                req_agent,
                "system",
                f"[Announce processing failed] {str(exc)[:200]}",
            )
            self._state.dropped(req_agent, run_id)
            grandparent = self.registry.resolve_requester_for_child_session(
                parent_child_key
            )
            if grandparent:
                g_req_key, _ = grandparent
                await self.deliver_to_requester(
                    requester_key=g_req_key,
                    child_session_key=parent_child_key,
                    run_id=run_id,
                    task=task,
                    result=f"Sub-agent aggregation failed: {exc}",
                    outcome="error",
                    label=label,
                    started_at=started_at,
                    ended_at=ended_at,
                )

        deliver_system_work(
            self.work_delivery,
            kind="announce",
            content=announce_msg,
            agent_id=req_agent,
            session_id=req_session,
            run_id=run_id,
            on_record_created=lambda record: self._state.bind_work(run_id, record.id),
            result_handler=_sub_session_announce_result,
            on_success=lambda: self._state.delivered(req_agent, run_id),
            on_cancel=lambda: self._state.dropped(req_agent, run_id),
            on_failure_async=_sub_session_announce_fail,
        )

    async def deliver_recovered_run(self, run_id: str, entry: SubagentRunRecord) -> bool:
        """启动恢复阶段的 announce 投递。

        与实时子 agent announce 不同，这里不做递归 requester 传播，
        而是保持当前 resume 逻辑：直接向 target requester 会话投递一条 announce。
        """
        parsed = self.parse_requester_key(entry.requester_session_key)
        if not parsed:
            return False
        req_agent, req_session = parsed
        main_sid = self.session_manager.resolve_main_session_id(req_agent)

        announce_msg = self.build_announce_message(
            run_id=run_id,
            task=entry.task,
            result=(entry.result_summary or "(no output)")[:500],
            outcome=entry.outcome or "completed",
            label=entry.label,
            started_at=entry.started_at,
            ended_at=entry.ended_at,
        )

        result_preview = (entry.result_summary or "(no output)")[:300]
        target_sid = main_sid if req_session == main_sid else req_session
        deliver_system_work(
            self.work_delivery,
            kind="announce",
            content=announce_msg,
            agent_id=req_agent,
            session_id=target_sid,
            run_id=run_id,
            on_record_created=lambda record: self._state.bind_work(run_id, record.id),
            on_success=lambda: (
                self._state.delivered(req_agent, run_id),
                self._emit_subagent_done(req_agent, run_id, result_preview),
            ),
            on_cancel=lambda: self._state.dropped(req_agent, run_id),
            on_failure=lambda: (
                self.session_manager.save_message(
                    target_sid,
                    req_agent,
                    "system",
                    announce_msg,
                ),
                self._state.dropped(req_agent, run_id),
                self._emit_subagent_done(req_agent, run_id, result_preview),
            ),
        )
        return True


subagent_announce_delivery = SubagentAnnounceDelivery()
