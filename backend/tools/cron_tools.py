"""Cron 工具 — Agent 在对话中管理定时任务与提醒"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CronToolInput(BaseModel):
    action: Literal["list", "add", "update", "remove", "run", "wake"] = Field(
        ...,
        description="操作类型：list 列出任务，add 创建，update 修改，remove 删除，run 立即执行，wake 立即发送提醒",
    )
    job_id: str | None = Field(default=None, description="任务 ID（update/remove/run 时必填）")
    name: str | None = Field(default=None, description="任务名称（add 时必填）")
    description: str | None = Field(default=None, description="任务描述")
    schedule: dict | None = Field(
        default=None,
        description="调度配置：{kind: at|every|cron, at?: ISO时间, everyMs?: 间隔毫秒, expr?: cron表达式, tz?: 时区}",
    )
    payload: dict | None = Field(
        default=None,
        description="Payload：{text: 提醒内容}",
    )
    text: str | None = Field(default=None, description="wake 时使用的提醒内容")


class CronTool(BaseTool):
    name: str = "cron"
    description: str = (
        "管理定时任务与提醒。action: list 列出任务；add 创建（需 name、schedule、payload）；"
        "update 修改（需 job_id）；remove 删除（需 job_id）；run 立即执行（需 job_id）；"
        "wake 立即发送提醒到主会话（需 text）。"
    )
    args_schema: type[BaseModel] = CronToolInput
    current_agent_id: str = "main"
    _cron_service: Any = None

    def _run(
        self,
        action: str,
        job_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        schedule: dict | None = None,
        payload: dict | None = None,
        text: str | None = None,
    ) -> str:
        service = self._cron_service
        if service is None:
            from scheduler.cron_service import cron_service

            service = cron_service
        if not service.is_enabled() and action != "list":
            return "cron 调度器当前处于禁用状态（config.cron.enabled=false）。请先启用后再执行该操作。"
        from scheduler.cron_service import CronServiceError

        if action == "list":
            jobs = service.list_jobs(
                agent_id=self.current_agent_id or "main"
            )
            if not jobs:
                return "暂无定时任务。"
            lines = []
            for j in jobs:
                s = j.schedule
                sched_str = ""
                if s.kind == "at":
                    sched_str = f"at {s.at}"
                elif s.kind == "every":
                    sched_str = f"every {s.every_ms}ms"
                else:
                    sched_str = f"cron {s.expr or ''}"
                lines.append(
                    f"- {j.id}: {j.name} ({sched_str}) "
                    f"[{'启用' if j.enabled else '禁用'}] {j.payload.text[:50]}..."
                )
            return "\n".join(lines)

        if action == "wake":
            if not (text or "").strip():
                return "wake 需要提供 text 参数。"
            agent_id = self.current_agent_id or "main"
            try:
                service.wake(
                    agent_id=agent_id,
                    text=(text or "").strip(),
                )
            except CronServiceError as exc:
                return f"发送提醒失败：{exc}"
            return f"已发送提醒到主会话，内容：{(text or '').strip()[:100]}..."

        if action == "add":
            if not (name or "").strip():
                return "add 需要提供 name 参数。"
            s = schedule or {}
            if not s.get("kind"):
                return "add 需要提供 schedule，含 kind (at|every|cron)。"
            p = payload or {}
            payload_text = str(p.get("text", "")).strip()
            if not payload_text:
                return "add 需要提供 payload.text（提醒内容）。"

            try:
                job = service.create_job(
                    name=(name or "").strip(),
                    description=(description or "").strip(),
                    agent_id=self.current_agent_id or "main",
                    enabled=True,
                    delete_after_run=s.get("kind") == "at",
                    schedule=s,
                    payload={
                        "kind": "systemEvent",
                        "text": payload_text,
                    },
                )
            except CronServiceError as exc:
                return f"创建任务失败：{exc}"
            return f"已创建任务 {job.id}：{job.name}"

        if action in ("update", "remove", "run"):
            if not (job_id or "").strip():
                return f"{action} 需要提供 job_id 参数。"
            target_id = (job_id or "").strip()
            agent_id = self.current_agent_id or "main"
            try:
                if action == "remove":
                    service.delete_job(
                        target_id,
                        agent_id=agent_id,
                    )
                    return f"已删除任务 {target_id}。"

                if action == "run":
                    target = service.get_job(
                        target_id,
                        agent_id=agent_id,
                    )
                    service.trigger_job(
                        target_id,
                        agent_id=agent_id,
                    )
                    return (
                        f"已触发任务 {target.id}："
                        f"{target.payload.text[:50]}..."
                    )

                payload_patch = None
                if (
                    isinstance(payload, dict)
                    and payload.get("text") is not None
                ):
                    payload_patch = {
                        "kind": "systemEvent",
                        "text": str(payload.get("text", "")).strip(),
                    }
                target = service.update_job(
                    target_id,
                    name=name,
                    description=description,
                    schedule=(
                        schedule
                        if isinstance(schedule, dict) and schedule
                        else None
                    ),
                    payload=payload_patch,
                    scope_agent_id=agent_id,
                )
                return f"已更新任务 {target.id}。"
            except CronServiceError as exc:
                if exc.code == "not_found":
                    return f"未找到任务 {job_id}。"
                return f"{action} 任务失败：{exc}"

        return f"未知操作：{action}"


def get_cron_tools(
    agent_id: str = "main",
    cron_service: Any = None,
) -> list[BaseTool]:
    """返回 cron 工具实例"""
    tool = CronTool(current_agent_id=agent_id)
    tool._cron_service = cron_service
    return [tool]
