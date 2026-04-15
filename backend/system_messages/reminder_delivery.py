"""Reminder delivery.

负责将提醒内容包装为 SessionWorkItem(kind="cron") 并投递到目标会话。

当前这层先提供一个统一入口，便于将 cron 从 heartbeat 交付链中剥离出来。
后续如果需要做持久化 work queue，可在这里替换具体实现，而不必重改 cron 入口。
"""

from __future__ import annotations

from sessions.session_dispatcher import PRIORITY_CRON
from sessions.session_manager import session_manager
from sessions.session_work_delivery import session_work_delivery


class ReminderDeliveryService:
    def build_cron_prompt(self, text: str) -> str:
        reminder_text = (text or "").strip()
        if not reminder_text:
            return (
                "A scheduled reminder has been triggered, but no reminder content was found. "
                "Reply briefly to indicate no reminder content was available."
            )
        return (
            "A scheduled reminder has been triggered. The reminder content is:\n\n"
            f"{reminder_text}\n\n"
            "Please relay this reminder to the user in a helpful and friendly way."
        )

    def deliver_cron_reminder(
        self,
        *,
        agent_id: str,
        text: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> int:
        target_session_id = session_id or session_manager.resolve_main_session_id(agent_id)
        return session_work_delivery.deliver(
            kind="cron",
            priority=PRIORITY_CRON,
            content=self.build_cron_prompt(text),
            agent_id=agent_id,
            session_id=target_session_id,
            run_id=run_id,
            recover_on_restart=True,
        )


reminder_delivery_service = ReminderDeliveryService()
