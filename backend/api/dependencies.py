"""FastAPI dependency providers for runtime compatibility globals."""

from __future__ import annotations

from typing import Any


def get_agent_manager() -> Any:
    from runtime.agent import agent_manager

    return agent_manager


def get_heartbeat_runner() -> Any:
    from system_messages.heartbeat import heartbeat_runner

    return heartbeat_runner


def get_event_bus() -> Any:
    from infra.event_bus import event_bus

    return event_bus


def get_cron_service() -> Any:
    from scheduler.cron_service import cron_service

    return cron_service


def get_task_history_service() -> Any:
    from scheduler.task_history_service import task_history_service

    return task_history_service


def get_session_work_history_service() -> Any:
    from sessions.session_work_history import session_work_history_service

    return session_work_history_service


def get_session_manager() -> Any:
    from sessions.session_manager import session_manager

    return session_manager


def get_user_turn_service() -> Any:
    from turns.service import user_turn_service

    return user_turn_service
