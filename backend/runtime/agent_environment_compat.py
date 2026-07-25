"""Environment adapter methods retained on AgentManager for compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _agent_module_binding(name: str) -> Any:
    from runtime import agent as agent_module

    return getattr(agent_module, name)


class AgentManagerEnvironmentCompatibilityMixin:
    @staticmethod
    def _log_compress(
        agent_id: str,
        session_id: str,
        archived_count: int,
        remaining_count: int,
    ) -> None:
        _agent_module_binding("audit_logger").log_compress(
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
        _agent_module_binding("event_bus").emit(agent_id, event)

    @staticmethod
    def _audit_runtime_event(
        agent_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        _agent_module_binding("audit_logger").log(
            agent_id,
            event_type,
            data,
        )

    @staticmethod
    def _write_skills_snapshot(agent_id: str) -> None:
        from tools.skills_scanner import write_skills_snapshot

        write_skills_snapshot(agent_id)

    @staticmethod
    def _has_bootstrap(agent_id: str) -> bool:
        from runtime.workspace import has_bootstrap

        return has_bootstrap(agent_id)

    @staticmethod
    def _get_locale() -> str:
        from config import get_config

        return get_config().get("app", {}).get("locale", "zh-CN")

    @staticmethod
    def _resolve_context_budget(agent_id: str) -> Any:
        from runtime.context_budget import resolve_budget

        return resolve_budget(agent_id)

    def _get_state_persist_config(self, agent_id: str) -> tuple[bool, int]:
        try:
            from config import resolve_agent_config

            config = resolve_agent_config(agent_id)
            persist_config = config.get("statePersist", {})
            return (
                persist_config.get("enabled", True),
                persist_config.get("autoSaveIntervalMinutes", 5),
            )
        except Exception:
            return True, 5

    def _get_state_path(self, agent_id: str) -> Path:
        agent_dir = _agent_module_binding("resolve_agent_dir")(agent_id)
        return agent_dir / "agent_state.json"

    @staticmethod
    def _resolve_think_level(agent_id: str) -> Any:
        from llm.thinking import resolve_agent_think_default

        return resolve_agent_think_default(agent_id)

    @staticmethod
    def _ensure_agent_workspace(agent_id: str) -> None:
        from runtime.workspace import ensure_agent_workspace

        ensure_agent_workspace(agent_id)
