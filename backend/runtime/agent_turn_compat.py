"""Legacy AgentManager turn-preparation forwarding methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.turn_context import SessionContextCacheEntry
from runtime.turn_preparation import TurnPreparation


def _agent_module_binding(name: str) -> Any:
    from runtime import agent as agent_module

    return getattr(agent_module, name)


class AgentManagerTurnPreparationCompatibilityMixin:
    """Keep legacy Manager patch points while adapters own the behavior."""

    def _collect_tools(self, agent_id: str, session_id: str = "") -> list:
        return self._turn_preparation_adapter.collect_tools(
            agent_id,
            session_id,
        )

    def _wrap_tools_for_session(
        self,
        agent_id: str,
        session_id: str,
        tools: list,
    ) -> list:
        return self._turn_preparation_adapter.wrap_tools_for_session(
            agent_id,
            session_id,
            tools,
        )

    def _build_tools(self, agent_id: str, session_id: str = "") -> list:
        return self._turn_preparation_adapter.build_tools(
            agent_id,
            session_id,
            collect_tools=self._collect_tools,
            filter_tools=self._filter_tools_by_policy,
            wrap_tools=self._wrap_tools_for_session,
        )

    def _resolve_tool_policy(self, agent_id: str) -> tuple[list[str], list[str]]:
        return self._turn_preparation_adapter.resolve_tool_policy(agent_id)

    def _filter_tools_by_policy(self, agent_id: str, tools: list) -> list:
        return self._turn_preparation_adapter.filter_tools_by_policy(
            agent_id,
            tools,
            resolve_policy=self._resolve_tool_policy,
        )

    def _build_messages(
        self,
        history: list[dict[str, Any]],
        new_message: str,
    ) -> list:
        return self._turn_preparation_adapter.build_messages(
            history,
            new_message,
            human_message=_agent_module_binding("HumanMessage"),
            ai_message=_agent_module_binding("AIMessage"),
            system_message=_agent_module_binding("SystemMessage"),
        )

    @staticmethod
    def _safe_mtime(path: Path) -> float | None:
        return TurnPreparation.safe_mtime(path)

    def _project_context_signature(
        self,
        agent_id: str,
        prompt_mode: str,
    ) -> tuple[Any, ...]:
        return self._turn_preparation_adapter.project_context_signature(
            agent_id,
            prompt_mode,
            safe_mtime=self._safe_mtime,
        )

    def _prompt_runtime_signature(
        self,
        agent_id: str,
    ) -> tuple[Any, ...]:
        return self._turn_preparation_adapter.prompt_runtime_signature(agent_id)

    def _pruning_signature(self, agent_id: str) -> str:
        return self._turn_preparation_adapter.pruning_signature(agent_id)

    def _tool_policy_signature(self, agent_id: str) -> tuple[Any, ...]:
        return self._turn_preparation_adapter.tool_policy_signature(
            agent_id,
            resolve_policy=self._resolve_tool_policy,
        )

    def _get_or_build_tool_names(self, agent_id: str) -> tuple[str, ...]:
        return self._turn_preparation_adapter.get_or_build_tool_names(
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
        return self._turn_preparation_adapter.get_or_build_prompt(
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
        )

    @staticmethod
    def _session_summary_fingerprint(summary: Any) -> str | None:
        return TurnPreparation.session_summary_fingerprint(summary)

    def _get_or_build_session_context(
        self,
        *,
        agent_id: str,
        session_id: str,
    ) -> SessionContextCacheEntry:
        return self._turn_preparation_adapter.get_or_build_session_context(
            agent_id=agent_id,
            session_id=session_id,
            pruning_signature=self._pruning_signature(agent_id),
        )
