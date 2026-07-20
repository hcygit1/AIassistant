"""Prepare the inputs shared by the turn orchestration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from runtime.turn_context import (
    SessionContextCacheEntry,
    TurnContext,
)


class TurnPreparation:
    """Coordinate prompt, history, and tool preparation for one turn."""

    def __init__(self, turn_context: TurnContext | None = None) -> None:
        self._turn_context = turn_context or TurnContext()

    def build_messages(
        self,
        history: list[dict[str, Any]],
        new_message: str,
        *,
        human_message: Callable[..., Any] = HumanMessage,
        ai_message: Callable[..., Any] = AIMessage,
        system_message: Callable[..., Any] = SystemMessage,
    ) -> list:
        return self._turn_context.build_messages(
            history,
            new_message,
            human_message=human_message,
            ai_message=ai_message,
            system_message=system_message,
        )

    @staticmethod
    def safe_mtime(path: Path) -> float | None:
        return TurnContext.safe_mtime(path)

    @staticmethod
    def project_context_signature(
        agent_id: str,
        prompt_mode: str,
        *,
        resolve_workspace: Callable[[str], Path],
        resolve_agent_dir: Callable[[str], Path],
        safe_mtime: Callable[[Path], float | None],
    ) -> tuple[Any, ...]:
        return TurnContext.project_context_signature(
            agent_id,
            prompt_mode,
            resolve_workspace=resolve_workspace,
            resolve_agent_dir=resolve_agent_dir,
            safe_mtime=safe_mtime,
        )

    @staticmethod
    def prompt_runtime_signature(
        agent_id: str,
        *,
        resolve_agent_config: Callable[[str], dict[str, Any]],
        get_heartbeat_config: Callable[[str], dict[str, Any]],
        get_current_model: Callable[[str], Any],
    ) -> tuple[Any, ...]:
        base_signature = TurnContext.prompt_runtime_signature(
            agent_id,
            resolve_agent_config_fn=resolve_agent_config,
            get_heartbeat_config_fn=get_heartbeat_config,
        )
        return (*base_signature, str(get_current_model(agent_id)))

    @staticmethod
    def pruning_signature(
        agent_id: str,
        *,
        resolve_agent_config: Callable[[str], dict[str, Any]],
    ) -> str:
        return TurnContext.pruning_signature(
            agent_id,
            resolve_agent_config_fn=resolve_agent_config,
        )

    @staticmethod
    def build_tools(
        registry: Any,
        data_dir: str,
        agent_id: str,
        session_id: str,
        *,
        collect_tools: Callable[[str, str], list],
        filter_tools: Callable[[str, list], list],
        wrap_tools: Callable[[str, str, list], list],
    ) -> list:
        return registry.build_tools(
            data_dir,
            agent_id,
            session_id,
            collect_tools=collect_tools,
            filter_tools=filter_tools,
            wrap_tools=wrap_tools,
        )

    @staticmethod
    def get_or_build_tool_names(
        registry: Any,
        agent_id: str,
        *,
        collect_tools: Callable[[str, str], list],
        filter_tools: Callable[[str, list], list],
        policy_signature: Callable[[str], tuple[Any, ...]],
    ) -> tuple[str, ...]:
        return registry.get_or_build_tool_names(
            agent_id,
            collect_tools=collect_tools,
            filter_tools=filter_tools,
            policy_signature=policy_signature,
        )

    def get_or_build_prompt(
        self,
        *,
        agent_id: str,
        prompt_mode: str,
        available_tool_names: list[str] | None,
        extra_system_prompt: str | None,
        locale: str,
        static_signature: tuple[Any, ...],
        runtime_signature: tuple[Any, ...],
        build_prompt: Callable[[Any], tuple[str, Any]],
        count_tokens: Callable[[str], int],
    ) -> tuple[str, Any, int]:
        return self._turn_context.get_or_build_prompt(
            agent_id=agent_id,
            prompt_mode=prompt_mode,
            available_tool_names=available_tool_names,
            extra_system_prompt=extra_system_prompt,
            locale=locale,
            static_signature=static_signature,
            runtime_signature=runtime_signature,
            build_prompt=build_prompt,
            count_tokens=count_tokens,
        )

    @staticmethod
    def session_summary_fingerprint(summary: Any) -> str | None:
        return TurnContext.session_summary_fingerprint(summary)

    def get_or_build_session_context(
        self,
        *,
        agent_id: str,
        session_id: str,
        session_path: Path,
        store: Any,
        load_history: Callable[[str, str], list[dict[str, Any]]],
        format_summary: Callable[[Any], str],
        prune_history: Callable[..., list[dict[str, Any]]],
        count_tokens: Callable[[str], int],
        count_messages_tokens: Callable[[list[dict[str, Any]]], int],
        pruning_signature: str,
    ) -> SessionContextCacheEntry:
        return self._turn_context.get_or_build_session_context(
            agent_id=agent_id,
            session_id=session_id,
            session_path=session_path,
            store=store,
            safe_mtime=self.safe_mtime,
            summary_fingerprint=self.session_summary_fingerprint,
            load_history=load_history,
            format_summary=format_summary,
            prune_history=prune_history,
            count_tokens=count_tokens,
            count_messages_tokens=count_messages_tokens,
            pruning_signature=pruning_signature,
        )
