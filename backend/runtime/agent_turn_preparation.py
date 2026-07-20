"""Agent-specific adapter for turn input preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from runtime.turn_context import SessionContextCacheEntry


class AgentTurnPreparationAdapter:
    """Bind generic turn preparation to Agent runtime dependencies."""

    def __init__(
        self,
        *,
        preparation: Any,
        tool_registry: Any,
        get_data_dir: Callable[[], str],
        get_memory_store: Callable[[str], Any],
        resolve_workspace: Callable[[str], Path],
        resolve_agent_dir: Callable[[str], Path],
        resolve_agent_config: Callable[[str], dict[str, Any]],
        get_heartbeat_config: Callable[[str], dict[str, Any]],
        get_current_model: Callable[[str], Any],
        build_prompt: Callable[[Any], tuple[str, Any]],
        load_history: Callable[[str, str], list[dict[str, Any]]],
        format_summary: Callable[[Any], str],
        prune_history: Callable[..., list[dict[str, Any]]],
        count_tokens: Callable[[str], int],
        count_messages_tokens: Callable[[list[dict[str, Any]]], int],
    ) -> None:
        self._preparation = preparation
        self._tool_registry = tool_registry
        self._get_data_dir = get_data_dir
        self._get_memory_store = get_memory_store
        self._resolve_workspace = resolve_workspace
        self._resolve_agent_dir = resolve_agent_dir
        self._resolve_agent_config = resolve_agent_config
        self._get_heartbeat_config = get_heartbeat_config
        self._get_current_model = get_current_model
        self._build_prompt = build_prompt
        self._load_history = load_history
        self._format_summary = format_summary
        self._prune_history = prune_history
        self._count_tokens = count_tokens
        self._count_messages_tokens = count_messages_tokens

    def collect_tools(self, agent_id: str, session_id: str = "") -> list:
        return self._tool_registry.collect_tools(agent_id, session_id)

    def wrap_tools_for_session(
        self,
        agent_id: str,
        session_id: str,
        tools: list,
    ) -> list:
        return self._tool_registry.wrap_tools(
            self._get_data_dir(),
            agent_id,
            session_id,
            tools,
        )

    def build_tools(
        self,
        agent_id: str,
        session_id: str,
        *,
        collect_tools: Callable[[str, str], list],
        filter_tools: Callable[[str, list], list],
        wrap_tools: Callable[[str, str, list], list],
    ) -> list:
        return self._preparation.build_tools(
            self._tool_registry,
            self._get_data_dir(),
            agent_id,
            session_id,
            collect_tools=collect_tools,
            filter_tools=filter_tools,
            wrap_tools=wrap_tools,
        )

    def resolve_tool_policy(self, agent_id: str) -> tuple[list[str], list[str]]:
        return self._tool_registry.resolve_policy(agent_id)

    def filter_tools_by_policy(
        self,
        agent_id: str,
        tools: list,
        *,
        resolve_policy: Callable[[str], tuple[list[str], list[str]]],
    ) -> list:
        return self._tool_registry.filter_tools(
            agent_id,
            tools,
            resolve_policy=resolve_policy,
        )

    def build_messages(
        self,
        history: list[dict[str, Any]],
        new_message: str,
        *,
        human_message: Callable[..., Any],
        ai_message: Callable[..., Any],
        system_message: Callable[..., Any],
    ) -> list:
        return self._preparation.build_messages(
            history,
            new_message,
            human_message=human_message,
            ai_message=ai_message,
            system_message=system_message,
        )

    def safe_mtime(self, path: Path) -> float | None:
        return self._preparation.safe_mtime(path)

    def project_context_signature(
        self,
        agent_id: str,
        prompt_mode: str,
        *,
        safe_mtime: Callable[[Path], float | None],
    ) -> tuple[Any, ...]:
        return self._preparation.project_context_signature(
            agent_id,
            prompt_mode,
            resolve_workspace=self._resolve_workspace,
            resolve_agent_dir=self._resolve_agent_dir,
            safe_mtime=safe_mtime,
        )

    def prompt_runtime_signature(self, agent_id: str) -> tuple[Any, ...]:
        return self._preparation.prompt_runtime_signature(
            agent_id,
            resolve_agent_config=self._resolve_agent_config,
            get_heartbeat_config=self._get_heartbeat_config,
            get_current_model=self._get_current_model,
        )

    def pruning_signature(self, agent_id: str) -> str:
        return self._preparation.pruning_signature(
            agent_id,
            resolve_agent_config=self._resolve_agent_config,
        )

    def tool_policy_signature(
        self,
        agent_id: str,
        *,
        resolve_policy: Callable[[str], tuple[list[str], list[str]]],
    ) -> tuple[Any, ...]:
        return self._tool_registry.policy_signature(
            agent_id,
            resolve_policy=resolve_policy,
        )

    def get_or_build_tool_names(
        self,
        agent_id: str,
        *,
        collect_tools: Callable[[str, str], list],
        filter_tools: Callable[[str, list], list],
        policy_signature: Callable[[str], tuple[Any, ...]],
    ) -> tuple[str, ...]:
        return self._preparation.get_or_build_tool_names(
            self._tool_registry,
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
    ) -> tuple[str, Any, int]:
        return self._preparation.get_or_build_prompt(
            agent_id=agent_id,
            prompt_mode=prompt_mode,
            available_tool_names=available_tool_names,
            extra_system_prompt=extra_system_prompt or None,
            locale=locale,
            static_signature=static_signature,
            runtime_signature=runtime_signature,
            build_prompt=self._build_prompt,
            count_tokens=self._count_tokens,
        )

    def get_or_build_session_context(
        self,
        *,
        agent_id: str,
        session_id: str,
        pruning_signature: str,
    ) -> SessionContextCacheEntry:
        session_path = (
            self._resolve_agent_dir(agent_id)
            / "sessions"
            / f"{session_id}.json"
        )
        return self._preparation.get_or_build_session_context(
            agent_id=agent_id,
            session_id=session_id,
            session_path=session_path,
            store=self._get_memory_store(agent_id),
            load_history=self._load_history,
            format_summary=self._format_summary,
            prune_history=self._prune_history,
            count_tokens=self._count_tokens,
            count_messages_tokens=self._count_messages_tokens,
            pruning_signature=pruning_signature,
        )

    def session_summary_fingerprint(self, summary: Any) -> str | None:
        return self._preparation.session_summary_fingerprint(summary)
