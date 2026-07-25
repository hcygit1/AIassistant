"""Explicit external bindings used to assemble one Agent runtime graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


RuntimeCallable = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class AgentRuntimeBindings:
    resolve_agent_model: RuntimeCallable
    resolve_fallback_candidates: RuntimeCallable
    find_model_by_id: RuntimeCallable
    get_model: RuntimeCallable
    invalidate_llm: RuntimeCallable
    get_or_create_llm: RuntimeCallable
    get_model_display_name: RuntimeCallable
    resolve_agent_workspace: RuntimeCallable
    resolve_agent_dir: RuntimeCallable
    resolve_agent_config: RuntimeCallable
    get_heartbeat_config: RuntimeCallable
    build_system_prompt_with_report: RuntimeCallable
    format_session_summary: RuntimeCallable
    prune_messages: RuntimeCallable
    count_tokens: RuntimeCallable
    count_messages_tokens: RuntimeCallable
    create_llm: RuntimeCallable
    get_run_tracker: RuntimeCallable
    get_audit_logger: RuntimeCallable
    get_session_manager: RuntimeCallable
    execute_command: RuntimeCallable
    parse_command: RuntimeCallable
    run_with_fallback_stream: RuntimeCallable
    detect_compaction_level: RuntimeCallable
    list_agents: RuntimeCallable

    @classmethod
    def from_module_symbols(
        cls,
        symbols: Mapping[str, Any],
    ) -> AgentRuntimeBindings:
        def function(name: str) -> RuntimeCallable:
            return lambda *args, **kwargs: symbols[name](*args, **kwargs)

        def method(owner: str, name: str) -> RuntimeCallable:
            return lambda *args, **kwargs: getattr(
                symbols[owner],
                name,
            )(*args, **kwargs)

        def value(name: str) -> RuntimeCallable:
            return lambda: symbols[name]

        return cls(
            resolve_agent_model=function("resolve_agent_model"),
            resolve_fallback_candidates=function(
                "resolve_fallback_candidates"
            ),
            find_model_by_id=method("models_config", "find_model_by_id"),
            get_model=method("models_config", "get_model"),
            invalidate_llm=method("llm_cache", "invalidate"),
            get_or_create_llm=method("llm_cache", "get_or_create"),
            get_model_display_name=function("get_model_display_name"),
            resolve_agent_workspace=function("resolve_agent_workspace"),
            resolve_agent_dir=function("resolve_agent_dir"),
            resolve_agent_config=function("resolve_agent_config"),
            get_heartbeat_config=function("get_heartbeat_config"),
            build_system_prompt_with_report=method(
                "prompt_builder",
                "build_system_prompt_with_report",
            ),
            format_session_summary=method(
                "prompt_builder",
                "format_session_summary",
            ),
            prune_messages=function("prune_messages"),
            count_tokens=function("count_tokens"),
            count_messages_tokens=function("count_messages_tokens"),
            create_llm=function("create_llm"),
            get_run_tracker=value("run_tracker"),
            get_audit_logger=value("audit_logger"),
            get_session_manager=value("session_manager"),
            execute_command=function("execute_command"),
            parse_command=function("parse_command"),
            run_with_fallback_stream=function("run_with_fallback_stream"),
            detect_compaction_level=symbols["detect_compaction_level"],
            list_agents=function("list_agents"),
        )
