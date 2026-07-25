"""Tool construction, policy filtering, wrapping, and name caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from config import (
    is_tool_name_allowed,
    resolve_tool_catalog_signature,
    resolve_tool_policy,
)
from tools.runtime_dependencies import (
    ToolRuntimeDependencies,
    default_tool_runtime_dependencies,
)


CollectTools = Callable[[str, str], list]
FilterTools = Callable[[str, list], list]
WrapTools = Callable[[str, str, list], list]
ResolvePolicy = Callable[[str], tuple[list[str], list[str]]]
PolicySignature = Callable[[str], tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class ToolNameCacheEntry:
    key: tuple[Any, ...]
    tool_names: tuple[str, ...]


class ToolRegistry:
    def __init__(
        self,
        subagent_service: Any,
        runtime_dependencies: ToolRuntimeDependencies | None = None,
    ) -> None:
        self._subagent_service = subagent_service
        self._runtime_dependencies = (
            runtime_dependencies or default_tool_runtime_dependencies()
        )
        self.name_cache: dict[tuple[Any, ...], ToolNameCacheEntry] = {}

    def collect_tools(self, agent_id: str, session_id: str = "") -> list:
        from tools import get_all_tools

        return get_all_tools(
            agent_id,
            subagent_service=self._subagent_service,
            session_id=session_id,
            runtime_dependencies=self._runtime_dependencies,
        )

    @staticmethod
    def wrap_tools(
        data_dir: str,
        agent_id: str,
        session_id: str,
        tools: list,
    ) -> list:
        from tools.persistence_wrapper import wrap_tools_for_persistence

        return wrap_tools_for_persistence(
            tools,
            data_dir=data_dir,
            agent_id=agent_id,
            session_id=session_id,
        )

    def build_tools(
        self,
        data_dir: str,
        agent_id: str,
        session_id: str = "",
        *,
        collect_tools: CollectTools | None = None,
        filter_tools: FilterTools | None = None,
        wrap_tools: WrapTools | None = None,
    ) -> list:
        collect = collect_tools or self.collect_tools
        apply_policy = filter_tools or self.filter_tools
        tools = collect(agent_id, session_id)
        tools = apply_policy(agent_id, tools)
        if wrap_tools is not None:
            return wrap_tools(agent_id, session_id, tools)
        return self.wrap_tools(data_dir, agent_id, session_id, tools)

    @staticmethod
    def resolve_policy(agent_id: str) -> tuple[list[str], list[str]]:
        return resolve_tool_policy(agent_id)

    def filter_tools(
        self,
        agent_id: str,
        tools: list,
        *,
        resolve_policy: ResolvePolicy | None = None,
    ) -> list:
        resolver = resolve_policy or self.resolve_policy
        allow, deny = resolver(agent_id)
        return [
            tool
            for tool in tools
            if is_tool_name_allowed(tool.name, allow, deny)
        ]

    def policy_signature(
        self,
        agent_id: str,
        *,
        resolve_policy: ResolvePolicy | None = None,
    ) -> tuple[Any, ...]:
        resolver = resolve_policy or self.resolve_policy
        allow, deny = resolver(agent_id)
        return (
            tuple(sorted(str(item) for item in allow)),
            tuple(sorted(str(item) for item in deny)),
        )

    def get_or_build_tool_names(
        self,
        agent_id: str,
        *,
        collect_tools: CollectTools | None = None,
        filter_tools: FilterTools | None = None,
        policy_signature: PolicySignature | None = None,
    ) -> tuple[str, ...]:
        signature = policy_signature or self.policy_signature
        cache_key = (
            agent_id,
            signature(agent_id),
            resolve_tool_catalog_signature(),
        )
        cached = self.name_cache.get(cache_key)
        if cached is not None:
            return cached.tool_names

        collect = collect_tools or self.collect_tools
        apply_policy = filter_tools or self.filter_tools
        tools = collect(agent_id, "")
        tools = apply_policy(agent_id, tools)
        tool_names = tuple(sorted(tool.name for tool in tools))
        self.name_cache[cache_key] = ToolNameCacheEntry(
            key=cache_key,
            tool_names=tool_names,
        )
        return tool_names
