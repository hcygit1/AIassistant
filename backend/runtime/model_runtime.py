"""Per-agent runtime model overrides and candidate resolution."""

from __future__ import annotations

from typing import Any, Callable

from llm.models_config import ModelRef, parse_model_ref


class ModelRuntime:
    def __init__(
        self,
        *,
        resolve_configured_model: Callable[[str], ModelRef],
        resolve_configured_candidates: Callable[
            [str],
            list[ModelRef],
        ],
        find_model: Callable[[str], Any],
        get_model: Callable[[ModelRef], Any],
        invalidate_llm: Callable[[str], None],
        get_or_create_llm: Callable[[str, ModelRef], Any],
        get_display_name: Callable[[ModelRef], str],
    ) -> None:
        self._resolve_configured_model = (
            resolve_configured_model
        )
        self._resolve_configured_candidates = (
            resolve_configured_candidates
        )
        self._find_model = find_model
        self._get_model = get_model
        self._invalidate_llm = invalidate_llm
        self._get_or_create_llm = get_or_create_llm
        self._get_display_name = get_display_name
        self._overrides: dict[str, ModelRef] = {}

    def resolve_current(self, agent_id: str) -> ModelRef:
        override = self._overrides.get(agent_id)
        if (
            override is not None
            and self._get_model(override) is not None
        ):
            return override
        if override is not None:
            self.restore_override(agent_id, None)
        return self._resolve_configured_model(agent_id)

    def get_override(
        self,
        agent_id: str,
    ) -> ModelRef | None:
        return self._overrides.get(agent_id)

    def resolve_candidates(
        self,
        agent_id: str,
    ) -> list[ModelRef]:
        configured: list[ModelRef] = []
        configured_seen: set[str] = set()
        for candidate in self._resolve_configured_candidates(
            agent_id
        ):
            model_definition = self._get_model(candidate)
            if model_definition is None:
                continue
            canonical = ModelRef(
                provider=candidate.provider,
                model=model_definition.id,
            )
            key = str(canonical)
            if key in configured_seen:
                continue
            configured_seen.add(key)
            configured.append(canonical)
        override = self.get_override(agent_id)
        if (
            override is not None
            and self._get_model(override) is None
        ):
            self.restore_override(agent_id, None)
            override = None
        if override is None:
            return configured
        return [
            override,
            *(
                candidate
                for candidate in configured
                if candidate != override
            ),
        ]

    def get_llm(self, agent_id: str) -> Any:
        return self._get_or_create_llm(
            agent_id,
            self.resolve_current(agent_id),
        )

    def switch(
        self,
        agent_id: str,
        model_raw: str,
    ) -> str:
        ref = parse_model_ref(model_raw)
        if not ref:
            raise ValueError(
                f"Invalid model reference: {model_raw}"
            )

        if not ref.provider:
            found = self._find_model(ref.model)
            if not found:
                raise ValueError(
                    f"Model '{ref.model}' not found "
                    "in any provider"
                )
            provider, model_definition = found
            ref = ModelRef(
                provider=provider.id,
                model=model_definition.id,
            )

        model_definition = self._get_model(ref)
        if model_definition is None:
            raise ValueError(
                f"Model '{ref}' is not configured"
            )
        ref = ModelRef(
            provider=ref.provider,
            model=model_definition.id,
        )

        display_name = self._get_display_name(ref)
        self._invalidate_llm(agent_id)
        self._get_or_create_llm(agent_id, ref)
        self._overrides[agent_id] = ref
        return display_name

    def restore_override(
        self,
        agent_id: str,
        override: ModelRef | None,
    ) -> None:
        self._invalidate_llm(agent_id)
        if override is None:
            self._overrides.pop(agent_id, None)
            return
        self._overrides[agent_id] = override

    def clear(self, agent_id: str | None = None) -> None:
        if agent_id is None:
            for override_agent_id in list(self._overrides):
                self._invalidate_llm(override_agent_id)
            self._overrides.clear()
            return
        if agent_id in self._overrides:
            self._invalidate_llm(agent_id)
            self._overrides.pop(agent_id, None)
