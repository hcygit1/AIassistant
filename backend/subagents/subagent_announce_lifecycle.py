"""Lifecycle projection for subagent announce delivery."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class SubagentAnnounceLifecycle:
    """Project announce work transitions to registry and event state."""

    def __init__(
        self,
        *,
        registry: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus

    @property
    def registry(self) -> Any:
        if self._registry is not None:
            return self._registry
        from subagents.subagent_registry import registry

        return registry

    @property
    def event_bus(self) -> Any:
        if self._event_bus is not None:
            return self._event_bus
        from infra.event_bus import event_bus

        return event_bus

    def emit(self, agent_id: str, run_id: str, result_delivery_state: str) -> None:
        from infra.event_bus import Events

        try:
            self.event_bus.emit(
                agent_id,
                Events.subagent_announce(
                    run_id=run_id,
                    result_delivery_state=result_delivery_state,
                ),
            )
        except Exception as exc:
            logger.warning(
                "Failed to emit subagent announce state run=%s: %s",
                run_id,
                exc,
            )

    def queued(self, agent_id: str, run_id: str) -> None:
        self.registry.set_result_delivery_state(run_id, "queued")
        self.emit(agent_id, run_id, "queued")

    def delivered(self, agent_id: str, run_id: str) -> None:
        self.registry.set_result_delivery_state(run_id, "delivering")
        self.registry.mark_result_delivery_delivered(run_id)
        self.emit(agent_id, run_id, "delivered")

    def dropped(self, agent_id: str, run_id: str) -> None:
        self.registry.mark_result_delivery_dropped(run_id)
        self.emit(agent_id, run_id, "dropped")

    def bind_work(self, run_id: str, work_id: str) -> None:
        self.registry.set_delivery_work_id(run_id, work_id)

    def emit_done(self, agent_id: str, run_id: str, result_preview: str) -> None:
        from infra.event_bus import Events

        try:
            self.event_bus.emit(
                agent_id,
                Events.subagent_done(
                    run_id=run_id,
                    result=result_preview,
                ),
            )
        except Exception as exc:
            logger.warning(
                "Failed to emit recovered subagent completion run=%s: %s",
                run_id,
                exc,
            )

    def complete_recovered_success(
        self,
        agent_id: str,
        run_id: str,
        result_preview: str,
    ) -> None:
        self.delivered(agent_id, run_id)
        self.emit_done(agent_id, run_id, result_preview)
        self.registry.remove_run(run_id)

    def complete_recovered_cancel(self, agent_id: str, run_id: str) -> None:
        self.dropped(agent_id, run_id)
        self.registry.remove_run(run_id)

    def complete_recovered_failure(
        self,
        *,
        agent_id: str,
        run_id: str,
        result_preview: str,
        persist_fallback: Callable[[], None],
    ) -> None:
        try:
            persist_fallback()
        except Exception as exc:
            logger.warning(
                "Failed to persist recovered announce fallback run=%s: %s",
                run_id,
                exc,
            )
        self.dropped(agent_id, run_id)
        self.emit_done(agent_id, run_id, result_preview)
        self.registry.remove_run(run_id)
