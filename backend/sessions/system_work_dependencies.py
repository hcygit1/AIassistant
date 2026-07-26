"""Shared dependency contract for system work entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def _default_session_manager() -> Any:
    from sessions.session_manager import session_manager

    return session_manager


def _default_work_delivery() -> Any:
    from sessions.session_work_delivery import session_work_delivery

    return session_work_delivery


@dataclass(frozen=True, slots=True)
class SystemWorkDependencies:
    session_manager_provider: Callable[[], Any]
    work_delivery_provider: Callable[[], Any]

    @property
    def session_manager(self) -> Any:
        return self.session_manager_provider()

    @property
    def work_delivery(self) -> Any:
        return self.work_delivery_provider()

    @classmethod
    def defaults(cls) -> "SystemWorkDependencies":
        return cls(
            session_manager_provider=_default_session_manager,
            work_delivery_provider=_default_work_delivery,
        )

    @classmethod
    def from_overrides(
        cls,
        *,
        session_manager: Any | None = None,
        work_delivery: Any | None = None,
    ) -> "SystemWorkDependencies":
        return cls(
            session_manager_provider=(
                _default_session_manager
                if session_manager is None
                else lambda: session_manager
            ),
            work_delivery_provider=(
                _default_work_delivery
                if work_delivery is None
                else lambda: work_delivery
            ),
        )

    @classmethod
    def resolve(
        cls,
        *,
        dependencies: "SystemWorkDependencies | None" = None,
        session_manager: Any | None = None,
        work_delivery: Any | None = None,
    ) -> "SystemWorkDependencies":
        if dependencies is not None and (
            session_manager is not None or work_delivery is not None
        ):
            raise ValueError(
                "dependencies cannot be combined with legacy dependencies"
            )
        return dependencies or cls.from_overrides(
            session_manager=session_manager,
            work_delivery=work_delivery,
        )


system_work_dependencies = SystemWorkDependencies.defaults()
