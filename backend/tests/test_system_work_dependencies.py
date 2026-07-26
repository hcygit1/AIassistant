from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.system_work_dependencies import (
    SystemWorkDependencies,
    system_work_dependencies,
)
from subagents.subagent_delivery import (
    SubagentAnnounceDelivery,
    subagent_announce_delivery,
)
from system_messages.heartbeat import HeartbeatRunner, heartbeat_runner
from system_messages.reminder_delivery import (
    ReminderDeliveryService,
    reminder_delivery_service,
)


class SystemWorkDependenciesTests(unittest.TestCase):
    def test_global_system_work_services_share_one_contract(self) -> None:
        for service in (
            heartbeat_runner,
            reminder_delivery_service,
            subagent_announce_delivery,
        ):
            self.assertIs(
                service._dependencies,
                system_work_dependencies,
            )

    def test_default_providers_resolve_legacy_globals_dynamically(self) -> None:
        dependencies = SystemWorkDependencies.defaults()
        session_manager = Mock()
        work_delivery = Mock()

        with (
            patch(
                "sessions.session_manager.session_manager",
                session_manager,
            ),
            patch(
                "sessions.session_work_delivery.session_work_delivery",
                work_delivery,
            ),
        ):
            self.assertIs(dependencies.session_manager, session_manager)
            self.assertIs(dependencies.work_delivery, work_delivery)

    def test_system_work_services_share_one_dependency_contract(self) -> None:
        session_manager = Mock()
        work_delivery = Mock()
        dependencies = SystemWorkDependencies.from_overrides(
            session_manager=session_manager,
            work_delivery=work_delivery,
        )

        services = (
            HeartbeatRunner(dependencies=dependencies),
            ReminderDeliveryService(dependencies=dependencies),
            SubagentAnnounceDelivery(dependencies=dependencies),
        )

        for service in services:
            self.assertIs(service.session_manager, session_manager)
            self.assertIs(service.work_delivery, work_delivery)

    def test_shared_contract_cannot_mix_with_legacy_dependencies(self) -> None:
        dependencies = SystemWorkDependencies.from_overrides()

        for service_type in (
            HeartbeatRunner,
            ReminderDeliveryService,
            SubagentAnnounceDelivery,
        ):
            with self.subTest(service_type=service_type.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "dependencies cannot be combined",
                ):
                    service_type(
                        dependencies=dependencies,
                        work_delivery=Mock(),
                    )


if __name__ == "__main__":
    unittest.main()
