from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app_lifecycle import ApplicationLifecycle


class ApplicationLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_cleans_cron_after_partial_start_failure(self) -> None:
        cron_scheduler = SimpleNamespace(
            start=AsyncMock(side_effect=RuntimeError("cron start failed")),
            stop=AsyncMock(),
        )
        work_delivery = SimpleNamespace(
            fail_unrecoverable_pending=Mock(return_value=0),
            recover_pending_work=Mock(return_value=0),
        )
        lifecycle = ApplicationLifecycle(
            load_config=Mock(),
            setup_logging=Mock(),
            scan_skills=Mock(),
            agent_manager=SimpleNamespace(
                initialize=AsyncMock(),
                close=AsyncMock(),
            ),
            skills_watcher=SimpleNamespace(start=Mock(), stop=Mock()),
            configure_work_recovery=Mock(),
            work_delivery_provider=lambda: work_delivery,
            list_agents=lambda: [],
            heartbeat_runner=SimpleNamespace(
                start=AsyncMock(),
                stop=AsyncMock(),
            ),
            start_subagent_archive=Mock(),
            stop_subagent_archive=Mock(),
            get_config=lambda: {"cron": {"enabled": True}},
            cron_scheduler_factory=lambda: cron_scheduler,
            resume_subagent_runs=AsyncMock(),
            data_dir=Path("/tmp/data"),
            log=Mock(),
        )
        application = FastAPI()

        with self.assertRaisesRegex(RuntimeError, "cron start failed"):
            await lifecycle.start(application)
        await lifecycle.stop(application)

        self.assertIs(application.state.cron_scheduler, cron_scheduler)
        cron_scheduler.stop.assert_awaited_once_with()

    async def test_start_and_stop_preserve_runtime_order(self) -> None:
        order: list[str] = []
        agent_manager = SimpleNamespace(
            initialize=AsyncMock(
                side_effect=lambda _path: order.append("agent-start")
            ),
            close=AsyncMock(
                side_effect=lambda **_kwargs: order.append("agent-stop")
            ),
        )
        skills_watcher = SimpleNamespace(
            start=Mock(side_effect=lambda: order.append("watcher-start")),
            stop=Mock(side_effect=lambda: order.append("watcher-stop")),
        )
        work_delivery = SimpleNamespace(
            fail_unrecoverable_pending=Mock(
                side_effect=lambda: order.append("work-fail") or 0
            ),
            recover_pending_work=Mock(
                side_effect=lambda: order.append("work-recover") or 0
            ),
        )
        heartbeat_runner = SimpleNamespace(
            start=AsyncMock(
                side_effect=lambda _ids: order.append("heartbeat-start")
            ),
            stop=AsyncMock(
                side_effect=lambda: order.append("heartbeat-stop")
            ),
        )
        cron_scheduler = SimpleNamespace(
            start=AsyncMock(
                side_effect=lambda: order.append("cron-start")
            ),
            stop=AsyncMock(
                side_effect=lambda: order.append("cron-stop")
            ),
        )
        cron_scheduler_factory = Mock(return_value=cron_scheduler)
        lifecycle = ApplicationLifecycle(
            load_config=Mock(side_effect=lambda: order.append("config")),
            setup_logging=Mock(side_effect=lambda: order.append("logging")),
            scan_skills=Mock(side_effect=lambda: order.append("scan")),
            agent_manager=agent_manager,
            skills_watcher=skills_watcher,
            configure_work_recovery=Mock(
                side_effect=lambda: order.append("work-configure")
            ),
            work_delivery_provider=lambda: work_delivery,
            list_agents=lambda: [{"id": "main"}],
            heartbeat_runner=heartbeat_runner,
            start_subagent_archive=Mock(
                side_effect=lambda: order.append("archive-start")
            ),
            stop_subagent_archive=Mock(
                side_effect=lambda: order.append("archive-stop")
            ),
            get_config=lambda: {"cron": {"enabled": True}},
            cron_scheduler_factory=cron_scheduler_factory,
            resume_subagent_runs=AsyncMock(
                side_effect=lambda: order.append("subagent-resume")
            ),
            data_dir=Path("/tmp/data"),
            log=Mock(),
        )
        application = FastAPI()

        await lifecycle.start(application)
        await lifecycle.stop(application)

        self.assertEqual(
            order,
            [
                "config",
                "logging",
                "scan",
                "agent-start",
                "watcher-start",
                "work-configure",
                "work-fail",
                "heartbeat-start",
                "work-recover",
                "archive-start",
                "cron-start",
                "subagent-resume",
                "watcher-stop",
                "archive-stop",
                "cron-stop",
                "heartbeat-stop",
                "agent-stop",
            ],
        )
        cron_scheduler_factory.assert_called_once_with()
        self.assertIs(application.state.cron_scheduler, cron_scheduler)
        cron_scheduler.start.assert_awaited_once_with()
        cron_scheduler.stop.assert_awaited_once_with()
        agent_manager.initialize.assert_awaited_once_with("/tmp/data")
        agent_manager.close.assert_awaited_once_with(timeout=30)


if __name__ == "__main__":
    unittest.main()
