from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.cron_api import router
from api.dependencies import (
    get_cron_service,
    get_session_work_history_service,
    get_task_history_service,
)


class CronApiDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cron_service = Mock()
        self.job = Mock()
        self.job.to_dict.return_value = {"id": "cron-1"}
        self.cron_service.list_jobs.return_value = [self.job]
        self.cron_service.create_job.return_value = self.job
        self.cron_service.get_job.return_value = self.job
        self.cron_service.update_job.return_value = self.job
        self.cron_service.create_reminder.return_value = self.job
        self.task_history = Mock()
        self.task_history.query.return_value = SimpleNamespace(
            items=[{"id": "task-1"}],
            total=1,
            limit=10,
            offset=0,
        )
        self.task_history.cancel.return_value = SimpleNamespace(
            ok=True,
            status="cancelled",
        )
        self.system_history = Mock()
        self.system_history.query.return_value = SimpleNamespace(
            items=[{"id": "work-1"}],
            total=1,
            limit=10,
            offset=0,
        )
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_cron_service] = (
            lambda: self.cron_service
        )
        app.dependency_overrides[get_task_history_service] = (
            lambda: self.task_history
        )
        app.dependency_overrides[get_session_work_history_service] = (
            lambda: self.system_history
        )
        self.app = app
        self.client = TestClient(app)

    def test_cron_routes_use_overridden_service(self) -> None:
        create_body = {
            "name": "report",
            "agent_id": "main",
            "schedule": {"kind": "every", "everyMs": 60_000},
            "payload": {"kind": "systemEvent", "text": "report"},
        }

        responses = [
            self.client.get("/api/cron/jobs"),
            self.client.post("/api/cron/jobs", json=create_body),
            self.client.get("/api/cron/jobs/cron-1"),
            self.client.patch(
                "/api/cron/jobs/cron-1",
                json={"name": "renamed"},
            ),
            self.client.delete("/api/cron/jobs/cron-1"),
            self.client.post("/api/cron/jobs/cron-1/run"),
            self.client.post(
                "/api/reminders",
                json={
                    "text": "wake",
                    "at": "2030-01-01T00:00:00Z",
                    "agent_id": "main",
                },
            ),
        ]

        self.assertTrue(
            all(response.status_code == 200 for response in responses)
        )
        self.cron_service.list_jobs.assert_called_once_with()
        self.cron_service.create_job.assert_called_once()
        self.cron_service.get_job.assert_called_once_with("cron-1")
        self.cron_service.update_job.assert_called_once()
        self.cron_service.delete_job.assert_called_once_with("cron-1")
        self.cron_service.trigger_job.assert_called_once_with("cron-1")
        self.cron_service.create_reminder.assert_called_once_with(
            text="wake",
            at="2030-01-01T00:00:00Z",
            agent_id="main",
        )

    def test_history_routes_use_overridden_services(self) -> None:
        task_response = self.client.get(
            "/api/tasks/history?agent_id=main&kind=cron&status=done"
            "&limit=10&offset=0"
        )
        cancel_response = self.client.post("/api/tasks/task-1/cancel")
        system_response = self.client.get(
            "/api/system-work/history?kind=heartbeat&status=done"
            "&agent_id=main&session_id=main-main&run_id=run-1"
            "&limit=10&offset=0"
        )

        self.assertEqual(task_response.status_code, 200)
        self.assertEqual(cancel_response.status_code, 200)
        self.assertEqual(system_response.status_code, 200)
        self.task_history.query.assert_called_once_with(
            agent_id="main",
            kind="cron",
            status="done",
            limit=10,
            offset=0,
        )
        self.task_history.cancel.assert_called_once_with("task-1")
        self.system_history.query.assert_called_once_with(
            kind="heartbeat",
            status="done",
            agent_id="main",
            session_id="main-main",
            run_id="run-1",
            limit=10,
            offset=0,
        )

    def test_dependencies_do_not_change_openapi_parameters(self) -> None:
        paths = self.app.openapi()["paths"]

        self.assertEqual(
            paths["/api/cron/jobs"]["post"].get("parameters", []),
            [],
        )
        self.assertEqual(
            {
                item["name"]
                for item in paths["/api/cron/jobs/{job_id}/run"][
                    "post"
                ]["parameters"]
            },
            {"job_id", "mode"},
        )
        self.assertEqual(
            {
                item["name"]
                for item in paths["/api/tasks/history"]["get"][
                    "parameters"
                ]
            },
            {"agent_id", "kind", "status", "limit", "offset"},
        )


if __name__ == "__main__":
    unittest.main()
