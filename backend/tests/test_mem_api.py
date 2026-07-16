from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.mem_api import mem_memories, mem_skills, mem_stats, mem_tasks


class MemoryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_stats_selects_requested_agent_and_uses_public_query(self) -> None:
        store = Mock(spec=["get_dashboard_stats"])
        store.get_dashboard_stats.return_value = {
            "totalChunks": 2,
            "totalTasks": 1,
        }
        manager = Mock()
        manager.mem_stores = {"worker": store}

        with patch("runtime.agent.agent_manager", manager):
            result = await mem_stats(agent_id="worker")

        store.get_dashboard_stats.assert_called_once_with()
        self.assertTrue(result["ok"])
        self.assertEqual(result["totalChunks"], 2)

    async def test_collection_endpoints_delegate_without_connection_access(self) -> None:
        store = Mock(
            spec=[
                "list_dashboard_tasks",
                "list_dashboard_skills",
                "list_dashboard_memories",
            ]
        )
        store.list_dashboard_tasks.return_value = ([{"id": "task-1"}], 1)
        store.list_dashboard_skills.return_value = [{"id": "skill-1"}]
        store.list_dashboard_memories.return_value = ([{"id": "chunk-1"}], 1)
        manager = Mock()
        manager.mem_stores = {"worker": store}

        with patch("runtime.agent.agent_manager", manager):
            tasks = await mem_tasks(
                agent_id="worker",
                status="",
                limit=10,
                offset=0,
            )
            skills = await mem_skills(agent_id="worker", status="")
            memories = await mem_memories(
                agent_id="worker",
                limit=10,
                page=1,
                session="",
                role="",
            )

        self.assertEqual(tasks["total"], 1)
        self.assertEqual(skills["skills"][0]["id"], "skill-1")
        self.assertEqual(memories["total"], 1)


if __name__ == "__main__":
    unittest.main()
