from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class MemoryDashboardQueryTests(unittest.TestCase):
    def test_dashboard_queries_have_a_read_model_owner(self) -> None:
        query_path = BACKEND_DIR / "mem" / "dashboard_queries.py"

        self.assertTrue(
            query_path.is_file(),
            "mem.dashboard_queries should own dashboard projections",
        )
        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MemoryDashboardQueries(", store_source)
        self.assertNotIn('"totalChunks"', store_source)
        self.assertNotIn('"sessionKey"', store_source)
        self.assertNotIn('"qualityScore"', store_source)

    def test_store_dashboard_contract_is_preserved(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    import tempfile
                    from pathlib import Path
                    from mem.models import Chunk, Skill, Task
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        for chunk in (
                            Chunk(
                                id="chunk-1", session_key="session-1", turn_id="turn-1",
                                seq=0, role="user", content="A" * 350,
                                task_id="task-1", created_at=100, updated_at=100,
                            ),
                            Chunk(
                                id="chunk-2", session_key="session-1", turn_id="turn-1",
                                seq=1, role="assistant", content="B" * 350,
                                summary="second", task_id="task-1",
                                created_at=200, updated_at=200,
                            ),
                            Chunk(
                                id="chunk-3", session_key="session-2", turn_id="turn-2",
                                seq=0, role="user", content="duplicate",
                                dedup_status="duplicate", created_at=300, updated_at=300,
                            ),
                        ):
                            store.insert_chunk(chunk)
                        store.insert_task(Task(
                            id="task-1", session_key="session-1", title="completed",
                            summary="S" * 450, status="completed", started_at=100,
                            ended_at=150, updated_at=150,
                        ))
                        store.insert_task(Task(
                            id="task-2", session_key="session-2", title="active",
                            status="active", started_at=200, updated_at=200,
                        ))
                        store.insert_skill(Skill(
                            id="skill-1", name="active-skill", description="D" * 350,
                            status="active", quality_score=0.9,
                            created_at=100, updated_at=100,
                        ))
                        store.insert_skill(Skill(
                            id="skill-2", name="archived-skill", status="archived",
                            created_at=200, updated_at=200,
                        ))

                        tasks, task_total = store.list_dashboard_tasks()
                        completed, completed_total = store.list_dashboard_tasks(
                            status="completed"
                        )
                        task_page, page_total = store.list_dashboard_tasks(
                            limit=1, offset=1
                        )
                        skills = store.list_dashboard_skills()
                        active_skills = store.list_dashboard_skills(status="active")
                        memories, memory_total = store.list_dashboard_memories()
                        filtered, filtered_total = store.list_dashboard_memories(
                            session="session-1", role="user"
                        )
                        payload = {
                            "stats": store.get_dashboard_stats(),
                            "tasks": tasks,
                            "taskTotal": task_total,
                            "completed": completed,
                            "completedTotal": completed_total,
                            "taskPage": task_page,
                            "pageTotal": page_total,
                            "skills": skills,
                            "activeSkills": active_skills,
                            "memories": memories,
                            "memoryTotal": memory_total,
                            "filtered": filtered,
                            "filteredTotal": filtered_total,
                        }
                        store.close()
                        print(json.dumps(payload, sort_keys=True))
                    """
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.splitlines()[-1])

        self.assertEqual(
            payload["stats"],
            {
                "totalChunks": 2,
                "totalTasks": 2,
                "completedTasks": 1,
                "totalSkills": 2,
                "totalSessions": 2,
                "roleBreakdown": {"assistant": 1, "user": 1},
                "dedupBreakdown": {"active": 2, "duplicate": 1},
                "timeRange": {"earliest": 100, "latest": 300},
            },
        )
        self.assertEqual(payload["taskTotal"], 2)
        self.assertEqual([item["id"] for item in payload["tasks"]], ["task-2", "task-1"])
        self.assertEqual(payload["tasks"][1]["chunkCount"], 2)
        self.assertEqual(len(payload["tasks"][1]["summary"]), 400)
        self.assertEqual(payload["completedTotal"], 1)
        self.assertEqual([item["id"] for item in payload["completed"]], ["task-1"])
        self.assertEqual(payload["pageTotal"], 2)
        self.assertEqual([item["id"] for item in payload["taskPage"]], ["task-1"])
        self.assertEqual([item["id"] for item in payload["skills"]], ["skill-2", "skill-1"])
        self.assertEqual(len(payload["skills"][1]["description"]), 300)
        self.assertEqual(payload["skills"][1]["qualityScore"], 0.9)
        self.assertEqual([item["id"] for item in payload["activeSkills"]], ["skill-1"])
        self.assertEqual(payload["memoryTotal"], 2)
        self.assertEqual(
            [item["id"] for item in payload["memories"]],
            ["chunk-2", "chunk-1"],
        )
        self.assertEqual(len(payload["memories"][0]["excerpt"]), 300)
        self.assertEqual(payload["filteredTotal"], 1)
        self.assertEqual([item["id"] for item in payload["filtered"]], ["chunk-1"])


if __name__ == "__main__":
    unittest.main()
