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


class MemorySearchQueryTests(unittest.TestCase):
    def test_fts_and_ann_queries_have_one_read_only_owner(self) -> None:
        query_path = BACKEND_DIR / "mem" / "search_queries.py"

        self.assertTrue(
            query_path.is_file(),
            "mem.search_queries should own FTS and ANN queries",
        )
        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MemorySearchQueries(", store_source)
        self.assertNotIn("v.embedding MATCH", store_source)
        self.assertNotIn("chunks_fts MATCH", store_source)

    def test_store_preserves_all_search_filters_and_results(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    import tempfile
                    from dataclasses import asdict
                    from pathlib import Path
                    from mem.models import Chunk, Skill, Task
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        chunks = (
                            Chunk(
                                id="chunk-task-owner", session_key="session-old",
                                turn_id="turn-1", seq=0, role="user",
                                content="needle " + "A" * 350,
                                summary="task owner summary", task_id="task-a",
                                owner="owner-a", created_at=100, updated_at=100,
                            ),
                            Chunk(
                                id="chunk-task-other", session_key="session-other",
                                turn_id="turn-2", seq=0, role="assistant",
                                content="needle task other", task_id="task-b",
                                owner="owner-b", created_at=200, updated_at=200,
                            ),
                            Chunk(
                                id="chunk-orphan-owner", session_key="session-old",
                                turn_id="turn-3", seq=0, role="assistant",
                                content="needle orphan owner", dedup_status="orphaned",
                                owner="owner-a", created_at=300, updated_at=300,
                            ),
                            Chunk(
                                id="chunk-orphan-current", session_key="session-current",
                                turn_id="turn-4", seq=0, role="user",
                                content="needle orphan current", owner="owner-a",
                                created_at=400, updated_at=400,
                            ),
                            Chunk(
                                id="chunk-duplicate", session_key="session-old",
                                turn_id="turn-5", seq=0, role="user",
                                content="needle duplicate", dedup_status="duplicate",
                                owner="owner-a", created_at=500, updated_at=500,
                            ),
                            Chunk(
                                id="chunk-far", session_key="session-far",
                                turn_id="turn-6", seq=0, role="assistant",
                                content="unrelated", owner="owner-a",
                                created_at=600, updated_at=600,
                            ),
                        )
                        vectors = {
                            "chunk-task-owner": [1.0, 0.0, 0.0],
                            "chunk-task-other": [0.95, 0.05, 0.0],
                            "chunk-orphan-owner": [0.9, 0.1, 0.0],
                            "chunk-orphan-current": [0.85, 0.15, 0.0],
                            "chunk-duplicate": [1.0, 0.0, 0.0],
                            "chunk-far": [0.0, 1.0, 0.0],
                        }
                        for chunk in chunks:
                            store.insert_chunk(chunk)
                            store.upsert_chunk_embedding(chunk.id, vectors[chunk.id])

                        tasks = (
                            Task(
                                id="task-a", session_key="session-old",
                                owner="owner-a", title="needle task a",
                                summary="completed owner task", status="completed",
                                started_at=100, ended_at=150, updated_at=150,
                            ),
                            Task(
                                id="task-b", session_key="session-other",
                                owner="owner-b", title="needle task b",
                                summary="completed other task", status="completed",
                                started_at=200, ended_at=250, updated_at=250,
                            ),
                            Task(
                                id="task-active", session_key="session-old",
                                owner="owner-a", title="needle active task",
                                status="active", started_at=300, updated_at=300,
                            ),
                        )
                        task_vectors = {
                            "task-a": [1.0, 0.0, 0.0],
                            "task-b": [0.9, 0.1, 0.0],
                            "task-active": [1.0, 0.0, 0.0],
                        }
                        for task in tasks:
                            store.insert_task(task)
                            store.upsert_task_embedding(task.id, task_vectors[task.id])

                        skills = (
                            Skill(
                                id="skill-a", name="needle-skill-a",
                                description="owner skill", owner="owner-a",
                                status="active", created_at=100, updated_at=100,
                            ),
                            Skill(
                                id="skill-b", name="needle-skill-b",
                                description="other skill", owner="owner-b",
                                status="draft", created_at=200, updated_at=200,
                            ),
                            Skill(
                                id="skill-archived", name="needle-skill-archived",
                                owner="owner-a", status="archived",
                                created_at=300, updated_at=300,
                            ),
                        )
                        skill_vectors = {
                            "skill-a": [1.0, 0.0, 0.0],
                            "skill-b": [0.9, 0.1, 0.0],
                            "skill-archived": [1.0, 0.0, 0.0],
                        }
                        for skill in skills:
                            store.insert_skill(skill)
                            store.upsert_skill_embedding(skill.id, skill_vectors[skill.id])

                        query_vec = [1.0, 0.0, 0.0]
                        ann_chunks = store.ann_search_chunks(
                            query_vec, top_k=10, exclude_session="session-current"
                        )
                        payload = {
                            "annChunks": sorted(hit.chunk_id for hit in ann_chunks),
                            "annChunkMetadata": asdict(next(
                                hit for hit in ann_chunks
                                if hit.chunk_id == "chunk-task-owner"
                            )),
                            "annTasks": sorted(
                                hit.task_id for hit in store.ann_search_tasks(
                                    query_vec, top_k=10, owner="owner-a"
                                )
                            ),
                            "annSkills": sorted(
                                hit.skill_id for hit in store.ann_search_skills(
                                    query_vec, top_k=10, owner="owner-a"
                                )
                            ),
                            "dedup": sorted(
                                hit.chunk_id for hit in store.ann_dedup_candidates(
                                    query_vec, threshold=0.5, top_k=10, owner="owner-a"
                                )
                            ),
                            "annTaskChunks": sorted(
                                hit.chunk_id for hit in store.ann_search_chunks_in_tasks(
                                    query_vec, ["task-a", "task-b"],
                                    top_k=10, owner="owner-a"
                                )
                            ),
                            "annOrphans": sorted(
                                hit.chunk_id for hit in store.ann_search_orphan_chunks(
                                    query_vec, top_k=10,
                                    exclude_session="session-current", owner="owner-a"
                                )
                            ),
                            "ftsChunks": sorted(
                                hit.chunk_id for hit in store.fts_search_chunks(
                                    "needle", limit=10,
                                    exclude_session="session-current"
                                )
                            ),
                            "ftsTasks": sorted(
                                hit.task_id for hit in store.fts_search_tasks(
                                    "needle", limit=10, owner="owner-a"
                                )
                            ),
                            "ftsSkills": sorted(
                                hit.skill_id for hit in store.fts_search_skills(
                                    "needle", limit=10, owner="owner-a"
                                )
                            ),
                            "ftsTaskChunks": sorted(
                                hit.chunk_id for hit in store.fts_search_chunks_in_tasks(
                                    "needle", ["task-a", "task-b"],
                                    limit=10, owner="owner-a"
                                )
                            ),
                            "ftsOrphans": sorted(
                                hit.chunk_id for hit in store.fts_search_orphan_chunks(
                                    "needle", limit=10,
                                    exclude_session="session-current", owner="owner-a"
                                )
                            ),
                            "emptyAnnTaskChunks": store.ann_search_chunks_in_tasks(
                                query_vec, []
                            ),
                            "emptyFtsTaskChunks": store.fts_search_chunks_in_tasks(
                                "needle", []
                            ),
                            "invalidFts": store.fts_search_chunks("!!!"),
                        }
                        store._conn.execute("DROP TABLE skills_fts")
                        payload["ftsErrorFallback"] = store.fts_search_skills("needle")
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
            payload["annChunks"],
            ["chunk-far", "chunk-task-other", "chunk-task-owner"],
        )
        self.assertEqual(payload["annTasks"], ["task-a"])
        self.assertEqual(payload["annSkills"], ["skill-a"])
        self.assertEqual(
            payload["dedup"],
            ["chunk-orphan-current", "chunk-task-owner"],
        )
        self.assertEqual(payload["annTaskChunks"], ["chunk-task-owner"])
        self.assertEqual(
            payload["annOrphans"],
            ["chunk-far", "chunk-orphan-owner"],
        )
        self.assertEqual(
            payload["ftsChunks"],
            ["chunk-task-other", "chunk-task-owner"],
        )
        self.assertEqual(payload["ftsTasks"], ["task-a"])
        self.assertEqual(payload["ftsSkills"], ["skill-a"])
        self.assertEqual(payload["ftsTaskChunks"], ["chunk-task-owner"])
        self.assertEqual(payload["ftsOrphans"], ["chunk-orphan-owner"])
        self.assertEqual(payload["emptyAnnTaskChunks"], [])
        self.assertEqual(payload["emptyFtsTaskChunks"], [])
        self.assertEqual(payload["invalidFts"], [])
        self.assertEqual(payload["ftsErrorFallback"], [])
        metadata = payload["annChunkMetadata"]
        self.assertEqual(metadata["summary"], "task owner summary")
        self.assertEqual(metadata["session_key"], "session-old")
        self.assertEqual(metadata["task_id"], "task-a")
        self.assertEqual(metadata["created_at"], 100)
        self.assertEqual(len(metadata["content_excerpt"]), 300)


if __name__ == "__main__":
    unittest.main()
