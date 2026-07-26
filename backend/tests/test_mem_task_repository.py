from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Task


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            owner TEXT NOT NULL DEFAULT 'agent:main',
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            boundary_summary TEXT NOT NULL DEFAULT '',
            boundary_compacted_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'paragraph',
            summary TEXT NOT NULL DEFAULT '',
            task_id TEXT,
            skill_id TEXT,
            owner TEXT NOT NULL DEFAULT 'agent:main',
            content_hash TEXT NOT NULL DEFAULT '',
            dedup_status TEXT NOT NULL DEFAULT 'active',
            dedup_target TEXT,
            dedup_reason TEXT,
            summary_source TEXT NOT NULL DEFAULT 'llm',
            embedding_status TEXT NOT NULL DEFAULT 'ok',
            embedding_error TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );"""
    )


def _insert_chunk(
    connection: sqlite3.Connection,
    chunk_id: str,
    *,
    session_key: str = "session-1",
    owner: str = "owner-a",
    task_id: str | None = None,
    dedup_status: str = "active",
    created_at: int = 1,
) -> None:
    connection.execute(
        """INSERT INTO chunks
        (id, session_key, turn_id, seq, role, content, kind, summary,
         task_id, skill_id, owner, content_hash, dedup_status, dedup_target,
         dedup_reason, summary_source, embedding_status, embedding_error,
         created_at, updated_at)
        VALUES (?, ?, ?, 0, 'user', ?, 'paragraph', '', ?, NULL, ?, ?, ?,
                NULL, NULL, 'llm', 'ok', NULL, ?, ?)""",
        (
            chunk_id,
            session_key,
            f"turn-{chunk_id}",
            chunk_id,
            task_id,
            owner,
            f"hash-{chunk_id}",
            dedup_status,
            created_at,
            created_at,
        ),
    )


def _repository_type() -> type[Any]:
    try:
        from mem.task_repository import TaskRepository
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.task_repository should own task persistence"
        ) from exc
    return TaskRepository


class TaskRepositoryTests(unittest.TestCase):
    def test_task_persistence_has_a_repository_owner(self) -> None:
        repository_path = BACKEND_DIR / "mem" / "task_repository.py"

        self.assertTrue(
            repository_path.is_file(),
            "mem.task_repository should own task persistence",
        )
        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("TaskRepository(", store_source)
        self.assertNotIn("INSERT OR REPLACE INTO tasks", store_source)
        self.assertNotIn("SELECT * FROM tasks WHERE id = ?", store_source)
        self.assertNotIn("UPDATE tasks SET title=?", store_source)

    def test_writes_preserve_mutation_commit_relation_and_fts_contract(self) -> None:
        TaskRepository = _repository_type()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tasks.db"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            _create_tables(connection)
            _insert_chunk(connection, "chunk-1")
            _insert_chunk(connection, "chunk-2", created_at=2)
            connection.commit()
            times = iter([100, 200, 300, 400, 500])
            synced: list[str] = []
            repository = TaskRepository(
                connection,
                sync_fts=synced.append,
                now_ms=lambda: next(times),
            )
            task = Task(
                id="task-1",
                session_key="session-1",
                owner="owner-a",
                title="initial",
                summary="initial summary",
            )

            repository.insert(task)
            with closing(sqlite3.connect(path)) as observer:
                committed_after_insert = observer.execute(
                    "SELECT COUNT(*) FROM tasks"
                ).fetchone()[0]
            repository.finalize(
                "task-1", "final", "final summary", "skipped"
            )
            repository.assign_chunks([], "task-1")
            repository.assign_chunks(["chunk-1", "chunk-2"], "task-1")
            repository.update("task-1")
            repository.update(
                "task-1",
                boundary_summary="boundary",
                boundary_compacted_count=7,
            )
            repository.update(
                "task-1",
                title="updated",
                summary="updated summary",
                status="completed",
            )
            stored = repository.get("task-1")
            assigned_rows = connection.execute(
                "SELECT id, task_id, updated_at FROM chunks ORDER BY id"
            ).fetchall()
            with closing(sqlite3.connect(path)) as observer:
                committed_title = observer.execute(
                    "SELECT title FROM tasks WHERE id='task-1'"
                ).fetchone()[0]
            connection.close()

        self.assertEqual((task.started_at, task.updated_at), (100, 100))
        self.assertEqual(committed_after_insert, 1)
        self.assertEqual(synced, ["task-1", "task-1", "task-1"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.title, "updated")
        self.assertEqual(stored.summary, "updated summary")
        self.assertEqual(stored.boundary_summary, "boundary")
        self.assertEqual(stored.boundary_compacted_count, 7)
        self.assertEqual(stored.status, "completed")
        self.assertEqual(stored.ended_at, 200)
        self.assertEqual(stored.updated_at, 500)
        self.assertEqual(
            [(row["task_id"], row["updated_at"]) for row in assigned_rows],
            [("task-1", 300), ("task-1", 300)],
        )
        self.assertEqual(committed_title, "updated")

    def test_reads_preserve_owner_session_status_and_ordering(self) -> None:
        TaskRepository = _repository_type()
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        _create_tables(connection)
        repository = TaskRepository(
            connection,
            sync_fts=lambda _task_id: None,
            now_ms=lambda: 999,
        )
        tasks = (
            Task(id="task-1", session_key="session-1", owner="owner-a", started_at=10, updated_at=10),
            Task(id="task-2", session_key="session-2", owner="owner-a", started_at=20, updated_at=20),
            Task(id="task-3", session_key="session-1", owner="owner-b", started_at=30, updated_at=30),
            Task(id="task-4", session_key="session-1", owner="owner-a", status="completed", started_at=40, updated_at=40),
            Task(id="task-5", session_key="session-1", owner="owner-a", started_at=50, updated_at=50),
        )
        for task in tasks:
            repository.insert(task)
        _insert_chunk(connection, "chunk-1", owner="owner-a", created_at=1)
        _insert_chunk(connection, "chunk-2", owner="owner-b", created_at=2)
        _insert_chunk(
            connection,
            "chunk-3",
            owner="owner-a",
            dedup_status="duplicate",
            created_at=3,
        )
        _insert_chunk(
            connection,
            "chunk-4",
            owner="owner-a",
            task_id="task-5",
            created_at=4,
        )
        _insert_chunk(
            connection,
            "chunk-5",
            session_key="session-2",
            owner="owner-a",
            created_at=5,
        )
        connection.commit()

        active = repository.get_active("owner-a")
        active_tasks = repository.get_all_active("owner-a")
        session_active = repository.get_active_by_session(
            "session-1", "owner-a"
        )
        other_owner_active = repository.get_active_by_session(
            "session-1", "owner-b"
        )
        unassigned = repository.get_unassigned_chunks("session-1")
        owner_unassigned = repository.get_unassigned_chunks(
            "session-1", "owner-a"
        )
        missing = repository.get("missing")
        missing_active = repository.get_active("missing-owner")
        connection.close()

        self.assertEqual(active.id if active else None, "task-5")
        self.assertEqual(
            [task.id for task in active_tasks],
            ["task-1", "task-2", "task-5"],
        )
        self.assertEqual(session_active.id if session_active else None, "task-5")
        self.assertEqual(
            other_owner_active.id if other_owner_active else None,
            "task-3",
        )
        self.assertEqual(
            [chunk.id for chunk in unassigned],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(
            [chunk.id for chunk in owner_unassigned],
            ["chunk-1"],
        )
        self.assertIsNone(missing)
        self.assertIsNone(missing_active)

    def test_store_resolves_task_dependencies_dynamically(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    import tempfile
                    from pathlib import Path
                    from unittest.mock import patch
                    from mem.models import Task
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        with (
                            patch("mem.store._now_ms", return_value=123),
                            patch.object(store, "_sync_task_fts") as sync_fts,
                        ):
                            task = Task(id="task-1", session_key="session-1")
                            store.insert_task(task)
                        stored = store.get_task("task-1")
                        payload = {
                            "startedAt": stored.started_at,
                            "updatedAt": stored.updated_at,
                            "syncCalls": sync_fts.call_args_list[0].args,
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
        self.assertEqual(
            json.loads(result.stdout.splitlines()[-1]),
            {
                "startedAt": 123,
                "updatedAt": 123,
                "syncCalls": ["task-1"],
            },
        )

    def test_replace_preserves_rowid_and_removes_previous_fts_content(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    import tempfile
                    from pathlib import Path
                    from mem.models import Task
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        store.insert_task(Task(
                            id="task-1", session_key="session-1",
                            title="legacytask",
                        ))
                        before_rowid = store._conn.execute(
                            "SELECT rowid FROM tasks WHERE id='task-1'"
                        ).fetchone()[0]
                        store.insert_task(Task(
                            id="task-1", session_key="session-2",
                            title="currenttask",
                        ))
                        after_rowid = store._conn.execute(
                            "SELECT rowid FROM tasks WHERE id='task-1'"
                        ).fetchone()[0]
                        payload = {
                            "taskCount": store._conn.execute(
                                "SELECT COUNT(*) FROM tasks"
                            ).fetchone()[0],
                            "ftsCount": store._conn.execute(
                                "SELECT COUNT(*) FROM tasks_fts"
                            ).fetchone()[0],
                            "legacyMatches": store._conn.execute(
                                "SELECT COUNT(*) FROM tasks_fts "
                                "WHERE tasks_fts MATCH 'legacytask'"
                            ).fetchone()[0],
                            "currentMatches": store._conn.execute(
                                "SELECT COUNT(*) FROM tasks_fts "
                                "WHERE tasks_fts MATCH 'currenttask'"
                            ).fetchone()[0],
                            "rowidPreserved": before_rowid == after_rowid,
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
        self.assertEqual(
            json.loads(result.stdout.splitlines()[-1]),
            {
                "taskCount": 1,
                "ftsCount": 1,
                "legacyMatches": 0,
                "currentMatches": 1,
                "rowidPreserved": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
