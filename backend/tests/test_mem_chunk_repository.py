from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Chunk


def _create_chunks_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE chunks (
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
        )"""
    )


def _repository_type() -> type[Any]:
    try:
        from mem.chunk_repository import ChunkRepository
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.chunk_repository should own chunk persistence"
        ) from exc
    return ChunkRepository


class ChunkRepositoryTests(unittest.TestCase):
    def test_chunk_persistence_has_a_repository_owner(self) -> None:
        repository_path = BACKEND_DIR / "mem" / "chunk_repository.py"

        self.assertTrue(
            repository_path.is_file(),
            "mem.chunk_repository should own chunk persistence",
        )
        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ChunkRepository(", store_source)
        self.assertNotIn("INSERT OR REPLACE INTO chunks", store_source)
        self.assertNotIn("UPDATE chunks SET summary = ?", store_source)
        self.assertNotIn("SELECT * FROM chunks WHERE id = ?", store_source)

    def test_writes_preserve_mutation_commit_status_and_fts_contract(self) -> None:
        ChunkRepository = _repository_type()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "chunks.db"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            _create_chunks_table(connection)
            times = iter([100, 200, 300, 400, 500, 600])
            synced: list[str] = []
            hashed: list[str] = []

            def content_hash(content: str) -> str:
                hashed.append(content)
                return f"hash:{content}"

            repository = ChunkRepository(
                connection,
                sync_fts=synced.append,
                now_ms=lambda: next(times),
                content_hash=content_hash,
            )
            chunk = Chunk(
                id="chunk-1",
                session_key="session-1",
                turn_id="turn-1",
                seq=0,
                role="user",
                content="content",
                task_id="task-1",
                owner="owner-1",
                summary_source="fallback",
            )

            repository.insert(chunk)
            with sqlite3.connect(path) as observer:
                committed_after_insert = observer.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
            active_hash = repository.find_active_by_hash("content", "owner-1")
            wrong_owner_hash = repository.find_active_by_hash("content", "owner-2")
            repository.update_summary("chunk-1", "first")
            repository.update_summary(
                "chunk-1", "second", summary_source="llm"
            )
            repository.update_embedding_status("chunk-1", "failed", "network")
            repository.mark_dedup_status(
                "chunk-1", "duplicate", "chunk-0", "same"
            )
            repository.orphan("chunk-1", "task_skipped")
            stored = repository.get("chunk-1")
            with sqlite3.connect(path) as observer:
                committed_updated_at = observer.execute(
                    "SELECT updated_at FROM chunks WHERE id='chunk-1'"
                ).fetchone()[0]
            connection.close()

        self.assertEqual(chunk.content_hash, "hash:content")
        self.assertEqual((chunk.created_at, chunk.updated_at), (100, 100))
        self.assertEqual(committed_after_insert, 1)
        self.assertEqual(active_hash, "chunk-1")
        self.assertIsNone(wrong_owner_hash)
        self.assertEqual(hashed, ["content", "content", "content"])
        self.assertEqual(synced, ["chunk-1", "chunk-1", "chunk-1"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.summary, "second")
        self.assertEqual(stored.summary_source, "llm")
        self.assertEqual(stored.embedding_status, "failed")
        self.assertEqual(stored.embedding_error, "network")
        self.assertEqual(stored.dedup_status, "orphaned")
        self.assertIsNone(stored.task_id)
        self.assertEqual(stored.dedup_target, "chunk-0")
        self.assertEqual(stored.dedup_reason, "task_skipped")
        self.assertEqual(stored.updated_at, 600)
        self.assertEqual(committed_updated_at, 600)

    def test_reads_preserve_active_retry_limit_owner_and_timeline_rules(self) -> None:
        ChunkRepository = _repository_type()
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        _create_chunks_table(connection)
        repository = ChunkRepository(
            connection,
            sync_fts=lambda _chunk_id: None,
            now_ms=lambda: 999,
            content_hash=lambda content: f"hash:{content}",
        )
        statuses = (
            ("ok", "llm", "active", "owner-a"),
            ("failed", "fallback", "active", "owner-a"),
            ("failed", "fallback", "active", "owner-b"),
            ("failed", "fallback", "duplicate", "owner-a"),
            ("failed", "fallback", "active", "owner-a"),
            ("ok", "llm", "active", "owner-a"),
            ("failed", "fallback", "active", "owner-a"),
            ("failed", "fallback", "active", "owner-a"),
        )
        for index, (embedding, source, dedup, owner) in enumerate(statuses):
            repository.insert(Chunk(
                id=f"chunk-{index}",
                session_key="session-1",
                turn_id=f"turn-{index}",
                seq=index,
                role="user" if index % 2 == 0 else "assistant",
                content=f"content-{index}",
                task_id="task-1",
                owner=owner,
                dedup_status=dedup,
                summary_source=source,
                embedding_status=embedding,
                created_at=index * 10 + 1,
                updated_at=index * 10 + 1,
            ))

        task_chunks = repository.get_by_task("task-1")
        limited = repository.get_by_task("task-1", limit=2)
        embedding_retry = repository.get_for_embedding_retry(limit=3)
        owner_embedding_retry = repository.get_for_embedding_retry(
            owner="owner-a", limit=10
        )
        summary_retry = repository.get_for_summary_retry(limit=3)
        owner_summary_retry = repository.get_for_summary_retry(
            owner="owner-b", limit=10
        )
        timeline = repository.get_in_range(
            "session-1", "turn-4", 4, window=1
        )
        owner_timeline = repository.get_in_range(
            "session-1", "turn-4", 4, window=1, owner="owner-a"
        )
        missing_timeline = repository.get_in_range(
            "session-1", "turn-4", 4, owner="owner-b"
        )
        missing = repository.get("missing")
        connection.close()

        self.assertEqual(
            [chunk.id for chunk in task_chunks],
            ["chunk-0", "chunk-1", "chunk-2", "chunk-4", "chunk-5", "chunk-6", "chunk-7"],
        )
        self.assertEqual([chunk.id for chunk in limited], ["chunk-0", "chunk-1"])
        self.assertEqual(
            [chunk.id for chunk in embedding_retry],
            ["chunk-1", "chunk-2", "chunk-4"],
        )
        self.assertEqual(
            [chunk.id for chunk in owner_embedding_retry],
            ["chunk-1", "chunk-4", "chunk-6", "chunk-7"],
        )
        self.assertEqual(
            [chunk.id for chunk in summary_retry],
            ["chunk-1", "chunk-2", "chunk-4"],
        )
        self.assertEqual(
            [chunk.id for chunk in owner_summary_retry],
            ["chunk-2"],
        )
        self.assertEqual(
            [chunk.id for chunk in timeline],
            ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5", "chunk-6", "chunk-7"],
        )
        self.assertEqual(
            [chunk.id for chunk in owner_timeline],
            ["chunk-0", "chunk-1", "chunk-3", "chunk-4", "chunk-5", "chunk-6", "chunk-7"],
        )
        self.assertEqual(missing_timeline, [])
        self.assertIsNone(missing)

    def test_store_resolves_chunk_dependencies_dynamically(self) -> None:
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
                    from mem.models import Chunk
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        with (
                            patch("mem.store._now_ms", return_value=123),
                            patch("mem.store._content_hash", return_value="patched-hash"),
                            patch.object(store, "_sync_chunk_fts") as sync_fts,
                        ):
                            chunk = Chunk(
                                id="chunk-1", session_key="session-1",
                                turn_id="turn-1", seq=0, role="user",
                                content="content",
                            )
                            store.insert_chunk(chunk)
                        stored = store.get_chunk("chunk-1")
                        payload = {
                            "contentHash": stored.content_hash,
                            "createdAt": stored.created_at,
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
                "contentHash": "patched-hash",
                "createdAt": 123,
                "updatedAt": 123,
                "syncCalls": ["chunk-1"],
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
                    from mem.models import Chunk
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        store.insert_chunk(Chunk(
                            id="chunk-1", session_key="session-1",
                            turn_id="turn-1", seq=0, role="user",
                            content="legacyterm",
                        ))
                        before_rowid = store._conn.execute(
                            "SELECT rowid FROM chunks WHERE id='chunk-1'"
                        ).fetchone()[0]
                        store.insert_chunk(Chunk(
                            id="chunk-1", session_key="session-1",
                            turn_id="turn-2", seq=1, role="assistant",
                            content="currentterm",
                        ))
                        after_rowid = store._conn.execute(
                            "SELECT rowid FROM chunks WHERE id='chunk-1'"
                        ).fetchone()[0]
                        payload = {
                            "chunkCount": store._conn.execute(
                                "SELECT COUNT(*) FROM chunks"
                            ).fetchone()[0],
                            "ftsCount": store._conn.execute(
                                "SELECT COUNT(*) FROM chunks_fts"
                            ).fetchone()[0],
                            "legacyMatches": store._conn.execute(
                                "SELECT COUNT(*) FROM chunks_fts "
                                "WHERE chunks_fts MATCH 'legacyterm'"
                            ).fetchone()[0],
                            "currentMatches": store._conn.execute(
                                "SELECT COUNT(*) FROM chunks_fts "
                                "WHERE chunks_fts MATCH 'currentterm'"
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
                "chunkCount": 1,
                "ftsCount": 1,
                "legacyMatches": 0,
                "currentMatches": 1,
                "rowidPreserved": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
