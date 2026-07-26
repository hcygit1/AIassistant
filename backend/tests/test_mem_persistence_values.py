from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _persistence_values() -> tuple[Any, Any]:
    try:
        from mem.persistence_values import content_hash, now_ms
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.persistence_values should own persisted value generation"
        ) from exc
    return content_hash, now_ms


class _WorkerStore:
    def __init__(self) -> None:
        self.inserted: dict[str, Any] = {}

    def insert_chunk(self, chunk: Any) -> None:
        self.inserted[chunk.id] = chunk


class MemoryPersistenceValueTests(unittest.IsolatedAsyncioTestCase):
    def test_importing_values_does_not_load_storage_or_pipeline(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mem.persistence_values; "
                    "blocked = sorted(name for name in sys.modules "
                    "if name in {'mem.store', 'mem.worker', 'mem.recall', "
                    "'mem.task_processor', 'mem.skill_evolver'}); "
                    "assert not blocked, blocked"
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_values_have_neutral_ownership_and_store_compatibility_aliases(
        self,
    ) -> None:
        values_path = BACKEND_DIR / "mem" / "persistence_values.py"
        self.assertTrue(
            values_path.is_file(),
            "mem.persistence_values should own persisted value generation",
        )

        worker_tree = ast.parse(
            (BACKEND_DIR / "mem" / "worker.py").read_text(encoding="utf-8")
        )
        store_imports = [
            node
            for node in ast.walk(worker_tree)
            if isinstance(node, ast.ImportFrom) and node.module == "mem.store"
        ]
        self.assertEqual(len(store_imports), 1)
        self.assertEqual([alias.name for alias in store_imports[0].names], ["MemStore"])

        content_hash, now_ms = _persistence_values()
        from mem.store import _content_hash, _now_ms

        self.assertIs(_content_hash, content_hash)
        self.assertIs(_now_ms, now_ms)

    def test_values_preserve_utf8_hash_and_millisecond_truncation(self) -> None:
        content_hash, now_ms = _persistence_values()
        content = "数据库端口是 6432"

        with patch("mem.persistence_values.time.time", return_value=1.2349):
            timestamp = now_ms()

        self.assertEqual(
            content_hash(content),
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(timestamp, 1234)

    async def test_worker_resolves_default_metadata_dependencies_at_store_time(
        self,
    ) -> None:
        from mem.worker import IngestMessage, MemWorker, PreparedChunk

        store = _WorkerStore()
        worker = MemWorker(store=store, embedder=None)  # type: ignore[arg-type]
        prepared = PreparedChunk(
            msg=IngestMessage(
                role="user",
                content="ignored original",
                session_key="session-1",
                turn_id="turn-1",
            ),
            chunk_id="chunk-1",
            kind="paragraph",
            content="normalized content",
            summary="summary",
            summary_source="fallback",
        )

        with (
            patch("mem.worker.now_ms", return_value=1234, create=True) as clock,
            patch(
                "mem.worker.content_hash",
                return_value="patched-hash",
                create=True,
            ) as hasher,
        ):
            await worker._store_prepared(prepared)

        chunk = store.inserted["chunk-1"]
        self.assertEqual((chunk.created_at, chunk.updated_at), (1234, 1234))
        self.assertEqual(chunk.content_hash, "patched-hash")
        clock.assert_called_once_with()
        hasher.assert_called_once_with("normalized content")

    async def test_worker_explicit_timestamp_does_not_read_default_clock(self) -> None:
        from mem.worker import IngestMessage, MemWorker, PreparedChunk

        store = _WorkerStore()
        worker = MemWorker(store=store, embedder=None)  # type: ignore[arg-type]
        prepared = PreparedChunk(
            msg=IngestMessage(
                role="assistant",
                content="content",
                session_key="session-1",
                turn_id="turn-1",
                timestamp=4321,
            ),
            chunk_id="chunk-2",
            kind="paragraph",
            content="content",
            summary="summary",
            summary_source="fallback",
        )

        with patch(
            "mem.worker.now_ms",
            side_effect=AssertionError("explicit timestamp should bypass the clock"),
            create=True,
        ):
            await worker._store_prepared(prepared)

        chunk = store.inserted["chunk-2"]
        self.assertEqual((chunk.created_at, chunk.updated_at), (4321, 4321))


if __name__ == "__main__":
    unittest.main()
