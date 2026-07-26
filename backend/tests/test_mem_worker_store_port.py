from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _worker_store_port() -> type[Any]:
    try:
        from mem.worker_store import MemWorkerStore
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.worker_store should own the worker storage port"
        ) from exc
    return MemWorkerStore


class MemWorkerStorePortTests(unittest.TestCase):
    def test_importing_worker_does_not_load_store_or_sqlite_vec(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mem.worker; "
                    "blocked = sorted(name for name in sys.modules "
                    "if name in {'mem.store', 'sqlite_vec'}); "
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

    def test_worker_store_port_contains_only_consumed_operations(self) -> None:
        MemWorkerStore = _worker_store_port()
        expected = {
            "ann_dedup_candidates",
            "find_active_chunk_by_hash",
            "get_chunks_for_embedding_retry",
            "get_chunks_for_summary_retry",
            "insert_chunk",
            "update_chunk_embedding_status",
            "update_chunk_summary",
            "upsert_chunk_embedding",
        }
        operations = {
            name
            for name, value in vars(MemWorkerStore).items()
            if not name.startswith("_") and callable(value)
        }

        self.assertTrue(getattr(MemWorkerStore, "_is_protocol", False))
        self.assertEqual(operations, expected)
        self.assertEqual(
            list(inspect.signature(MemWorkerStore.ann_dedup_candidates).parameters),
            ["self", "query_vec", "threshold", "top_k", "owner"],
        )
        self.assertEqual(
            list(inspect.signature(MemWorkerStore.update_chunk_summary).parameters),
            ["self", "chunk_id", "summary", "summary_source"],
        )

        from mem.store import MemStore

        for name in expected:
            self.assertEqual(
                str(inspect.signature(getattr(MemWorkerStore, name))),
                str(inspect.signature(getattr(MemStore, name))),
                name,
            )

    def test_worker_depends_on_port_without_changing_constructor_shape(self) -> None:
        worker_tree = ast.parse(
            (BACKEND_DIR / "mem" / "worker.py").read_text(encoding="utf-8")
        )
        imports = {
            node.module: {alias.name for alias in node.names}
            for node in ast.walk(worker_tree)
            if isinstance(node, ast.ImportFrom)
        }

        self.assertNotIn("mem.store", imports)
        self.assertIn("MemWorkerStore", imports.get("mem.worker_store", set()))

        from mem.worker import MemWorker

        signature = inspect.signature(MemWorker.__init__)
        self.assertEqual(
            list(signature.parameters),
            [
                "self",
                "store",
                "embedder",
                "llm_base_url",
                "llm_api_key",
                "llm_model",
                "dedup_threshold",
                "on_chunks_ingested",
            ],
        )
        self.assertEqual(signature.parameters["store"].annotation, "MemWorkerStore")


if __name__ == "__main__":
    unittest.main()
