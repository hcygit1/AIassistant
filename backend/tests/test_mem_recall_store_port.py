from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, get_type_hints


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _recall_store_port() -> type[Any]:
    try:
        from mem.recall_store import MemRecallStore
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.recall_store should own the recall storage port"
        ) from exc
    return MemRecallStore


class MemRecallStorePortTests(unittest.TestCase):
    def test_importing_recall_does_not_load_store_or_sqlite_vec(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mem.recall; "
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

    def test_recall_store_port_matches_consumed_store_operations(self) -> None:
        MemRecallStore = _recall_store_port()
        expected = {
            "exact_search_chunks_in_tasks",
            "ann_search_orphan_chunks",
            "ann_search_tasks",
            "fts_search_chunks_in_tasks",
            "fts_search_orphan_chunks",
            "fts_search_tasks",
            "get_chunk",
            "get_task",
        }
        operations = {
            name
            for name, value in vars(MemRecallStore).items()
            if not name.startswith("_") and callable(value)
        }

        self.assertTrue(getattr(MemRecallStore, "_is_protocol", False))
        self.assertEqual(operations, expected)

        from mem.store import MemStore

        for name in expected:
            self.assertEqual(
                str(inspect.signature(getattr(MemRecallStore, name))),
                str(inspect.signature(getattr(MemStore, name))),
                name,
            )

    def test_recall_depends_on_port_without_constructor_changes(self) -> None:
        recall_tree = ast.parse(
            (BACKEND_DIR / "mem" / "recall.py").read_text(encoding="utf-8")
        )
        imports = {
            node.module: {alias.name for alias in node.names}
            for node in ast.walk(recall_tree)
            if isinstance(node, ast.ImportFrom)
        }

        self.assertNotIn("mem.store", imports)
        self.assertIn("MemRecallStore", imports.get("mem.recall_store", set()))

        from mem.recall import MemRecall

        signature = inspect.signature(MemRecall.__init__)
        self.assertEqual(
            list(signature.parameters),
            ["self", "store", "embedder", "config", "agent_id"],
        )
        self.assertEqual(signature.parameters["store"].annotation, "MemRecallStore")

    def test_factory_store_annotation_resolves_to_port(self) -> None:
        MemRecallStore = _recall_store_port()
        from mem.recall import MemRecall

        hints = get_type_hints(MemRecall.from_config)

        self.assertIs(hints["store"], MemRecallStore)


if __name__ == "__main__":
    unittest.main()
