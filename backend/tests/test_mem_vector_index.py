from __future__ import annotations

import ast
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


def _vector_index_type() -> type[Any]:
    try:
        from mem.vector_index import MemoryVectorIndex
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.vector_index should own sqlite-vec writes"
        ) from exc
    return MemoryVectorIndex


def _create_vector_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE vec_chunks (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        );
        CREATE TABLE vec_tasks (
            task_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        );
        CREATE TABLE vec_skills (
            skill_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        );
        """
    )
    connection.commit()


class MemoryVectorIndexTests(unittest.TestCase):
    def test_vector_writes_have_an_explicit_owner(self) -> None:
        vector_index_path = BACKEND_DIR / "mem" / "vector_index.py"
        self.assertTrue(
            vector_index_path.is_file(),
            "mem.vector_index should own sqlite-vec writes",
        )

        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MemoryVectorIndex(", store_source)
        self.assertNotIn("INSERT OR REPLACE INTO vec_chunks", store_source)
        self.assertNotIn("DELETE FROM vec_chunks", store_source)
        self.assertNotIn("INSERT OR REPLACE INTO vec_tasks", store_source)
        self.assertNotIn("INSERT OR REPLACE INTO vec_skills", store_source)

        vector_index_source = vector_index_path.read_text(encoding="utf-8")
        self.assertIn("INSERT OR REPLACE INTO vec_chunks", vector_index_source)
        self.assertIn("DELETE FROM vec_chunks", vector_index_source)
        self.assertIn("INSERT OR REPLACE INTO vec_tasks", vector_index_source)
        self.assertIn("INSERT OR REPLACE INTO vec_skills", vector_index_source)

    def test_store_keeps_embedding_method_signatures(self) -> None:
        tree = ast.parse(
            (BACKEND_DIR / "mem" / "store.py").read_text(encoding="utf-8")
        )
        store_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MemStore"
        )
        methods = {
            node.name: node
            for node in store_class.body
            if isinstance(node, ast.FunctionDef)
        }

        expected = {
            "upsert_chunk_embedding": ["self", "chunk_id", "vec"],
            "delete_chunk_embedding": ["self", "chunk_id"],
            "upsert_task_embedding": ["self", "task_id", "vec"],
            "upsert_skill_embedding": ["self", "skill_id", "vec"],
        }
        for name, arguments in expected.items():
            method = methods[name]
            self.assertEqual([arg.arg for arg in method.args.args], arguments)
            self.assertEqual(method.args.defaults, [])
            self.assertEqual(ast.unparse(method.returns), "None")

    def test_writes_replace_delete_serialize_and_commit(self) -> None:
        MemoryVectorIndex = _vector_index_type()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "vectors.db"
            connection = sqlite3.connect(path)
            _create_vector_tables(connection)
            serialized: list[list[float]] = []

            def serialize(vector: list[float]) -> bytes:
                serialized.append(list(vector))
                return json.dumps(vector).encode("ascii")

            index = MemoryVectorIndex(connection, serialize_vector=serialize)
            index.upsert_chunk("chunk-1", [1.0, 2.0, 3.0])
            index.upsert_chunk("chunk-1", [4.0, 5.0, 6.0])
            with closing(sqlite3.connect(path)) as observer:
                chunk_rows = observer.execute(
                    "SELECT chunk_id, embedding FROM vec_chunks"
                ).fetchall()

            index.delete_chunk("chunk-1")
            with closing(sqlite3.connect(path)) as observer:
                chunk_count_after_delete = observer.execute(
                    "SELECT COUNT(*) FROM vec_chunks"
                ).fetchone()[0]

            index.upsert_task("task-1", [7.0, 8.0, 9.0])
            with closing(sqlite3.connect(path)) as observer:
                task_rows = observer.execute(
                    "SELECT task_id, embedding FROM vec_tasks"
                ).fetchall()

            index.upsert_skill("skill-1", [10.0, 11.0, 12.0])
            with closing(sqlite3.connect(path)) as observer:
                skill_rows = observer.execute(
                    "SELECT skill_id, embedding FROM vec_skills"
                ).fetchall()
            connection.close()

        self.assertEqual(
            serialized,
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
        )
        self.assertEqual(chunk_rows, [("chunk-1", b"[4.0, 5.0, 6.0]")])
        self.assertEqual(chunk_count_after_delete, 0)
        self.assertEqual(task_rows, [("task-1", b"[7.0, 8.0, 9.0]")])
        self.assertEqual(skill_rows, [("skill-1", b"[10.0, 11.0, 12.0]")])

    def test_real_store_uses_dynamic_serializer_and_three_dimensional_tables(
        self,
    ) -> None:
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
                    from sqlite_vec import serialize_float32 as real_serialize
                    from mem.store import MemStore

                    calls = []

                    def serialize(vector):
                        calls.append(list(vector))
                        return real_serialize(vector)

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        with patch("mem.store.serialize_float32", side_effect=serialize):
                            store.upsert_chunk_embedding("chunk-1", [1.0, 0.0, 0.0])
                            store.upsert_task_embedding("task-1", [0.0, 1.0, 0.0])
                            store.upsert_skill_embedding("skill-1", [0.0, 0.0, 1.0])
                        before_delete = {
                            "chunks": store._conn.execute(
                                "SELECT COUNT(*) FROM vec_chunks"
                            ).fetchone()[0],
                            "tasks": store._conn.execute(
                                "SELECT COUNT(*) FROM vec_tasks"
                            ).fetchone()[0],
                            "skills": store._conn.execute(
                                "SELECT COUNT(*) FROM vec_skills"
                            ).fetchone()[0],
                        }
                        store.delete_chunk_embedding("chunk-1")
                        after_delete = store._conn.execute(
                            "SELECT COUNT(*) FROM vec_chunks"
                        ).fetchone()[0]
                        store.close()
                        print(json.dumps({
                            "afterDelete": after_delete,
                            "beforeDelete": before_delete,
                            "calls": calls,
                        }, sort_keys=True))
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
                "afterDelete": 0,
                "beforeDelete": {"chunks": 1, "skills": 1, "tasks": 1},
                "calls": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
