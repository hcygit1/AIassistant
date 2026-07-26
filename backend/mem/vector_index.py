"""Write boundary for sqlite-vec memory indexes."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable


class MemoryVectorIndex:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        serialize_vector: Callable[[list[float]], bytes],
    ) -> None:
        self._connection = connection
        self._serialize_vector = serialize_vector

    def upsert_chunk(self, chunk_id: str, vector: list[float]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, self._serialize_vector(vector)),
        )
        self._connection.commit()

    def delete_chunk(self, chunk_id: str) -> None:
        self._connection.execute(
            "DELETE FROM vec_chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        self._connection.commit()

    def upsert_task(self, task_id: str, vector: list[float]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO vec_tasks(task_id, embedding) VALUES (?, ?)",
            (task_id, self._serialize_vector(vector)),
        )
        self._connection.commit()

    def upsert_skill(self, skill_id: str, vector: list[float]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO vec_skills(skill_id, embedding) VALUES (?, ?)",
            (skill_id, self._serialize_vector(vector)),
        )
        self._connection.commit()
