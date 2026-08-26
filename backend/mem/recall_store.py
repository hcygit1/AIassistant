"""Read-only storage port consumed by memory recall."""

from __future__ import annotations

from typing import Protocol

from mem.models import Chunk, SearchHit, Task, TaskSearchHit


class MemRecallStore(Protocol):
    def fts_search_tasks(
        self,
        query: str,
        limit: int = 10,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        ...

    def ann_search_tasks(
        self,
        query_vec: list[float],
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        ...

    def get_task(self, task_id: str) -> Task | None:
        ...

    def fts_search_orphan_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        ...

    def ann_search_orphan_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        ...

    def fts_search_chunks_in_tasks(
        self,
        query: str,
        task_ids: list[str],
        limit: int | None = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        ...

    def exact_search_chunks_in_tasks(
        self,
        query_vec: list[float],
        task_ids: list[str],
        top_k: int | None = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        ...
