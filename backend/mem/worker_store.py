"""Storage port consumed by the memory ingestion worker."""

from __future__ import annotations

from typing import Protocol

from mem.models import Chunk, EmbeddingStatus, SearchHit, SummarySource


class MemWorkerStore(Protocol):
    def find_active_chunk_by_hash(self, content: str, owner: str) -> str | None:
        ...

    def ann_dedup_candidates(
        self,
        query_vec: list[float],
        threshold: float,
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[SearchHit]:
        ...

    def insert_chunk(self, chunk: Chunk) -> None:
        ...

    def upsert_chunk_embedding(self, chunk_id: str, vec: list[float]) -> None:
        ...

    def get_chunks_for_summary_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        ...

    def update_chunk_summary(
        self,
        chunk_id: str,
        summary: str,
        *,
        summary_source: SummarySource | None = None,
    ) -> None:
        ...

    def get_chunks_for_embedding_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        ...

    def update_chunk_embedding_status(
        self,
        chunk_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> None:
        ...
