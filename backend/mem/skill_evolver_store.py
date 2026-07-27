"""Storage port consumed by the memory skill evolver."""

from __future__ import annotations

from typing import Any, Protocol

from mem.models import Chunk, Skill, SkillSearchHit


class MemSkillEvolverStore(Protocol):
    def get_chunks_by_task(
        self,
        task_id: str,
        limit: int | None = None,
    ) -> list[Chunk]:
        ...

    def fts_search_skills(
        self,
        query: str,
        limit: int = 10,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        ...

    def ann_search_skills(
        self,
        query_vec: list[float],
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        ...

    def get_skill(self, skill_id: str) -> Skill | None:
        ...

    def insert_skill(self, skill: Skill) -> None:
        ...

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
        ...

    def update_skill(self, skill_id: str, **fields: Any) -> None:
        ...
