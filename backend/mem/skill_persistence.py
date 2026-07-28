"""Persistence boundary for newly generated memory skills."""

from __future__ import annotations

from typing import Protocol

from mem.models import Skill


class SkillPersistenceStore(Protocol):
    def insert_skill(self, skill: Skill) -> None: ...

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None: ...


class SkillPersistenceEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...


async def persist_new_skill(
    skill: Skill,
    *,
    store: SkillPersistenceStore,
    embedder: SkillPersistenceEmbedder,
) -> Exception | None:
    store.insert_skill(skill)
    try:
        vector = await embedder.embed_query(f"{skill.name} {skill.description}")
        store.upsert_skill_embedding(skill.id, vector)
    except Exception as error:
        return error
    return None
