"""Related skill retrieval and LLM selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from mem.models import Skill, Task
from mem.skill_evolver_store import MemSkillEvolverStore


RELATED_SKILL_JUDGE_PROMPT = """\
Decide whether a completed TASK should be merged into an EXISTING SKILL. \
The task and skill must represent the SAME workflow, not merely the same broad domain/topic.

TASK TITLE: {TASK_TITLE}
TASK SUMMARY:
{TASK_SUMMARY}

CANDIDATE SKILLS:
{SKILL_LIST}

Rules:
- Output ONE skill index (1 to {N}) ONLY if it is clearly the same workflow/task type
- Broad domain overlap is NOT enough
- If the task would make the skill broader or more mixed, output 0
- Output 0 if none is highly relevant. When in doubt, output 0

Reply JSON only: {{"selectedIndex": 0, "reason": "..."}}"""


class SkillRelationEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...


class SkillRelationJudge(Protocol):
    async def __call__(
        self,
        task: Task,
        candidates: list[Skill],
    ) -> Skill | None: ...


class SkillRelationLlmCall(Protocol):
    async def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str: ...


JsonParser = Callable[[str, dict[str, Any]], dict[str, Any]]


async def find_related_skill(
    task: Task,
    *,
    store: MemSkillEvolverStore,
    embedder: SkillRelationEmbedder,
    judge_related: SkillRelationJudge,
) -> Skill | None:
    query = (task.summary or "")[:600]
    if not query.strip():
        return None

    try:
        fts_hits = store.fts_search_skills(query, limit=10, owner=task.owner)
    except Exception:
        fts_hits = []

    vec_hits: list[tuple[str, float]] = []
    try:
        query_vector = await embedder.embed_query(query)
        ann_hits = store.ann_search_skills(
            query_vector,
            top_k=10,
            owner=task.owner,
        )
        vec_hits = [(hit.skill_id, hit.score) for hit in ann_hits]
    except Exception:
        pass

    candidate_ids: set[str] = set()
    for hit in fts_hits:
        candidate_ids.add(hit.skill_id)
    for skill_id, _score in vec_hits:
        candidate_ids.add(skill_id)

    if not candidate_ids:
        return None

    candidates: list[Skill] = []
    for skill_id in candidate_ids:
        skill = store.get_skill(skill_id)
        if (
            skill
            and skill.status == "active"
            and skill.owner == (task.owner or "agent:main")
        ):
            candidates.append(skill)

    if not candidates:
        return None

    return await judge_related(task, candidates)


async def judge_related_skill(
    task: Task,
    candidates: list[Skill],
    *,
    llm_call: SkillRelationLlmCall,
    parse_json: JsonParser,
) -> Skill | None:
    skill_list = "\n\n".join(
        f"{index + 1}. [{skill.name}]\n   {(skill.description or '')[:300]}"
        for index, skill in enumerate(candidates)
    )
    prompt = (
        RELATED_SKILL_JUDGE_PROMPT
        .replace("{TASK_TITLE}", task.title or "(no title)")
        .replace("{TASK_SUMMARY}", (task.summary or "")[:800])
        .replace("{SKILL_LIST}", skill_list)
        .replace("{N}", str(len(candidates)))
    )
    raw = await llm_call(prompt, max_tokens=256, temperature=0)
    parsed = parse_json(raw, {"selectedIndex": 0, "reason": ""})
    selected_index = parsed.get("selectedIndex", 0)
    if (
        isinstance(selected_index, (int, float))
        and 1 <= int(selected_index) <= len(candidates)
    ):
        return candidates[int(selected_index) - 1]
    return None
