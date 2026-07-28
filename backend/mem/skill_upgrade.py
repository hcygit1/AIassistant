"""Execution boundary for upgrading existing memory skills."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mem.models import Skill, Task
from mem.skill_evaluation import UpgradeEvalResult


SKILL_UPGRADE_PROMPT = """\
You are upgrading an existing skill based on new task experience.

Current skill:
Name: {SKILL_NAME}
Content:
{SKILL_CONTENT}

New task:
Title: {TITLE}
Summary: {SUMMARY}
Upgrade type: {UPGRADE_TYPE}
Merge strategy: {MERGE_STRATEGY}

Output the COMPLETE updated SKILL.md. Increment the version. \
Preserve existing quality while incorporating new insights. \
LANGUAGE RULE: Match the language of the existing skill content.

Output ONLY the complete updated SKILL.md."""


class SkillUpgradeStore(Protocol):
    def update_skill(self, skill_id: str, **fields: Any) -> None: ...

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None: ...


class SkillUpgradeEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...


class SkillUpgradeLlmCall(Protocol):
    async def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str: ...


PathProvider = Callable[[str], Any]
DescriptionExtractor = Callable[[str], str]


@dataclass(frozen=True)
class SkillUpgradeOutcome:
    upgraded: bool = False
    version: int = 0
    llm_error: Exception | None = None


def build_skill_upgrade_prompt(
    task: Task,
    skill: Skill,
    eval_result: UpgradeEvalResult,
    *,
    existing_content: str,
    prompt_template: str | None = None,
) -> str:
    template = SKILL_UPGRADE_PROMPT if prompt_template is None else prompt_template
    return (
        template
        .replace("{SKILL_NAME}", skill.name)
        .replace("{SKILL_CONTENT}", existing_content[:4000])
        .replace("{TITLE}", task.title or "")
        .replace("{SUMMARY}", (task.summary or "")[:2000])
        .replace("{UPGRADE_TYPE}", eval_result.upgrade_type)
        .replace("{MERGE_STRATEGY}", eval_result.merge_strategy)
    )


async def execute_skill_upgrade(
    task: Task,
    skill: Skill,
    eval_result: UpgradeEvalResult,
    *,
    store: SkillUpgradeStore,
    embedder: SkillUpgradeEmbedder,
    llm_call: SkillUpgradeLlmCall,
    extract_description: DescriptionExtractor,
    path_provider: PathProvider,
    prompt_template: str | None = None,
) -> SkillUpgradeOutcome:
    existing_content = ""
    if skill.dir_path:
        skill_file = path_provider(skill.dir_path) / "SKILL.md"
        if skill_file.exists():
            existing_content = skill_file.read_text(encoding="utf-8")

    if not existing_content:
        existing_content = skill.description or ""

    prompt = build_skill_upgrade_prompt(
        task,
        skill,
        eval_result,
        existing_content=existing_content,
        prompt_template=prompt_template,
    )
    try:
        new_content = await llm_call(prompt, max_tokens=4096, temperature=0.2)
    except Exception as error:
        return SkillUpgradeOutcome(llm_error=error)

    if not new_content or len(new_content.strip()) < 50:
        return SkillUpgradeOutcome()

    new_version = skill.version + 1
    new_description = extract_description(new_content) or skill.description

    if skill.dir_path:
        path_provider(skill.dir_path).mkdir(parents=True, exist_ok=True)
        (path_provider(skill.dir_path) / "SKILL.md").write_text(
            new_content,
            encoding="utf-8",
        )

    store.update_skill(
        skill.id,
        description=new_description,
        version=new_version,
    )

    try:
        vector = await embedder.embed_query(f"{skill.name} {new_description}")
        store.upsert_skill_embedding(skill.id, vector)
    except Exception:
        pass

    return SkillUpgradeOutcome(upgraded=True, version=new_version)
