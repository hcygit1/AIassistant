"""File artifacts and domain models for newly generated skills."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from mem.models import Skill, Task


PathProvider = Callable[[str], Any]
SkillFactory = Callable[..., Skill]


def extract_skill_description(skill_content: str) -> str:
    match = re.search(r'description:\s*"([^"]+)"', skill_content)
    if match:
        return match.group(1).strip()
    match = re.search(r"description:\s*(.+)", skill_content)
    if match:
        return match.group(1).strip().strip('"').strip("'")
    return ""


def write_skill_file(
    skill_store_dir: str,
    name: str,
    content: str,
    *,
    path_provider: PathProvider,
) -> str:
    if not skill_store_dir:
        return ""
    skill_dir = str(path_provider(skill_store_dir) / name)
    path_provider(skill_dir).mkdir(parents=True, exist_ok=True)
    (path_provider(skill_dir) / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def build_new_skill(
    *,
    skill_id: str,
    name: str,
    description: str,
    skill_dir: str,
    task: Task,
    quality_score: float,
    skill_factory: SkillFactory | None = None,
) -> Skill:
    factory = Skill if skill_factory is None else skill_factory
    return factory(
        id=skill_id,
        name=name,
        description=description,
        dir_path=skill_dir,
        version=1,
        status="active" if quality_score >= 6.0 else "draft",
        installed=0,
        owner=task.owner or "agent:main",
        visibility="private",
        quality_score=quality_score,
    )
