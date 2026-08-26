"""LLM evaluation for creating and upgrading memory skills."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mem.models import Skill, Task


@dataclass
class CreateEvalResult:
    should_generate: bool = False
    reason: str = ""
    suggested_name: str = ""
    suggested_tags: list[str] | None = None
    confidence: float = 0.0


@dataclass
class UpgradeEvalResult:
    should_upgrade: bool = False
    upgrade_type: str = "refine"
    dimensions: list[str] | None = None
    reason: str = ""
    merge_strategy: str = ""
    confidence: float = 0.0


CREATE_EVAL_PROMPT = """\
You are a strict experience evaluation expert. Based on the completed task record below, \
decide whether this task contains **reusable, transferable** experience worth distilling into a "skill".

STRICT criteria — must meet ALL of:
1. **Repeatable**: The task type is likely to recur
2. **Transferable**: The approach would help others facing the same problem
3. **Technical depth**: Contains non-trivial steps, commands, code, configs, or diagnostic reasoning
4. **Single workflow**: The task can be expressed as one coherent workflow, not a grab bag of unrelated operations

NOT worth distilling:
- Pure factual Q&A, casual chat, opinion discussion
- Single-turn simple answers with no workflow
- One-off personal tasks, organizing personal information
- Simple information lookup or summarization
- Tasks that mix multiple independent goals or unrelated workflows
- Tasks that are mostly project-specific status updates rather than reusable procedure

Task title: {TITLE}
Task summary:
{SUMMARY}

LANGUAGE RULE: "reason" MUST use the SAME language as the task title/summary. \
"suggestedName" stays in English kebab-case.

Naming rules for "suggestedName":
- Must name a concrete workflow or task type, not a broad domain
- Prefer action-oriented names like debugging-x, recovering-y, migrating-z
- Avoid vague names like database-fix, backend-help, troubleshooting

Reply in JSON only:
{{"shouldGenerate": boolean, "reason": "brief explanation", "suggestedName": "kebab-case", \
"suggestedTags": ["tag1"], "confidence": 0.0-1.0}}"""


UPGRADE_EVAL_PROMPT = """\
Existing skill (v{VERSION}):
Name: {SKILL_NAME}
Description: {SKILL_DESC}

Newly completed task:
Title: {TITLE}
Summary:
{SUMMARY}

Does the new task bring substantive improvements to the existing skill?

Worth upgrading only if the new task adds at least one of:
- a newly verified step or workflow branch
- a correction to an incorrect or outdated instruction
- a clearly new supported scenario within the SAME workflow
- an important new pitfall, constraint, prerequisite, or verification step

NOT worth upgrading:
- identical content
- same workflow but just phrased differently
- broader domain overlap without the same workflow
- speculative improvements that were not actually validated in the task
- cosmetic wording changes

Reply in JSON only:
{{"shouldUpgrade": boolean, "upgradeType": "refine"|"extend"|"fix", \
"dimensions": ["..."], "reason": "...", "mergeStrategy": "...", "confidence": 0.0-1.0}}"""


class SkillEvaluationLlmCall(Protocol):
    async def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str: ...


JsonParser = Callable[[str, dict[str, Any]], dict[str, Any]]


def _field(parsed: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in parsed:
            return parsed[name]
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_float(value: Any) -> float:
    return float(value)


async def evaluate_skill_creation(
    task: Task,
    *,
    llm_call: SkillEvaluationLlmCall,
    parse_json: JsonParser,
) -> CreateEvalResult:
    prompt = (
        CREATE_EVAL_PROMPT
        .replace("{TITLE}", task.title or "")
        .replace("{SUMMARY}", (task.summary or "")[:3000])
    )
    raw = await llm_call(prompt, max_tokens=1024, temperature=0)
    parsed = parse_json(raw, {})
    return CreateEvalResult(
        should_generate=_as_bool(_field(parsed, "shouldGenerate", "should_generate", default=False)),
        reason=str(parsed.get("reason", "")),
        suggested_name=str(_field(parsed, "suggestedName", "suggested_name", default="")),
        suggested_tags=_field(parsed, "suggestedTags", "suggested_tags"),
        confidence=_as_float(parsed.get("confidence", 0)),
    )


async def evaluate_skill_upgrade(
    task: Task,
    skill: Skill,
    *,
    llm_call: SkillEvaluationLlmCall,
    parse_json: JsonParser,
) -> UpgradeEvalResult:
    prompt = (
        UPGRADE_EVAL_PROMPT
        .replace("{VERSION}", str(skill.version))
        .replace("{SKILL_NAME}", skill.name)
        .replace("{SKILL_DESC}", (skill.description or "")[:1000])
        .replace("{TITLE}", task.title or "")
        .replace("{SUMMARY}", (task.summary or "")[:3000])
    )
    raw = await llm_call(prompt, max_tokens=1024, temperature=0)
    parsed = parse_json(raw, {})
    return UpgradeEvalResult(
        should_upgrade=_as_bool(_field(parsed, "shouldUpgrade", "should_upgrade", default=False)),
        upgrade_type=str(_field(parsed, "upgradeType", "upgrade_type", default="refine")),
        dimensions=parsed.get("dimensions"),
        reason=str(parsed.get("reason", "")),
        merge_strategy=str(_field(parsed, "mergeStrategy", "merge_strategy", default="")),
        confidence=_as_float(parsed.get("confidence", 0)),
    )
