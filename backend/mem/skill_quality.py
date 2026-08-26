"""Quality scoring for generated memory skills."""

from __future__ import annotations

import re
from collections.abc import Callable
from string import Template
from typing import Any, Protocol

from mem.models import Task


SKILL_QUALITY_PROMPT = (
    "You are a strict skill quality reviewer. Evaluate whether this skill is high-quality enough "
    "to be activated by default.\n\n"
    "Score the skill using these rubric dimensions:\n"
    "1. workflowFocused: Is it a SINGLE clear workflow rather than a mixed/broad topic?\n"
    "2. clearTrigger: Does it clearly say what it does and when to use it?\n"
    "3. actionableSteps: Are the steps concrete and executable?\n"
    "4. verifiedOnly: Does it avoid speculative or unverified advice?\n"
    "5. hasVerification: Does it explain how to confirm success?\n"
    "6. specificEnough: Does it preserve important commands, paths, configs, errors, constraints, or versions when needed?\n\n"
    "Scoring rules:\n"
    "- If workflowFocused is false, score must be below 6\n"
    "- If clearTrigger is false, score must be below 6\n"
    "- If actionableSteps is false, score must be below 6\n"
    "- If verification is completely missing, penalize heavily\n"
    "- If the skill invents unverified alternatives or broad generic advice, penalize heavily\n"
    "- Only give 6 or above if the skill is genuinely reusable, focused, and operational\n\n"
    "Task title: $TITLE\n\n"
    "Skill content (first 2500 chars):\n$CONTENT\n\n"
    "Reply in JSON only:\n"
    '{"workflowFocused": true, "clearTrigger": true, "actionableSteps": true, '
    '"verifiedOnly": true, "hasVerification": true, "specificEnough": true, '
    '"score": 0.0, "reason": "brief explanation"}'
)


class SkillQualityLlmCall(Protocol):
    async def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str: ...


JsonParser = Callable[[str, dict[str, Any]], dict[str, Any]]


async def score_skill_quality(
    content: str,
    task: Task,
    *,
    llm_call: SkillQualityLlmCall,
    parse_json: JsonParser,
) -> float:
    prompt = Template(SKILL_QUALITY_PROMPT).substitute(
        TITLE=str(task.title),
        CONTENT=content[:2500],
    )
    try:
        raw = await llm_call(prompt, max_tokens=512, temperature=0)
        parsed = parse_json(raw, {})
        score = parsed.get("score")
        if isinstance(score, (int, float)):
            return min(10.0, max(0.0, float(score)))
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if match:
            return min(10.0, max(0.0, float(match.group(1))))
    except Exception:
        pass
    return 5.0
