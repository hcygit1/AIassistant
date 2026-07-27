"""LLM content generation for new memory skills."""

from __future__ import annotations

from typing import Protocol

from mem.models import Task
from mem.skill_evaluation import CreateEvalResult


SKILL_GENERATE_PROMPT = """\
You are a Skill creation expert. Distill the following completed task into a reusable SKILL.md.

Core principles:
- The skill must capture ONE coherent workflow only
- Description is the trigger mechanism: it must say what the skill does and when to use it
- Body < 400 lines, focused
- Use imperative form, explain WHY not just HOW
- Generalize from the specific task; keep verified commands/code
- Include only procedures that were actually validated in the task record
- Do not add unverified alternatives, speculative advice, or broad background explanation
- LANGUAGE RULE: Write in the SAME language as the user's messages in the task record.
  "name" field uses English kebab-case; everything else matches user's language.

Output format:
---
name: "{NAME}"
description: "..."
metadata: {{ "openclaw": {{ "emoji": "..." }} }}
---

# Title

## What this skill does
(1-2 short paragraphs describing the workflow outcome)

## When to use this skill
(2-4 bullet points)

## Prerequisites
(required environment, access, dependencies, assumptions; write "None" if not needed)

## Steps
(Numbered steps with reasoning)

## Verification
(how to confirm the workflow succeeded)

## Pitfalls and solutions
(What went wrong + fix)

Task title: {TITLE}
Task summary:
{SUMMARY}

Original goal:
{ORIGINAL_GOAL}

Key evidence:
{EVIDENCE}

Output ONLY the complete SKILL.md content."""


class SkillGenerationLlmCall(Protocol):
    async def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str: ...


def build_skill_generation_prompt(
    task: Task,
    eval_result: CreateEvalResult,
    *,
    original_goal: str,
    evidence: str,
    prompt_template: str | None = None,
) -> str:
    template = SKILL_GENERATE_PROMPT if prompt_template is None else prompt_template
    return (
        template
        .replace("{NAME}", eval_result.suggested_name)
        .replace("{TITLE}", task.title or "")
        .replace("{SUMMARY}", (task.summary or "")[:2000])
        .replace("{ORIGINAL_GOAL}", original_goal[:1200])
        .replace("{EVIDENCE}", evidence[:8000])
    )


async def generate_skill_content(
    prompt: str,
    *,
    llm_call: SkillGenerationLlmCall,
) -> str:
    return await llm_call(prompt, max_tokens=4096, temperature=0.2)
