"""Skill 进化器 — 评估 + 生成 + 升级 + 安装

按 docs/memory-system-refactor.md §6:
  Task 完成 → 规则过滤 → LLM 评估 → 四步流水线生成 Skill
  相似 Task → 搜索 Skill → LLM 判断升级
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from mem.store import MemStore, Task, Chunk, Skill
from mem.embedder import MemEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CREATE_EVAL_PROMPT = """\
You are a strict experience evaluation expert. Based on the completed task record below, \
decide whether this task contains **reusable, transferable** experience worth distilling into a "skill".

STRICT criteria — must meet ALL of:
1. **Repeatable**: The task type is likely to recur
2. **Transferable**: The approach would help others facing the same problem
3. **Technical depth**: Contains non-trivial steps, commands, code, configs, or diagnostic reasoning

NOT worth distilling:
- Pure factual Q&A, casual chat, opinion discussion
- Single-turn simple answers with no workflow
- One-off personal tasks, organizing personal information
- Simple information lookup or summarization

Task title: {TITLE}
Task summary:
{SUMMARY}

LANGUAGE RULE: "reason" MUST use the SAME language as the task title/summary. \
"suggestedName" stays in English kebab-case.

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

Worth upgrading: faster path, more elegant, fewer dependencies, corrects errors, \
adds edge cases, covers new scenario, fixes outdated info.
NOT worth upgrading: identical, worse approach, trivial difference.

Reply in JSON only:
{{"shouldUpgrade": boolean, "upgradeType": "refine"|"extend"|"fix", \
"dimensions": ["..."], "reason": "...", "mergeStrategy": "...", "confidence": 0.0-1.0}}"""

SKILL_GENERATE_PROMPT = """\
You are a Skill creation expert. Distill the following completed task into a reusable SKILL.md.

Core principles:
- Description (~100 words) is the trigger mechanism — be proactive about when to use
- Body < 400 lines, focused
- Use imperative form, explain WHY not just HOW
- Generalize from the specific task; keep verified commands/code
- LANGUAGE RULE: Write in the SAME language as the user's messages in the task record.
  "name" field uses English kebab-case; everything else matches user's language.

Output format:
---
name: "{NAME}"
description: "..."
metadata: {{ "openclaw": {{ "emoji": "..." }} }}
---

# Title

## When to use this skill
(2-4 bullet points)

## Steps
(Numbered steps with reasoning)

## Pitfalls and solutions
(What went wrong + fix)

## Key takeaways
(3-5 bullet points)

Task title: {TITLE}
Task summary:
{SUMMARY}

Task conversation:
{CONVERSATION}

Output ONLY the complete SKILL.md content."""

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

RELATED_SKILL_JUDGE_PROMPT = """\
Decide whether a completed TASK should be merged into an EXISTING SKILL. \
The task and skill must be in the SAME domain/topic.

TASK TITLE: {TASK_TITLE}
TASK SUMMARY:
{TASK_SUMMARY}

CANDIDATE SKILLS:
{SKILL_LIST}

Rules:
- Output ONE skill index (1 to {N}) ONLY if clearly same domain
- Output 0 if none is highly relevant. When in doubt, output 0

Reply JSON only: {{"selectedIndex": 0, "reason": "..."}}"""


# ---------------------------------------------------------------------------
# MemSkillEvolver
# ---------------------------------------------------------------------------

class MemSkillEvolver:
    """Evaluates completed Tasks and generates/upgrades Skills."""

    def __init__(
        self,
        store: MemStore,
        embedder: MemEmbedder,
        *,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        skill_store_dir: str = "",
        min_chunks_for_eval: int = 6,
        min_confidence: float = 0.7,
        auto_install: bool = False,
        enabled: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.skill_store_dir = skill_store_dir
        self.min_chunks_for_eval = min_chunks_for_eval
        self.min_confidence = min_confidence
        self.auto_install = auto_install
        self.enabled = enabled
        self._lock = asyncio.Lock()
        self._queue: list[Task] = []
        self._processing = False

    # ------------------------------------------------------------------
    # Public entry — called by MemTaskProcessor
    # ------------------------------------------------------------------

    async def on_task_completed(self, task: Task) -> None:
        if not self.enabled:
            return
        async with self._lock:
            if self._processing:
                self._queue.append(task)
                return
        await self._drain(task)

    async def _drain(self, task: Task) -> None:
        self._processing = True
        try:
            await self._process_one(task)
            while self._queue:
                next_task = self._queue.pop(0)
                await self._process_one(next_task)
        finally:
            self._processing = False

    async def _process_one(self, task: Task) -> None:
        try:
            await self._process(task)
        except Exception as e:
            logger.error("SkillEvolver error for task %s: %s", task.id, e)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _process(self, task: Task) -> None:
        chunks = self.store.get_chunks_by_task(task.id)

        skip = self._rule_filter(chunks, task)
        if skip:
            logger.debug("SkillEvolver: task %s skipped by rule filter: %s", task.id, skip)
            return

        related = await self._find_related_skill(task)

        if related:
            await self._handle_existing_skill(task, chunks, related)
        else:
            await self._handle_new_skill(task, chunks)

    # ------------------------------------------------------------------
    # Rule filter (§6.2)
    # ------------------------------------------------------------------

    def _rule_filter(self, chunks: list[Chunk], task: Task) -> str | None:
        if len(chunks) < self.min_chunks_for_eval:
            return f"chunks不足 ({len(chunks)} < {self.min_chunks_for_eval})"
        if task.status == "skipped":
            return "task状态为skipped"
        if len(task.summary or "") < 100:
            return f"summary过短 ({len(task.summary or '')} < 100)"
        if not any(c.role == "user" for c in chunks):
            return "无用户消息"
        if not any(c.role == "assistant" for c in chunks):
            return "无助手回复"
        return None

    # ------------------------------------------------------------------
    # Find related existing Skill (FTS + ANN → LLM judge)
    # ------------------------------------------------------------------

    async def _find_related_skill(self, task: Task) -> Skill | None:
        query = (task.summary or "")[:600]
        if not query.strip():
            return None

        try:
            fts_hits = self.store.fts_search_skills(query, limit=10)
        except Exception:
            fts_hits = []

        vec_hits: list[tuple[str, float]] = []
        try:
            q_vec = await self.embedder.embed_query(query)
            ann = self.store.ann_search_skills(q_vec, top_k=10)
            vec_hits = [(h.skill_id, h.score) for h in ann]
        except Exception:
            pass

        candidate_ids: set[str] = set()
        for h in fts_hits:
            candidate_ids.add(h.skill_id)
        for sid, _ in vec_hits:
            candidate_ids.add(sid)

        if not candidate_ids:
            return None

        candidates: list[Skill] = []
        for sid in candidate_ids:
            skill = self.store.get_skill(sid)
            if skill and skill.status in ("active", "draft"):
                candidates.append(skill)

        if not candidates:
            return None

        return await self._judge_related(task, candidates)

    async def _judge_related(self, task: Task, candidates: list[Skill]) -> Skill | None:
        skill_list = "\n\n".join(
            f"{i+1}. [{s.name}]\n   {(s.description or '')[:300]}"
            for i, s in enumerate(candidates)
        )
        prompt = (
            RELATED_SKILL_JUDGE_PROMPT
            .replace("{TASK_TITLE}", task.title or "(no title)")
            .replace("{TASK_SUMMARY}", (task.summary or "")[:800])
            .replace("{SKILL_LIST}", skill_list)
            .replace("{N}", str(len(candidates)))
        )
        try:
            raw = await self._llm_call(prompt, max_tokens=256, temperature=0)
            parsed = _parse_json(raw, {"selectedIndex": 0, "reason": ""})
            idx = parsed.get("selectedIndex", 0)
            if isinstance(idx, (int, float)) and 1 <= int(idx) <= len(candidates):
                return candidates[int(idx) - 1]
        except Exception as e:
            logger.warning("Skill relation judge failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Handle new skill
    # ------------------------------------------------------------------

    async def _handle_new_skill(self, task: Task, chunks: list[Chunk]) -> None:
        eval_result = await self._evaluate_create(task)

        if not eval_result.should_generate or eval_result.confidence < self.min_confidence:
            logger.debug(
                "SkillEvolver: not generating for '%s' (confidence=%.2f)",
                task.title, eval_result.confidence,
            )
            return

        logger.info("SkillEvolver: generating skill '%s'", eval_result.suggested_name)
        skill = await self._generate_skill(task, chunks, eval_result)
        if skill:
            logger.info("SkillEvolver: skill '%s' created (id=%s)", skill.name, skill.id)

    # ------------------------------------------------------------------
    # Handle existing skill (upgrade)
    # ------------------------------------------------------------------

    async def _handle_existing_skill(
        self, task: Task, chunks: list[Chunk], skill: Skill,
    ) -> None:
        eval_result = await self._evaluate_upgrade(task, skill)

        if not eval_result.should_upgrade or eval_result.confidence < self.min_confidence:
            if eval_result.confidence < 0.3:
                await self._handle_new_skill(task, chunks)
            return

        logger.info("SkillEvolver: upgrading skill '%s' — %s", skill.name, eval_result.reason)
        await self._upgrade_skill(task, skill, eval_result)

    # ------------------------------------------------------------------
    # LLM Evaluator
    # ------------------------------------------------------------------

    async def _evaluate_create(self, task: Task) -> CreateEvalResult:
        prompt = (
            CREATE_EVAL_PROMPT
            .replace("{TITLE}", task.title or "")
            .replace("{SUMMARY}", (task.summary or "")[:3000])
        )
        try:
            raw = await self._llm_call(prompt, max_tokens=512, temperature=0)
            parsed = _parse_json(raw, {})
            return CreateEvalResult(
                should_generate=bool(parsed.get("shouldGenerate", False)),
                reason=str(parsed.get("reason", "")),
                suggested_name=str(parsed.get("suggestedName", "")),
                suggested_tags=parsed.get("suggestedTags"),
                confidence=float(parsed.get("confidence", 0)),
            )
        except Exception as e:
            logger.warning("Skill create eval failed: %s", e)
            return CreateEvalResult(reason=f"error: {e}")

    async def _evaluate_upgrade(self, task: Task, skill: Skill) -> UpgradeEvalResult:
        prompt = (
            UPGRADE_EVAL_PROMPT
            .replace("{VERSION}", str(skill.version))
            .replace("{SKILL_NAME}", skill.name)
            .replace("{SKILL_DESC}", (skill.description or "")[:1000])
            .replace("{TITLE}", task.title or "")
            .replace("{SUMMARY}", (task.summary or "")[:3000])
        )
        try:
            raw = await self._llm_call(prompt, max_tokens=512, temperature=0)
            parsed = _parse_json(raw, {})
            return UpgradeEvalResult(
                should_upgrade=bool(parsed.get("shouldUpgrade", False)),
                upgrade_type=str(parsed.get("upgradeType", "refine")),
                dimensions=parsed.get("dimensions"),
                reason=str(parsed.get("reason", "")),
                merge_strategy=str(parsed.get("mergeStrategy", "")),
                confidence=float(parsed.get("confidence", 0)),
            )
        except Exception as e:
            logger.warning("Skill upgrade eval failed: %s", e)
            return UpgradeEvalResult(reason=f"error: {e}")

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------

    async def _generate_skill(
        self, task: Task, chunks: list[Chunk], eval_result: CreateEvalResult,
    ) -> Skill | None:
        conversation = "\n\n".join(
            f"[{'User' if c.role == 'user' else 'Assistant'}]: {c.content[:500]}"
            for c in chunks[:30]
        )

        prompt = (
            SKILL_GENERATE_PROMPT
            .replace("{NAME}", eval_result.suggested_name)
            .replace("{TITLE}", task.title or "")
            .replace("{SUMMARY}", (task.summary or "")[:2000])
            .replace("{CONVERSATION}", conversation[:8000])
        )

        try:
            skill_content = await self._llm_call(prompt, max_tokens=4096, temperature=0.2)
        except Exception as e:
            logger.error("Skill generation LLM call failed: %s", e)
            return None

        if not skill_content or len(skill_content.strip()) < 50:
            logger.warning("Skill generation returned empty/short content")
            return None

        skill_id = str(uuid.uuid4())
        name = eval_result.suggested_name or f"skill-{skill_id[:8]}"
        description = _extract_description(skill_content) or (task.summary or "")[:200]

        skill_dir = ""
        if self.skill_store_dir:
            skill_dir = str(Path(self.skill_store_dir) / name)
            Path(skill_dir).mkdir(parents=True, exist_ok=True)
            (Path(skill_dir) / "SKILL.md").write_text(skill_content, encoding="utf-8")

        quality_score = await self._score_quality(skill_content, task)

        skill = Skill(
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
        self.store.insert_skill(skill)

        try:
            vec = await self.embedder.embed_query(f"{name} {description}")
            self.store.upsert_skill_embedding(skill_id, vec)
        except Exception as e:
            logger.warning("Skill embedding failed: %s", e)

        return skill

    async def _score_quality(self, content: str, task: Task) -> float:
        prompt = (
            f"Rate this skill document quality (0-10). Consider: completeness, clarity, "
            f"actionability, whether it captures the key steps from the task.\n\n"
            f"Task title: {task.title}\n\n"
            f"Skill content (first 2000 chars):\n{content[:2000]}\n\n"
            f"Reply with a single number (0-10):"
        )
        try:
            raw = await self._llm_call(prompt, max_tokens=10, temperature=0)
            m = re.search(r"(\d+(?:\.\d+)?)", raw)
            if m:
                return min(10.0, max(0.0, float(m.group(1))))
        except Exception:
            pass
        return 5.0

    # ------------------------------------------------------------------
    # Upgrader
    # ------------------------------------------------------------------

    async def _upgrade_skill(
        self, task: Task, skill: Skill, eval_result: UpgradeEvalResult,
    ) -> None:
        existing_content = ""
        if skill.dir_path:
            skill_file = Path(skill.dir_path) / "SKILL.md"
            if skill_file.exists():
                existing_content = skill_file.read_text(encoding="utf-8")

        if not existing_content:
            existing_content = skill.description or ""

        prompt = (
            SKILL_UPGRADE_PROMPT
            .replace("{SKILL_NAME}", skill.name)
            .replace("{SKILL_CONTENT}", existing_content[:4000])
            .replace("{TITLE}", task.title or "")
            .replace("{SUMMARY}", (task.summary or "")[:2000])
            .replace("{UPGRADE_TYPE}", eval_result.upgrade_type)
            .replace("{MERGE_STRATEGY}", eval_result.merge_strategy)
        )

        try:
            new_content = await self._llm_call(prompt, max_tokens=4096, temperature=0.2)
        except Exception as e:
            logger.error("Skill upgrade LLM call failed: %s", e)
            return

        if not new_content or len(new_content.strip()) < 50:
            return

        new_version = skill.version + 1
        new_desc = _extract_description(new_content) or skill.description

        if skill.dir_path:
            Path(skill.dir_path).mkdir(parents=True, exist_ok=True)
            (Path(skill.dir_path) / "SKILL.md").write_text(new_content, encoding="utf-8")

        self.store.update_skill(skill.id, description=new_desc, version=new_version)

        try:
            vec = await self.embedder.embed_query(f"{skill.name} {new_desc}")
            self.store.upsert_skill_embedding(skill.id, vec)
        except Exception:
            pass

        logger.info("Skill '%s' upgraded to v%d", skill.name, new_version)

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    async def _llm_call(
        self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.1,
    ) -> str:
        endpoint = f"{self.llm_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        body = {
            "model": self.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        return choices[0].get("message", {}).get("content", "").strip() if choices else ""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        store: MemStore,
        embedder: MemEmbedder,
        skill_store_dir: str = "",
    ) -> MemSkillEvolver:
        llm_cfg = config.get("llm", {})
        embedding_cfg = config.get("embedding", {})
        skill_cfg = config.get("skill_evolution", {})
        return cls(
            store=store,
            embedder=embedder,
            llm_base_url=llm_cfg.get("base_url") or embedding_cfg.get("base_url", ""),
            llm_api_key=llm_cfg.get("api_key") or embedding_cfg.get("api_key", ""),
            llm_model=llm_cfg.get("model") or "qwen-plus",
            skill_store_dir=skill_store_dir,
            min_chunks_for_eval=skill_cfg.get("min_chunks_for_eval", 6),
            min_confidence=skill_cfg.get("min_confidence", 0.7),
            auto_install=skill_cfg.get("auto_install", False),
            enabled=skill_cfg.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return fallback
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return fallback


def _extract_description(skill_content: str) -> str:
    m = re.search(r'description:\s*"([^"]+)"', skill_content)
    if m:
        return m.group(1).strip()
    m = re.search(r"description:\s*(.+)", skill_content)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return ""
