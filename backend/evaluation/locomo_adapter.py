"""Adapt LoCoMo conversations to PIPIXIA Task/Chunk recall records."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mem.models import Chunk, Task

_SESSION_RE = re.compile(r"^session_(\d+)$")
_EVIDENCE_RE = re.compile(r"D\d+:\d+")
_DATE_FORMAT = "%I:%M %p on %d %B, %Y"


@dataclass(frozen=True)
class LoCoMoCase:
    case_id: str
    query: str
    answer: str
    category: int
    golden_task_ids: list[str]
    golden_chunk_ids: list[str]
    is_abstention: bool


@dataclass(frozen=True)
class AdaptedConversation:
    sample_id: str
    tasks: list[Task]
    chunks: list[Chunk]
    cases: list[LoCoMoCase]


def _timestamp_ms(value: str) -> int:
    try:
        return int(datetime.strptime(value, _DATE_FORMAT).timestamp() * 1000)
    except (TypeError, ValueError):
        return 0


def _session_items(conversation: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    items = [
        (key, value) for key, value in conversation.items()
        if _SESSION_RE.match(key) and isinstance(value, list)
    ]
    return sorted(items, key=lambda item: int(_SESSION_RE.match(item[0]).group(1)))


def _speaker_roles(sessions: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, str]:
    speakers: list[str] = []
    for _, turns in sessions:
        for turn in turns:
            speaker = str(turn.get("speaker", "")).strip()
            if speaker and speaker not in speakers:
                speakers.append(speaker)
    return {speaker: "user" if index == 0 else "assistant" for index, speaker in enumerate(speakers)}


def _evidence_ids(raw_evidence: Any) -> list[str]:
    values = raw_evidence if isinstance(raw_evidence, list) else [raw_evidence]
    matches: list[str] = []
    for value in values:
        matches.extend(_EVIDENCE_RE.findall(str(value)))
    return list(dict.fromkeys(matches))


def _select_cases(
    qa_items: list[dict[str, Any]],
    max_questions: int,
    *,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    answerable = [
        (index, item)
        for index, item in enumerate(qa_items)
        if _evidence_ids(item.get("evidence"))
    ]
    if max_questions <= 0 or max_questions >= len(answerable):
        return answerable

    by_category: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for indexed_item in answerable:
        category = int(indexed_item[1].get("category", 0))
        by_category.setdefault(category, []).append(indexed_item)

    rng = random.Random(seed)
    for items in by_category.values():
        rng.shuffle(items)

    selected: list[tuple[int, dict[str, Any]]] = []
    categories = sorted(by_category)
    while len(selected) < max_questions:
        added = False
        for category in categories:
            items = by_category[category]
            if items and len(selected) < max_questions:
                selected.append(items.pop())
                added = True
        if not added:
            break
    return selected


def adapt_conversation(
    source: dict[str, Any],
    *,
    max_questions: int = 20,
    include_abstention: bool = True,
    seed: int = 42,
    session_summaries: dict[str, str] | None = None,
) -> AdaptedConversation:
    sample_id = str(source.get("sample_id") or source.get("id") or "locomo-sample")
    conversation = source.get("conversation") or {}
    sessions = _session_items(conversation)
    roles = _speaker_roles(sessions)
    session_summaries = session_summaries or source.get("session_summary") or {}
    tasks: list[Task] = []
    chunks: list[Chunk] = []
    chunk_to_task: dict[str, str] = {}
    for session_key, turns in sessions:
        task_id = f"{sample_id}:{session_key}"
        started_at = _timestamp_ms(str(conversation.get(f"{session_key}_date_time", "")))
        summary = str(
            session_summaries.get(f"{session_key}_summary")
            or " ".join(str(turn.get("text", "")).strip() for turn in turns)
        )
        tasks.append(Task(id=task_id, session_key=f"locomo:{sample_id}:{session_key}", owner="eval:locomo", title=f"LoCoMo {session_key}", summary=summary[:2000], status="completed", started_at=started_at, ended_at=started_at, updated_at=started_at))
        for seq, turn in enumerate(turns):
            chunk_id = str(turn.get("dia_id") or f"{task_id}:turn_{seq}")
            content = str(turn.get("text", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            chunks.append(Chunk(id=chunk_id, session_key=f"locomo:{sample_id}:{session_key}", turn_id=chunk_id, seq=seq, role=roles.get(speaker, "assistant"), content=content, summary=content[:300], task_id=task_id, owner="eval:locomo", created_at=started_at + seq, updated_at=started_at + seq))
            chunk_to_task[chunk_id] = task_id
    qa_items = list(source.get("qa") or [])
    selected = _select_cases(qa_items, max_questions, seed=seed)
    if include_abstention:
        abstentions = [
            (index, item)
            for index, item in enumerate(qa_items)
            if not _evidence_ids(item.get("evidence"))
        ]
        remaining = len(abstentions) if max_questions <= 0 else max(0, max_questions - len(selected))
        selected.extend(abstentions[:remaining])
    cases: list[LoCoMoCase] = []
    for source_index, item in selected:
        evidence = _evidence_ids(item.get("evidence"))
        golden_tasks = list(dict.fromkeys(chunk_to_task[chunk_id] for chunk_id in evidence if chunk_id in chunk_to_task))
        cases.append(LoCoMoCase(case_id=f"{sample_id}:qa_{source_index:03d}", query=str(item.get("question", "")).strip(), answer=str(item.get("answer", "")).strip(), category=int(item.get("category", 0)), golden_task_ids=golden_tasks, golden_chunk_ids=evidence, is_abstention=not evidence))
    return AdaptedConversation(sample_id=sample_id, tasks=tasks, chunks=chunks, cases=cases)


def load_pilot(
    path: str | Path,
    *,
    sample_index: int = 0,
    max_questions: int = 20,
    include_abstention: bool = False,
    seed: int = 42,
    session_summaries: dict[str, str] | None = None,
) -> AdaptedConversation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return adapt_conversation(
        data[sample_index],
        max_questions=max_questions,
        include_abstention=include_abstention,
        seed=seed,
        session_summaries=session_summaries,
    )


def load_source(path: str | Path, *, sample_index: int = 0) -> dict[str, Any]:
    """Load one raw LoCoMo conversation without applying evaluation sampling."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data[sample_index]
