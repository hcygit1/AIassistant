"""Adapters for using SkillLearnBench with PIPIXIA skill distillation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mem.models import Chunk, Task


@dataclass(frozen=True)
class SkillLearnInstance:
    family: str
    instance_id: str
    path: str
    instruction_path: str
    verifier_path: str


def discover_family(benchmark_root: Path, family: str) -> list[SkillLearnInstance]:
    family_dir = benchmark_root / "tasks" / family
    if not family_dir.is_dir():
        raise FileNotFoundError(f"SkillLearnBench family not found: {family_dir}")
    instances: list[SkillLearnInstance] = []
    for instance_dir in sorted(family_dir.iterdir()):
        instruction = instance_dir / "instruction.md"
        verifier = instance_dir / "tests" / "test_outputs.py"
        if not instance_dir.is_dir() or not instruction.is_file():
            continue
        if not verifier.is_file():
            raise FileNotFoundError(f"verifier not found: {verifier}")
        instances.append(SkillLearnInstance(
            family=family,
            instance_id=instance_dir.name,
            path=str(instance_dir),
            instruction_path=str(instruction),
            verifier_path=str(verifier),
        ))
    if len(instances) < 2:
        raise ValueError(f"family requires at least two instances: {family}")
    return instances


def benchmark_revision(benchmark_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=benchmark_root,
        capture_output=True, text=True, check=True,
    )
    return completed.stdout.strip()


def build_manifest(benchmark_root: Path, families: list[str]) -> dict[str, Any]:
    entries = []
    for family in families:
        instances = discover_family(benchmark_root, family)
        entries.append({
            "family": family,
            "seed_instance": asdict(instances[0]),
            "evaluation_instances": [asdict(item) for item in instances[1:]],
        })
    return {
        "benchmark": "SkillLearnBench",
        "benchmark_root": str(benchmark_root.resolve()),
        "revision": benchmark_revision(benchmark_root),
        "leakage_policy": "instruction and successful trajectory only; verifier and solution excluded",
        "families": entries,
    }


def load_trajectory(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    messages: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = str(item.get("role") or item.get("type") or "assistant")
        content = item.get("content") or item.get("text") or item.get("message")
        if isinstance(content, dict):
            content = content.get("content") or content.get("text")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") if isinstance(part, dict) else part)
                for part in content
            )
        if content:
            messages.append((role, str(content)))
    if not messages and text.strip():
        messages.append(("assistant", text.strip()))
    return messages


def build_completed_task(
    instance: SkillLearnInstance,
    trajectory_path: Path,
    *,
    owner: str = "eval:skilllearnbench",
) -> tuple[Task, list[Chunk]]:
    instruction = Path(instance.instruction_path).read_text(encoding="utf-8")
    messages = [("user", instruction), *load_trajectory(trajectory_path)]
    expanded: list[tuple[str, str]] = []
    for role, content in messages:
        pieces = [content[i:i + 1800] for i in range(0, len(content), 1800)] or [""]
        expanded.extend((role, piece) for piece in pieces if piece.strip())
    if not any(role == "assistant" for role, _ in expanded):
        raise ValueError("successful trajectory must contain assistant output")
    if len(expanded) < 6:
        raise ValueError("successful trajectory is too short for skill distillation")

    task_id = f"skilllearnbench:{instance.instance_id}"
    summary = (
        f"Completed verified SkillLearnBench workflow '{instance.family}'. "
        f"The task instruction and successful execution trajectory are retained as evidence. "
        f"Goal: {instruction[:2200]}"
    )
    task = Task(
        id=task_id, session_key=task_id, owner=owner,
        title=f"SkillLearnBench: {instance.family}", summary=summary,
        status="completed",
    )
    chunks = [
        Chunk(
            id=f"{task_id}:chunk:{index}", session_key=task_id,
            turn_id=f"turn:{index}", seq=index, role=role,
            content=content, task_id=task_id, owner=owner,
        )
        for index, (role, content) in enumerate(expanded)
    ]
    return task, chunks
