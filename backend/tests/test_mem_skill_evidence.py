from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Chunk


def _evidence_functions() -> tuple[Any, Any, Any]:
    try:
        from mem.skill_evidence import (
            build_skill_evidence,
            chunk_signal_score,
            extract_original_goal,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_evidence should own skill evidence construction"
        ) from exc
    return extract_original_goal, chunk_signal_score, build_skill_evidence


def _chunk(
    chunk_id: str,
    role: str,
    content: str,
    *,
    summary: str = "",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        session_key="session-1",
        turn_id=f"turn-{chunk_id}",
        seq=0,
        role=role,
        content=content,
        summary=summary,
        owner="agent:main",
    )


class SkillEvidenceTests(unittest.TestCase):
    def test_evidence_logic_has_a_neutral_owner(self) -> None:
        evidence_path = BACKEND_DIR / "mem" / "skill_evidence.py"
        self.assertTrue(
            evidence_path.is_file(),
            "mem.skill_evidence should own skill evidence construction",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_evidence import", evolver_source)
        for pattern_name in (
            "COMMAND_RE",
            "PATH_RE",
            "STRUCTURED_SIGNAL_RE",
            "RESULT_SIGNAL_RE",
            "FILLER_RE",
        ):
            self.assertNotIn(f"{pattern_name} = re.compile", evolver_source)

    def test_evidence_prefers_goal_error_commands_and_final_result(self) -> None:
        extract_original_goal, chunk_signal_score, build_skill_evidence = (
            _evidence_functions()
        )
        chunks = [
            _chunk("c1", "user", "帮我修复 postgres 连接问题"),
            _chunk("c2", "assistant", "好的"),
            _chunk("c3", "user", "报错 ECONNREFUSED 127.0.0.1:5432"),
            _chunk(
                "c4",
                "assistant",
                "把 DB_PORT 改为 6432，并执行 `docker compose restart api`",
                summary="DB_PORT 改为 6432，重启 api",
            ),
            _chunk(
                "c5",
                "assistant",
                "最终验证成功，连接恢复正常，配置文件是 /app/.env",
                summary="验证成功，配置文件 /app/.env",
            ),
        ]

        self.assertEqual(
            extract_original_goal(chunks),
            "帮我修复 postgres 连接问题",
        )
        self.assertEqual(chunk_signal_score(chunks[1], 1, len(chunks)), -100)
        self.assertEqual(
            build_skill_evidence(chunks),
            "\n".join(
                [
                    "1. [User] 帮我修复 postgres 连接问题",
                    "2. [User] 报错 ECONNREFUSED 127.0.0.1:5432",
                    "3. [Assistant] DB_PORT 改为 6432，重启 api",
                    "4. [Assistant] 验证成功，配置文件 /app/.env",
                ]
            ),
        )

    def test_evidence_fallback_deduplication_and_limit_are_preserved(self) -> None:
        extract_original_goal, _chunk_signal_score, build_skill_evidence = (
            _evidence_functions()
        )
        system_only = [_chunk("system", "system", "system prompt")]
        duplicates = [
            _chunk(
                f"c{i}",
                "assistant",
                f"最终验证成功，状态码 {1000 + i}",
                summary=("Same Result", "same result", "SAME RESULT")[i],
            )
            for i in range(3)
        ]
        limited = [
            _chunk(
                f"limit-{i}",
                "assistant",
                f"最终验证成功，状态码 {2000 + i}",
                summary=f"结果 {i}",
            )
            for i in range(12)
        ]

        self.assertEqual(
            extract_original_goal(system_only),
            "(no explicit user goal found)",
        )
        self.assertEqual(build_skill_evidence(system_only), "(no supporting evidence)")
        deduplicated = build_skill_evidence(duplicates)
        limited_evidence = build_skill_evidence(limited)
        truncated = build_skill_evidence(
            [_chunk("long", "user", "x" * 705)]
        )
        self.assertEqual(deduplicated, "1. [Assistant] SAME RESULT")
        self.assertEqual(
            limited_evidence.splitlines(),
            [
                "1. [Assistant] 结果 11",
                "2. [Assistant] 结果 10",
                "3. [Assistant] 结果 9",
                "4. [Assistant] 结果 8",
                "5. [Assistant] 结果 7",
                "6. [Assistant] 结果 6",
                "7. [Assistant] 结果 5",
                "8. [Assistant] 结果 4",
            ],
        )
        truncated_text = truncated.split("] ", 1)[1]
        self.assertEqual(len(truncated_text), 700)
        self.assertTrue(truncated_text.endswith("..."))

    def test_skill_evolver_compatibility_uses_overridable_signal_score(self) -> None:
        extract_original_goal, chunk_signal_score, _build_skill_evidence = (
            _evidence_functions()
        )
        from mem.skill_evolver import MemSkillEvolver

        calls: list[str] = []

        class CustomEvolver(MemSkillEvolver):
            @staticmethod
            def _chunk_signal_score(chunk: Chunk, index: int, total: int) -> int:
                calls.append(chunk.id)
                return {"c1": 10, "c2": 1}[chunk.id]

        evolver = object.__new__(CustomEvolver)
        chunks = [
            _chunk("c1", "assistant", "第一条普通流程记录，没有结果信号"),
            _chunk("c2", "assistant", "第二条普通流程记录，没有结果信号"),
        ]

        self.assertEqual(
            evolver._extract_original_goal(chunks),
            extract_original_goal(chunks),
        )
        self.assertEqual(
            MemSkillEvolver._chunk_signal_score(chunks[0], 0, 2),
            chunk_signal_score(chunks[0], 0, 2),
        )
        evidence = evolver._build_skill_evidence(chunks)
        self.assertEqual(calls, ["c1", "c2"])
        self.assertEqual(
            evidence.splitlines(),
            [
                "1. [Assistant] 第一条普通流程记录，没有结果信号",
                "2. [Assistant] 第二条普通流程记录，没有结果信号",
            ],
        )


if __name__ == "__main__":
    unittest.main()
