"""Small deterministic LoCoMo pilot runner."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.locomo_adapter import AdaptedConversation, load_pilot


@dataclass(frozen=True)
class CaseMetrics:
    hit_rate_at_k: float
    evidence_recall_at_k: float
    mrr: float
    required_fact_coverage: float

    @property
    def recall_at_k(self) -> float:
        """Compatibility alias for the old, incorrectly named hit-rate field."""
        return self.hit_rate_at_k


@dataclass(frozen=True)
class PilotSummary:
    system: str
    total_cases: int
    recall_at_k: float
    mrr: float
    abstention_cases: int


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text) if token.strip()}


def evaluate_case(*, retrieved_ids: list[str], golden_ids: list[str], answer: str, context: str, k: int, is_abstention: bool = False) -> CaseMetrics:
    if is_abstention or not golden_ids:
        return CaseMetrics(0.0, 0.0, 0.0, 0.0)
    top_k = retrieved_ids[:k]
    golden = set(golden_ids)
    rank = next((index + 1 for index, item in enumerate(retrieved_ids) if item in golden), None)
    matched = golden.intersection(top_k)
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(context)
    coverage = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    return CaseMetrics(
        1.0 if matched else 0.0,
        len(matched) / len(golden),
        1.0 / rank if rank else 0.0,
        coverage,
    )


def _lexical_score(query: str, text: str) -> int:
    return len(_tokens(query) & _tokens(text))


def direct_chunk_retrieve(adapted: AdaptedConversation, query: str, top_k: int) -> list[str]:
    ranked = sorted(adapted.chunks, key=lambda chunk: (-_lexical_score(query, chunk.content), chunk.id))
    return [chunk.id for chunk in ranked[:top_k]]


def waterfall_retrieve(adapted: AdaptedConversation, query: str, top_k: int) -> list[str]:
    chunks_by_task: dict[str, list] = {}
    for chunk in adapted.chunks:
        chunks_by_task.setdefault(chunk.task_id or "", []).append(chunk)
    ranked_tasks = sorted(adapted.tasks, key=lambda task: (-_lexical_score(query, task.summary), task.id))
    result: list[str] = []
    for task in ranked_tasks[: max(1, min(5, len(ranked_tasks)))]:
        ranked_chunks = sorted(chunks_by_task.get(task.id, []), key=lambda chunk: (-_lexical_score(query, chunk.content), chunk.id))
        result.extend(chunk.id for chunk in ranked_chunks)
        if len(result) >= top_k:
            break
    return result[:top_k]


def _run_system(adapted: AdaptedConversation, system: str, top_k: int) -> PilotSummary:
    metrics: list[CaseMetrics] = []
    abstentions = sum(case.is_abstention for case in adapted.cases)
    for case in adapted.cases:
        retrieved = direct_chunk_retrieve(adapted, case.query, top_k) if system == "direct" else waterfall_retrieve(adapted, case.query, top_k)
        context = " ".join(chunk.content for chunk in adapted.chunks if chunk.id in retrieved)
        metrics.append(evaluate_case(retrieved_ids=retrieved, golden_ids=case.golden_chunk_ids, answer=case.answer, context=context, k=top_k, is_abstention=case.is_abstention))
    denominator = max(1, len(metrics) - abstentions)
    return PilotSummary(system, len(metrics), sum(item.recall_at_k for item in metrics) / denominator, sum(item.mrr for item in metrics) / denominator, abstentions)


def run_pilot(path: str | Path, *, sample_index: int = 0, max_questions: int = 20, top_k: int = 5) -> dict[str, object]:
    adapted = load_pilot(path, sample_index=sample_index, max_questions=max_questions, include_abstention=True)
    return {"source": str(path), "sample_id": adapted.sample_id, "task_count": len(adapted.tasks), "chunk_count": len(adapted.chunks), "case_count": len(adapted.cases), "systems": [asdict(_run_system(adapted, "direct", top_k)), asdict(_run_system(adapted, "waterfall", top_k))]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal LoCoMo recall pilot")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = run_pilot(args.data, sample_index=args.sample_index, max_questions=args.max_questions, top_k=args.top_k)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
