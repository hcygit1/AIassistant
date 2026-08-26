"""Offline first-phase Skill candidate gate and release command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.skill_release_gate import VariantMetrics, evaluate_release_gate, static_skill_check
from mem.skill_version_store import SkillVersion, SkillVersionStore


def _metrics(report: dict[str, object]) -> dict[str, VariantMetrics]:
    rows = report.get("systems") or []
    result: dict[str, VariantMetrics] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("system", ""))
        result[name] = VariantMetrics(
            pass_rate=float(row.get("pass_rate", 0)),
            avg_tokens=float(row.get("avg_tokens", 0)),
            external_failures=int(row.get("external_failures", 0)),
        )
    return result


def gate_report(
    report_path: Path,
    *,
    static_check_passed: bool | None = None,
    regression_candidate_passed: bool = False,
    candidate_path: Path | None = None,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    static_reasons: tuple[str, ...] = ()
    if candidate_path is not None:
        static_check_passed, static_reasons = static_skill_check(candidate_path)
    decision = evaluate_release_gate(
        static_check_passed=static_check_passed,
        validation=_metrics(report),
        regression_candidate_passed=regression_candidate_passed,
    )
    reasons = [*static_reasons, *decision.reasons]
    return {"status": decision.status, "reasons": reasons, "report": str(report_path)}


def publish_candidate(
    *, evolution_root: Path, active_root: Path, skill_id: str, version: int,
) -> None:
    store = SkillVersionStore(evolution_root=evolution_root, active_root=active_root)
    path = store.candidate_dir(skill_id, version) / "candidate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    candidate = SkillVersion(
        skill_id=str(data["skill_id"]), version=int(data["version"]),
        status=str(data["status"]), parent_version=data.get("parent_version"),
        source=str(data.get("source", "")), reason=str(data.get("reason", "")),
    )
    store.publish(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="PIPIXIA offline Skill evolution")
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--candidate", type=Path)
    gate.add_argument("--evolution-root", type=Path)
    gate.add_argument("--skill")
    gate.add_argument("--version", type=int)
    gate.add_argument("--static-check-passed", action="store_true")
    gate.add_argument("--regression-passed", action="store_true")
    publish = sub.add_parser("publish")
    publish.add_argument("--evolution-root", type=Path, required=True)
    publish.add_argument("--active-root", type=Path, required=True)
    publish.add_argument("--skill", required=True)
    publish.add_argument("--version", type=int, required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--active-root", type=Path, required=True)
    rollback.add_argument("--skill", required=True)
    rollback.add_argument("--version", type=int)
    args = parser.parse_args()
    if args.command == "gate":
        result = gate_report(
            args.report,
            static_check_passed=args.static_check_passed,
            regression_candidate_passed=args.regression_passed,
            candidate_path=getattr(args, "candidate", None),
        )
        if args.evolution_root and args.skill and args.version:
            store = SkillVersionStore(
                evolution_root=args.evolution_root,
                active_root=args.evolution_root.parent / "skills",
            )
            metadata = store.candidate_dir(args.skill, args.version) / "candidate.json"
            data = json.loads(metadata.read_text(encoding="utf-8"))
            candidate = SkillVersion(
                skill_id=str(data["skill_id"]), version=int(data["version"]),
                status=str(data["status"]), parent_version=data.get("parent_version"),
                source=str(data.get("source", "")), reason=str(data.get("reason", "")),
            )
            updated = store.set_candidate_status(candidate, str(result["status"]))
            result["candidate_status"] = updated.status
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "rollback":
        store = SkillVersionStore(evolution_root=args.active_root.parent / "skill_evolution", active_root=args.active_root)
        print(json.dumps({"active_version": store.rollback(args.skill, args.version)}))
        return
    publish_candidate(
        evolution_root=args.evolution_root, active_root=args.active_root,
        skill_id=args.skill, version=args.version,
    )


if __name__ == "__main__":
    main()
