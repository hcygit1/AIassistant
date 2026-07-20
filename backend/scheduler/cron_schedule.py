"""Cron schedule calculation and definition validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from scheduler.cron_errors import CronServiceError
from scheduler.cron_types import CronJob, CronPayload, CronSchedule


def compute_next_run(
    job: CronJob,
    now_ms: int,
    last_run_ms: int | None = None,
) -> int | None:
    if not job.enabled:
        return None
    schedule = job.schedule
    if schedule.kind == "at":
        return _parse_at_ms(schedule.at, schedule.tz)
    if schedule.kind == "every":
        every_ms = schedule.every_ms or 0
        if every_ms <= 0:
            return None
        anchor = (
            last_run_ms
            if last_run_ms is not None
            else job.created_at_ms or now_ms
        )
        return anchor + every_ms
    if schedule.kind == "cron" and schedule.expr:
        tz = _resolve_timezone(schedule.tz)
        now = datetime.fromtimestamp(now_ms / 1000, tz=tz)
        next_dt = croniter(schedule.expr, now).get_next(datetime)
        return int(next_dt.timestamp() * 1000)
    return None


def build_schedule(
    data: dict[str, Any],
    *,
    now_ms: int,
    default_timezone: Callable[[], str],
    current: CronSchedule | None = None,
) -> CronSchedule:
    raw = data or {}
    kind = raw.get(
        "kind",
        current.kind if current is not None else None,
    )
    if kind not in ("at", "every", "cron"):
        raise CronServiceError(
            "invalid_schedule",
            "schedule.kind must be at, every, or cron",
        )
    timezone_name = raw.get(
        "tz",
        current.tz if current is not None else None,
    )
    if kind in ("at", "cron") and not timezone_name:
        timezone_name = default_timezone()
    schedule = CronSchedule(
        kind=kind,
        at=raw.get(
            "at",
            current.at if current is not None else None,
        ),
        every_ms=raw.get(
            "everyMs",
            current.every_ms if current is not None else None,
        ),
        expr=raw.get(
            "expr",
            current.expr if current is not None else None,
        ),
        tz=timezone_name,
    )
    try:
        if kind == "at":
            at_ms = _parse_at_ms(schedule.at, schedule.tz)
            if at_ms is None or at_ms <= now_ms:
                raise ValueError("at must be in the future")
        elif kind == "every":
            schedule.every_ms = int(schedule.every_ms or 0)
            if schedule.every_ms <= 0:
                raise ValueError("everyMs must be positive")
        else:
            if not schedule.expr or not croniter.is_valid(schedule.expr):
                raise ValueError("invalid cron expression")
            _resolve_timezone(schedule.tz)
    except CronServiceError:
        raise
    except (TypeError, ValueError) as exc:
        raise CronServiceError(
            "invalid_schedule",
            str(exc),
        ) from exc
    return schedule


def build_payload(data: dict[str, Any]) -> CronPayload:
    payload = data or {}
    if payload.get("kind") != "systemEvent":
        raise CronServiceError(
            "invalid_payload",
            "payload.kind must be systemEvent",
        )
    text = str(payload.get("text", "")).strip()
    if not text:
        raise CronServiceError(
            "invalid_payload",
            "payload.text is required",
        )
    return CronPayload(kind="systemEvent", text=text)


def schedule_state(job: CronJob) -> tuple[Any, ...]:
    schedule = job.schedule
    return (
        job.enabled,
        job.delete_after_run,
        schedule.kind,
        schedule.at,
        schedule.every_ms,
        schedule.expr,
        schedule.tz,
    )


def _parse_at_ms(value: str | None, tz_name: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=_resolve_timezone(tz_name or "UTC")
        )
    return int(parsed.timestamp() * 1000)


def _resolve_timezone(name: str | None):
    if not name:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise CronServiceError(
            "invalid_schedule",
            f"unknown timezone: {name}",
        ) from exc
