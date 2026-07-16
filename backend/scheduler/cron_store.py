"""Cron store — JSON 持久化"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fcntl

from config import DATA_DIR

from .cron_types import CronJob, CronStore


class CronStoreError(Exception):
    pass


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


@contextmanager
def cron_store_transaction(path: Path) -> Iterator[None]:
    resolved = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        process_lock = _PATH_LOCKS.setdefault(
            resolved,
            threading.RLock(),
        )
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with process_lock:
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def resolve_cron_store_path(override: str | None = None) -> Path:
    """解析 cron store 路径"""
    if override and str(override).strip():
        return Path(override).resolve()
    return DATA_DIR / "cron" / "jobs.json"


def load_cron_store(path: Path | None = None) -> CronStore:
    """加载 cron store"""
    p = path or resolve_cron_store_path()
    if not p.exists():
        return CronStore()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise CronStoreError(
            f"Failed to load cron store {p}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CronStoreError(
            f"Invalid cron store root in {p}"
        )
    jobs_raw = data.get("jobs")
    if jobs_raw is not None and not isinstance(jobs_raw, list):
        raise CronStoreError(
            f"Invalid cron jobs list in {p}"
        )
    jobs: list[CronJob] = []
    if isinstance(jobs_raw, list):
        for index, job_data in enumerate(jobs_raw):
            if not isinstance(job_data, dict) or not job_data.get("id"):
                raise CronStoreError(
                    f"Invalid cron job at index {index} in {p}"
                )
            try:
                jobs.append(CronJob.from_dict(job_data))
            except Exception as exc:
                raise CronStoreError(
                    f"Invalid cron job at index {index} in {p}: {exc}"
                ) from exc
    return CronStore(version=int(data.get("version", 1)), jobs=jobs)


def save_cron_store(store: CronStore, path: Path | None = None) -> None:
    """持久化 cron store"""
    p = path or resolve_cron_store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": store.version,
        "jobs": [j.to_dict() for j in store.jobs],
    }
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{p.name}.",
        suffix=".tmp",
        dir=p.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, p)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
