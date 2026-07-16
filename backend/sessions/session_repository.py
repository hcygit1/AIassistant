"""JSON persistence boundary for session transcripts and indexes."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable


class SessionDataCorruptionError(ValueError):
    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"会话文件损坏，无法解析: {path}")


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        existing_mode = None
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            if existing_mode is not None:
                fchmod = getattr(os, "fchmod", None)
                if callable(fchmod):
                    fchmod(temp_file.fileno(), existing_mode)
                else:
                    os.chmod(temp_path, existing_mode)
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


class _SessionCache:
    def __init__(self, max_size: int = 20):
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class SessionRepository:
    _caches: dict[str, _SessionCache] = {}
    _caches_guard = threading.Lock()
    _agent_locks: dict[str, threading.RLock] = {}
    _agent_locks_guard = threading.Lock()
    _session_locks: dict[str, threading.RLock] = {}
    _session_locks_guard = threading.Lock()
    _index_locks: dict[str, threading.RLock] = {}
    _index_locks_guard = threading.Lock()

    def __init__(
        self,
        *,
        resolve_sessions_dir: Callable[[str], Path],
    ) -> None:
        self._resolve_sessions_dir = resolve_sessions_dir

    def sessions_dir(self, agent_id: str) -> Path:
        return self._resolve_sessions_dir(agent_id)

    def _agent_key(self, agent_id: str) -> str:
        return str(self.sessions_dir(agent_id).resolve())

    def _cache_key(self, session_id: str, agent_id: str) -> str:
        return f"{agent_id}:{session_id}"

    def _get_cache(self, agent_id: str) -> _SessionCache:
        key = self._agent_key(agent_id)
        with self._caches_guard:
            return self._caches.setdefault(key, _SessionCache(max_size=30))

    def session_path(self, session_id: str, agent_id: str) -> Path:
        return self.sessions_dir(agent_id) / f"{session_id}.json"

    def index_path(self, agent_id: str) -> Path:
        return self.sessions_dir(agent_id) / "sessions.json"

    def archive_path(self, agent_id: str, filename: str) -> Path:
        return self.sessions_dir(agent_id) / "archive" / filename

    def session_file_exists(self, session_id: str, agent_id: str) -> bool:
        return self.session_path(session_id, agent_id).is_file()

    def session_file_size(self, session_id: str, agent_id: str) -> int:
        path = self.session_path(session_id, agent_id)
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def directory_size(self, agent_id: str) -> int:
        root = self.sessions_dir(agent_id)
        if not root.exists():
            return 0
        total = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def prune_oldest_archives(
        self,
        agent_id: str,
        *,
        current_total: int,
        target_total: int,
    ) -> tuple[int, int, int]:
        archive = self.archive_path(agent_id, "placeholder").parent
        if not archive.exists():
            return current_total, 0, 0
        files: list[tuple[Path, float, int]] = []
        for path in archive.iterdir():
            if not path.is_file():
                continue
            try:
                stats = path.stat()
                files.append((path, stats.st_mtime, stats.st_size))
            except OSError:
                continue
        files.sort(key=lambda item: item[1])
        removed = 0
        freed = 0
        total = current_total
        for path, _, size in files:
            if total <= target_total:
                break
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            freed += size
            removed += 1
        return total, removed, freed

    def get_agent_lock(self, agent_id: str) -> threading.RLock:
        key = self._agent_key(agent_id)
        with self._agent_locks_guard:
            return self._agent_locks.setdefault(key, threading.RLock())

    def get_session_lock(
        self,
        session_id: str,
        agent_id: str,
    ) -> threading.RLock:
        key = f"{self._agent_key(agent_id)}:{session_id}"
        with self._session_locks_guard:
            return self._session_locks.setdefault(key, threading.RLock())

    def get_index_lock(self, agent_id: str) -> threading.RLock:
        key = self._agent_key(agent_id)
        with self._index_locks_guard:
            return self._index_locks.setdefault(key, threading.RLock())

    def load_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        with self.get_agent_lock(agent_id):
            with self.get_session_lock(session_id, agent_id):
                return self._load_session_locked(session_id, agent_id)

    def _load_session_locked(
        self,
        session_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        cache_key = self._cache_key(session_id, agent_id)
        cache = self._get_cache(agent_id)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        path = self.session_path(session_id, agent_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SessionDataCorruptionError(path) from exc
        if isinstance(data, list):
            data = {
                "label": "未命名",
                "agent_id": agent_id,
                "created_at": time.time(),
                "updated_at": time.time(),
                "messages": data,
            }
        if not isinstance(data, dict):
            raise SessionDataCorruptionError(path)
        cache.put(cache_key, data)
        return data

    def save_session(
        self,
        session_id: str,
        agent_id: str,
        data: dict[str, Any],
    ) -> None:
        with self.get_agent_lock(agent_id):
            with self.get_session_lock(session_id, agent_id):
                _write_json_atomic(
                    self.session_path(session_id, agent_id),
                    data,
                )
                self._get_cache(agent_id).put(
                    self._cache_key(session_id, agent_id),
                    data,
                )

    def invalidate_session(self, session_id: str, agent_id: str) -> None:
        self._get_cache(agent_id).invalidate(
            self._cache_key(session_id, agent_id)
        )

    def clear_cache(self) -> None:
        with self._caches_guard:
            caches = list(self._caches.values())
        for cache in caches:
            cache.clear()

    def delete_session_file(self, session_id: str, agent_id: str) -> bool:
        with self.get_agent_lock(agent_id):
            with self.get_session_lock(session_id, agent_id):
                path = self.session_path(session_id, agent_id)
                if not path.exists():
                    self.invalidate_session(session_id, agent_id)
                    return False
                path.unlink()
                self.invalidate_session(session_id, agent_id)
                return True

    def archive_session_file(
        self,
        session_id: str,
        agent_id: str,
        destination: Path,
    ) -> bool:
        with self.get_agent_lock(agent_id):
            with self.get_session_lock(session_id, agent_id):
                path = self.session_path(session_id, agent_id)
                if not path.exists():
                    self.invalidate_session(session_id, agent_id)
                    return False
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
                self.invalidate_session(session_id, agent_id)
                return True

    def load_index(self, agent_id: str) -> dict[str, dict[str, Any]]:
        with self.get_index_lock(agent_id):
            path = self.index_path(agent_id)
            if not path.exists():
                return {}
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except FileNotFoundError:
                return {}
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise SessionDataCorruptionError(path) from exc
            if not isinstance(data, dict):
                raise SessionDataCorruptionError(path)
            return data

    def save_index(
        self,
        agent_id: str,
        store: dict[str, dict[str, Any]],
    ) -> None:
        with self.get_agent_lock(agent_id):
            with self.get_index_lock(agent_id):
                _write_json_atomic(self.index_path(agent_id), store)
