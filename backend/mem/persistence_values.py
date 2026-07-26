"""Default value generation shared by memory ingestion and persistence."""

from __future__ import annotations

import hashlib
import time


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def now_ms() -> int:
    return int(time.time() * 1000)
