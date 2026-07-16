"""Pure session-title derivation rules."""

from __future__ import annotations

from typing import Any


class SessionTitleService:
    BOOTSTRAP_PREFIXES = (
        "a new session was started via /new or /reset",
        "[system message]",
    )

    def is_bootstrap_text(self, text: str | None) -> bool:
        raw = (text or "").strip().lower()
        return bool(raw) and any(
            raw.startswith(prefix) for prefix in self.BOOTSTRAP_PREFIXES
        )

    def derive(
        self,
        data: dict[str, Any] | None,
        *,
        session_id: str = "",
        updated_at: float | None = None,
        max_length: int = 60,
    ) -> str:
        max_len = max(1, int(max_length))
        if not data:
            return "未命名"
        label = str(data.get("label", "")).strip()
        if label and not self.is_bootstrap_text(label):
            return label[:max_len]
        for field in ("displayName", "subject"):
            value = str(data.get(field, "")).strip()
            if value:
                return value[:max_len]
        for message in data.get("messages", []):
            if message.get("role") != "user":
                continue
            text = " ".join(str(message.get("content", "")).split()).strip()
            if (
                not text
                or self.is_bootstrap_text(text)
                or text.startswith("/")
                or text.startswith(("http://", "https://"))
            ):
                continue
            return self._truncate(text, max_len)
        if session_id and updated_at:
            return f"{session_id} @ {int(updated_at)}"
        return "未命名"

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        for separator in ("。", ".", "！", "!", "？", "?", ";", "；", "\n"):
            index = text.find(separator, max_len // 2)
            if 0 < index <= max_len:
                return text[:index].strip() + "…"
        cut = text[:max(0, max_len - 1)]
        last_space = cut.rfind(" ")
        if last_space > max_len * 0.6:
            return cut[:last_space] + "…"
        return cut + "…"
