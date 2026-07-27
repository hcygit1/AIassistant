"""Pure evidence selection for memory skill generation."""

from __future__ import annotations

import re
from collections.abc import Callable

from mem.models import Chunk


COMMAND_RE = re.compile(
    r"\b(?:git|python|pip|npm|pnpm|yarn|node|uv|poetry|docker|"
    r"docker-compose|kubectl|curl|wget|make)\b",
    re.IGNORECASE,
)
PATH_RE = re.compile(
    r"(?:/[\w./-]+|\b[\w.-]+\."
    r"(?:py|json|ya?ml|toml|env|ini|md|sh|ts|tsx|js|jsx)\b)"
)
STRUCTURED_SIGNAL_RE = re.compile(
    r"\b(?:v?\d+\.\d+(?:\.\d+)?|[A-Z]+-\d+|\d{2,5}|0x[a-fA-F0-9]+)\b"
)
RESULT_SIGNAL_RE = re.compile(
    r"(最终|修复|解决|成功|失败|验证|报错|错误|改为|需要|应该|fixed|resolved|"
    r"success|failed|verify|error|updated|changed|final)",
    re.IGNORECASE,
)
FILLER_RE = re.compile(
    r"^(?:好的|收到|明白了|谢谢|你好|测试|ok|okay|thanks|thank you|got it|"
    r"understood|sure)\s*[.!?。！？]*$",
    re.IGNORECASE,
)

SignalScore = Callable[[Chunk, int, int], int]


def extract_original_goal(chunks: list[Chunk]) -> str:
    first_user = next(
        (chunk for chunk in chunks if chunk.role == "user" and chunk.content.strip()),
        None,
    )
    if not first_user:
        return "(no explicit user goal found)"
    return first_user.content.strip()


def chunk_signal_score(chunk: Chunk, index: int, total: int) -> int:
    text = (chunk.content or "").strip()
    if not text or FILLER_RE.match(text):
        return -100

    score = 0
    if COMMAND_RE.search(text):
        score += 3
    if PATH_RE.search(text):
        score += 2
    if STRUCTURED_SIGNAL_RE.search(text):
        score += 2
    if RESULT_SIGNAL_RE.search(text) or RESULT_SIGNAL_RE.search(chunk.summary or ""):
        score += 3
    if chunk.summary:
        score += 1
    if total > 0 and index >= max(0, total - 6):
        score += 1
    if 20 <= len(text) <= 600:
        score += 1
    return score


def build_skill_evidence(
    chunks: list[Chunk],
    *,
    signal_score: SignalScore | None = None,
) -> str:
    conv = [chunk for chunk in chunks if chunk.role in ("user", "assistant")]
    if not conv:
        return "(no supporting evidence)"

    score_chunk = chunk_signal_score if signal_score is None else signal_score

    def format_chunk(chunk: Chunk) -> str:
        label = "User" if chunk.role == "user" else "Assistant"
        text = (chunk.summary or chunk.content).strip()
        if len(text) > 700:
            text = text[:697] + "..."
        return f"[{label}] {text}"

    first_user = next(
        (chunk for chunk in conv if chunk.role == "user" and chunk.content.strip()),
        None,
    )

    selected: list[Chunk] = []
    seen_texts: set[str] = set()

    def add(chunk: Chunk | None) -> None:
        if not chunk:
            return
        key = ((chunk.summary or chunk.content).strip() or chunk.id).lower()
        if key in seen_texts:
            return
        seen_texts.add(key)
        selected.append(chunk)

    add(first_user)

    error_candidate = next(
        (
            chunk
            for chunk in conv
            if RESULT_SIGNAL_RE.search(chunk.content or "")
            and (
                "报错" in chunk.content
                or "错误" in chunk.content
                or "error" in chunk.content.lower()
            )
        ),
        None,
    )
    add(error_candidate)

    scored = sorted(
        (
            (score_chunk(chunk, index, len(conv)), index, chunk)
            for index, chunk in enumerate(conv)
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    for score, _index, chunk in scored:
        if score <= 0:
            continue
        add(chunk)
        if len(selected) >= 8:
            break

    result_candidate = next(
        (
            chunk
            for chunk in reversed(conv)
            if RESULT_SIGNAL_RE.search(chunk.content or "")
            or RESULT_SIGNAL_RE.search(chunk.summary or "")
        ),
        None,
    )
    add(result_candidate)

    return "\n".join(
        f"{index + 1}. {format_chunk(chunk)}"
        for index, chunk in enumerate(selected[:8])
    )
