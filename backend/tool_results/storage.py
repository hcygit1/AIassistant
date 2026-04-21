"""Disk persistence helpers for large tool results."""

from __future__ import annotations

import logging
from pathlib import Path

from tool_results.constants import PREVIEW_SIZE_CHARS, TOOL_RESULTS_SUBDIR
from tool_results.models import PersistedToolResult, PersistToolResultError, PersistOutcome
from runtime.source_sink_guard import wrap_untrusted_content

logger = logging.getLogger(__name__)

PERSISTED_OUTPUT_OPEN = "<persisted-output>"
PERSISTED_OUTPUT_CLOSE = "</persisted-output>"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_tool_results_dir(data_dir: str, agent_id: str, session_id: str) -> Path:
    return Path(data_dir) / agent_id / "sessions" / session_id / TOOL_RESULTS_SUBDIR


def ensure_tool_results_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def generate_preview(text: str, max_chars: int = PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Return (preview, has_more).

    Tries to cut at a newline boundary within the last 50 % of *max_chars*
    to avoid splitting mid-line.
    """
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    last_nl = truncated.rfind("\n")
    cut = last_nl if last_nl > max_chars * 0.5 else max_chars
    return text[:cut], True


# ---------------------------------------------------------------------------
# Human-readable size
# ---------------------------------------------------------------------------

def _human_size(n: int) -> str:
    if n < 1_024:
        return f"{n} characters"
    if n < 1_024 * 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n / (1_024 * 1_024):.1f} MB"


# ---------------------------------------------------------------------------
# Write to disk
# ---------------------------------------------------------------------------

def persist_tool_result(
    content: str,
    file_stem: str,
    tool_results_dir: Path,
) -> PersistOutcome:
    """Write *content* to ``{tool_results_dir}/{file_stem}.txt``.

    Uses exclusive-create (``open(..., "x")``) so repeated calls with the
    same *file_stem* are idempotent — the first write wins.
    """
    ensure_tool_results_dir(tool_results_dir)
    filepath = tool_results_dir / f"{file_stem}.txt"

    try:
        with filepath.open("x", encoding="utf-8") as fh:
            fh.write(content)
    except FileExistsError:
        pass  # idempotent: existing file is the canonical result
    except OSError as exc:
        return PersistToolResultError(error=str(exc))

    preview, has_more = generate_preview(content)
    return PersistedToolResult(
        filepath=str(filepath.resolve()),
        original_size=len(content),
        preview=preview,
        has_more=has_more,
    )


# ---------------------------------------------------------------------------
# Build the replacement message shown to the model
# ---------------------------------------------------------------------------

def build_persisted_output_message(result: PersistedToolResult) -> str:
    body_lines = [
        PERSISTED_OUTPUT_OPEN,
        (
            f"Output too large ({_human_size(result.original_size)}). "
            f"Full output saved to: {result.filepath}"
        ),
        "",
        f"Preview (first {PREVIEW_SIZE_CHARS} chars):",
        result.preview,
    ]
    if result.has_more:
        body_lines.extend(["", "..."])
    body_lines.append(PERSISTED_OUTPUT_CLOSE)
    body = "\n".join(body_lines)
    return wrap_untrusted_content(
        body,
        source_type="persisted_tool_output",
    )


def is_already_persisted(text: str) -> bool:
    """True if *text* was already replaced by a persisted-output block."""
    return text.startswith(PERSISTED_OUTPUT_OPEN)
