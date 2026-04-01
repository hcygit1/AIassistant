"""Thresholds and directory constants for tool result persistence."""

import math

# Characters above which a tool result is written to disk instead of being
# passed verbatim into the LLM context.
DEFAULT_MAX_RESULT_SIZE_CHARS: int = 50_000

# Number of characters shown in the inline preview (first N chars).
PREVIEW_SIZE_CHARS: int = 2_000

# Sub-directory under  <data_dir>/<agent_id>/sessions/<session_id>/
TOOL_RESULTS_SUBDIR: str = "tool-results"

# Hard cap on a single persisted file (64 MiB).
MAX_PERSISTED_FILE_BYTES: int = 64 * 1024 * 1024

# Per-tool overrides: tools whose output tends to be large get a tighter
# threshold so the persistence kicks in earlier.
TOOL_SPECIFIC_OVERRIDES: dict[str, int] = {
    "exec": 40_000,
    "python_repl": 20_000,
    "read": 40_000,
    "grep": 30_000,
    "web_search": 40_000,
    "web_fetch": 40_000,
}

# Tools that should never be persisted (output too small / structural).
NEVER_PERSIST_TOOLS: frozenset[str] = frozenset(
    {
        "memory_search",
        "memory_get",
        "status",
        "process_list",
        "process_kill",
        "apply_patch",
    }
)


def get_threshold(tool_name: str) -> float:
    """Return the effective persistence threshold (chars) for *tool_name*."""
    if tool_name in NEVER_PERSIST_TOOLS:
        return math.inf
    return float(TOOL_SPECIFIC_OVERRIDES.get(tool_name, DEFAULT_MAX_RESULT_SIZE_CHARS))
