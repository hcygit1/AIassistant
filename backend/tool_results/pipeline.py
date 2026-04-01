"""High-level entry point: decide whether to persist a tool result."""

from __future__ import annotations

import math
import uuid
from pathlib import Path

from tool_results.constants import get_threshold
from tool_results.models import PersistToolResultError
from tool_results.storage import (
    build_persisted_output_message,
    get_tool_results_dir,
    is_already_persisted,
    persist_tool_result,
)


def maybe_persist_tool_output(
    output: str,
    tool_name: str,
    data_dir: str,
    agent_id: str,
    session_id: str,
    *,
    file_stem: str | None = None,
) -> str:
    """Return *output* unchanged, or a ``<persisted-output>`` replacement.

    Parameters
    ----------
    output:
        The raw string returned by the tool.
    tool_name:
        Used to look up the per-tool threshold.
    data_dir / agent_id / session_id:
        Used to build the on-disk path
        ``<data_dir>/<agent_id>/sessions/<session_id>/tool-results/``.
    file_stem:
        Optional explicit filename stem (without extension).  Defaults to a
        short UUID so each invocation gets a unique file.
    """
    if not output or not output.strip():
        return output

    if is_already_persisted(output):
        return output

    threshold = get_threshold(tool_name)
    if math.isinf(threshold) or len(output) <= threshold:
        return output

    stem = file_stem or f"{tool_name}_{uuid.uuid4().hex[:12]}"
    results_dir: Path = get_tool_results_dir(data_dir, agent_id, session_id)

    outcome = persist_tool_result(output, stem, results_dir)

    if isinstance(outcome, PersistToolResultError):
        # Persistence failed — fall back to inline truncation so the model
        # still gets something useful rather than nothing.
        import logging
        logging.getLogger(__name__).warning(
            "tool_result_persist failed for %s: %s", tool_name, outcome.error
        )
        return output

    return build_persisted_output_message(outcome)
