"""Tool result persistence layer.

Large tool outputs are written to disk instead of being passed verbatim into
the LLM context.  Only a short preview and the file path are returned so the
model can fetch the full content on demand via read / grep.
"""

from tool_results.pipeline import maybe_persist_tool_output, get_tool_results_dir

__all__ = ["maybe_persist_tool_output", "get_tool_results_dir"]
