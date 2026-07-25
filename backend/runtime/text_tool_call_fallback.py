"""Execute tool calls emitted as model text instead of native events."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Awaitable, Callable

from infra.event_bus import Events
from runtime.turn_event_stream import TerminalTurnError, TurnEventStreamState


logger = logging.getLogger(__name__)


class TextToolCallFallbackProcessor:
    """Translate parsed text tool calls into the normal tool event contract."""

    def __init__(
        self,
        *,
        request: Any,
        turn: Any,
        state: TurnEventStreamState,
        run_tracker: Any,
        audit_logger: Any,
        emit_event: Callable[[str, dict[str, Any]], None],
        loop_detector: Any,
        parse_text_tool_calls: Callable[[str], list[Any]],
        strip_tool_call_patterns: Callable[[str], str],
        invoke_tool_async: Callable[..., Awaitable[str]],
        format_tool_error: Callable[[str, Exception], str],
        is_untrusted_source_tool: Callable[[str], bool],
        new_tool_call_id: Callable[[], str],
        infer_tool_result_status: Callable[[str], tuple[str, str | None]],
        loop_warning_is_breaker: Callable[[str], bool],
        log: Any = logger,
    ) -> None:
        self._request = request
        self._turn = turn
        self.state = state
        self._run_tracker = run_tracker
        self._audit_logger = audit_logger
        self._emit_event = emit_event
        self._loop_detector = loop_detector
        self._parse_text_tool_calls = parse_text_tool_calls
        self._strip_tool_call_patterns = strip_tool_call_patterns
        self._invoke_tool_async = invoke_tool_async
        self._format_tool_error = format_tool_error
        self._is_untrusted_source_tool = is_untrusted_source_tool
        self._new_tool_call_id = new_tool_call_id
        self._infer_tool_result_status = infer_tool_result_status
        self._loop_warning_is_breaker = loop_warning_is_breaker
        self._logger = log

    async def stream(self) -> AsyncGenerator[dict[str, Any], None]:
        state = self.state
        parsed_calls = self._parse_text_tool_calls(state.full_response)
        if not parsed_calls or state.tool_calls_log:
            return

        if not state.content_refresh_sent:
            yield {
                "type": "content_refresh",
                "content": self._strip_tool_call_patterns(
                    state.full_response
                ),
            }
            state.content_refresh_sent = True

        tools_by_name = {
            getattr(tool, "name", ""): tool
            for tool in self._request.tools
        }
        for tool_name, tool_args in parsed_calls:
            matched_tool = tools_by_name.get(tool_name)
            if not matched_tool:
                continue
            state.step_count += 1
            args = dict(tool_args) if tool_args else {}
            if tool_name == "read" and not args.get("path"):
                args["path"] = "IDENTITY.md"
                self._logger.info(
                    "Fallback read: 无 path 参数，使用默认 IDENTITY.md"
                )
            tool_call_id = self._new_tool_call_id()
            self._run_tracker.record_tool_start(
                self._turn.run_id,
                tool_name,
                args,
                tool_call_id=tool_call_id,
            )
            self._logger.info(
                "Fallback tool call: %s(%s)",
                tool_name,
                args,
            )
            yield {
                "type": "tool_start",
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "input": args,
                "step": state.step_count,
                "max_steps": self._request.recursion_limit,
            }
            try:
                result = (
                    await self._invoke_tool_async(
                        matched_tool,
                        args,
                        user_message=self._request.message,
                        recent_untrusted_content=(
                            state.recent_untrusted_content
                        ),
                    )
                )[:2000]
            except Exception as error:
                result = self._format_tool_error(tool_name, error)
            if self._is_untrusted_source_tool(tool_name):
                state.recent_untrusted_content = True
            status, tool_error = self._infer_tool_result_status(result)
            self._run_tracker.record_tool_end(
                self._turn.run_id,
                tool_name,
                result,
                error=tool_error,
                tool_call_id=tool_call_id,
            )
            self._audit_logger.log_tool_call(
                self._request.agent_id,
                self._turn.run_id,
                tool_name,
                args,
                result,
                tool_call_id=tool_call_id,
                status=status,
                error=tool_error,
            )
            yield {
                "type": "tool_end",
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "status": status,
                "error": tool_error,
                "output": result,
            }
            state.tool_calls_log.append(
                {
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                    "status": status,
                    "input": args,
                    "output": result,
                    "error": tool_error,
                }
            )
            loop_warning = self._loop_detector.record(
                tool_name,
                args,
                result,
            )
            if loop_warning:
                self._audit_logger.log_tool_loop_warning(
                    self._request.agent_id,
                    self._turn.run_id,
                    tool_name,
                    loop_warning,
                    tool_call_id=tool_call_id,
                )
                loop_event = Events.tool_loop_warning(
                    run_id=self._turn.run_id,
                    tool=tool_name,
                    warning=loop_warning,
                    tool_call_id=tool_call_id,
                )
                self._emit_event(self._request.agent_id, loop_event)
                yield loop_event
                if self._loop_warning_is_breaker(loop_warning):
                    raise TerminalTurnError(loop_warning)

        state.full_response = self._strip_tool_call_patterns(
            state.full_response
        )
