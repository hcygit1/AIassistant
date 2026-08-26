"""Translate native model and tool events for one Agent turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, AsyncGenerator, Callable

from infra.event_bus import Events
from llm.model_selection import ModelCandidateError
from runtime.security_context import (
    mark_recent_untrusted_content,
    runtime_security_context,
)
from runtime.source_sink_guard import is_untrusted_source_tool


class TerminalTurnError(RuntimeError):
    pass


@dataclass
class TurnEventStreamState:
    full_response: str = ""
    tool_calls_log: list[dict[str, Any]] = field(default_factory=list)
    tool_input_by_run_id: dict[str, Any] = field(default_factory=dict)
    tool_call_id_by_run_id: dict[str, str] = field(default_factory=dict)
    streaming_model_run_id: str | None = None
    step_count: int = 0
    content_refresh_sent: bool = False
    recent_untrusted_content: bool = False
    turn_start_emitted: bool = False


class TurnEventStreamProcessor:
    """Consume LangGraph events while retaining turn orchestration outside."""

    def __init__(
        self,
        *,
        agent: Any,
        request: Any,
        messages: list[Any],
        turn: Any,
        turn_start_event: dict[str, Any],
        run_tracker: Any,
        audit_logger: Any,
        get_lifecycle_hooks: Callable[[], Any],
        emit_event: Callable[[str, dict[str, Any]], None],
        loop_detector: Any,
        new_tool_call_id: Callable[[], str],
        infer_tool_result_status: Callable[[str], tuple[str, str | None]],
        loop_warning_is_breaker: Callable[[str], bool],
        parse_text_tool_calls: Callable[[str], list[Any]] | None = None,
        strip_tool_call_patterns: Callable[[str], str] | None = None,
    ) -> None:
        if parse_text_tool_calls is None or strip_tool_call_patterns is None:
            from runtime.tool_call_parser import (
                parse_text_tool_calls as default_parse_text_tool_calls,
                strip_tool_call_patterns as default_strip_tool_call_patterns,
            )

            parse_text_tool_calls = (
                parse_text_tool_calls or default_parse_text_tool_calls
            )
            strip_tool_call_patterns = (
                strip_tool_call_patterns or default_strip_tool_call_patterns
            )
        self._agent = agent
        self._request = request
        self._messages = messages
        self._turn = turn
        self._turn_start_event = turn_start_event
        self._run_tracker = run_tracker
        self._audit_logger = audit_logger
        self._get_lifecycle_hooks = get_lifecycle_hooks
        self._emit_event = emit_event
        self._loop_detector = loop_detector
        self._new_tool_call_id = new_tool_call_id
        self._infer_tool_result_status = infer_tool_result_status
        self._loop_warning_is_breaker = loop_warning_is_breaker
        self._parse_text_tool_calls = parse_text_tool_calls
        self._strip_tool_call_patterns = strip_tool_call_patterns
        self.state = TurnEventStreamState(
            recent_untrusted_content=any(
                self._contains_untrusted_marker(
                    str(message.get("content", ""))
                )
                for message in request.history[-4:]
            )
        )

    @staticmethod
    def _contains_untrusted_marker(content: str) -> bool:
        from runtime.source_sink_guard import contains_untrusted_marker

        return contains_untrusted_marker(content)

    async def _stream_model_events(self) -> AsyncGenerator[dict[str, Any], None]:
        iterator = self._agent.astream_events(
            {"messages": self._messages},
            version="v2",
            config={"recursion_limit": self._request.recursion_limit},
        ).__aiter__()
        while True:
            try:
                yield await anext(iterator)
            except StopAsyncIteration:
                return
            except Exception as error:
                raise ModelCandidateError(str(error)) from error

    async def stream(self) -> AsyncGenerator[dict[str, Any], None]:
        state = self.state
        with runtime_security_context(
            self._request.message,
            recent_untrusted_content=state.recent_untrusted_content,
        ):
            async for event in self._stream_model_events():
                if not state.turn_start_emitted:
                    state.turn_start_emitted = True
                    yield self._turn_start_event
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    async for output_event in self._handle_model_stream(event):
                        yield output_event
                elif kind == "on_chat_model_end":
                    if event.get("run_id") == state.streaming_model_run_id:
                        state.streaming_model_run_id = None
                elif kind == "on_tool_start":
                    async for output_event in self._handle_tool_start(event):
                        yield output_event
                elif kind == "on_tool_end":
                    async for output_event in self._handle_tool_end(event):
                        yield output_event
        if not state.turn_start_emitted:
            state.turn_start_emitted = True
            yield self._turn_start_event

    async def _handle_model_stream(
        self,
        event: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        state = self.state
        event_run_id = event.get("run_id", "")
        if state.streaming_model_run_id is None:
            state.streaming_model_run_id = event_run_id
        elif event_run_id != state.streaming_model_run_id:
            return

        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            content = chunk.content
            if isinstance(content, str):
                state.full_response += content
                yield {"type": "token", "content": content}
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        text = block.get("text", "")
                        if text:
                            state.full_response += text
                            yield {"type": "token", "content": text}

        if chunk and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage = chunk.usage_metadata
            input_details = self._usage_value(usage, "input_token_details") or {}
            self._run_tracker.record_tokens(
                self._turn.run_id,
                input_tokens=self._usage_field(usage, "input_tokens", "prompt_tokens"),
                output_tokens=self._usage_field(usage, "output_tokens", "completion_tokens"),
                cache_read=(
                    self._usage_field(input_details, "cache_read", "cache_read_tokens")
                ),
            )

    @staticmethod
    def _usage_value(usage: Any, *names: str) -> Any:
        if isinstance(usage, Mapping):
            for name in names:
                value = usage.get(name)
                if value is not None:
                    return value
            return None
        for name in names:
            value = getattr(usage, name, None)
            if value is not None:
                return value
        return None

    @classmethod
    def _usage_field(cls, usage: Any, *names: str) -> int:
        """Read numeric provider usage metadata from a dict or object."""
        value = cls._usage_value(usage, *names)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    async def _handle_tool_start(
        self,
        event: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        state = self.state
        if (
            not state.content_refresh_sent
            and state.full_response
            and self._parse_text_tool_calls(state.full_response)
        ):
            yield {
                "type": "content_refresh",
                "content": self._strip_tool_call_patterns(state.full_response),
            }
            state.content_refresh_sent = True

        tool_name = event.get("name", "")
        tool_input = event.get("data", {}).get("input") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        lifecycle_hooks = self._get_lifecycle_hooks()
        if lifecycle_hooks:
            await lifecycle_hooks.on_before_tool_call(
                self._request.agent_id,
                self._turn.run_id,
                tool_name,
                tool_input,
            )
        state.step_count += 1
        event_run_id = str(event.get("run_id", ""))
        tool_call_id = self._new_tool_call_id()
        if event_run_id:
            state.tool_input_by_run_id[event_run_id] = tool_input
            state.tool_call_id_by_run_id[event_run_id] = tool_call_id
        self._run_tracker.record_tool_start(
            self._turn.run_id,
            tool_name,
            tool_input,
            tool_call_id=tool_call_id,
        )
        yield {
            "type": "tool_start",
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "input": tool_input,
            "step": state.step_count,
            "max_steps": self._request.recursion_limit,
        }

    async def _handle_tool_end(
        self,
        event: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        state = self.state
        tool_output = event.get("data", {}).get("output", "")
        if isinstance(tool_output, str):
            output = tool_output
        elif hasattr(tool_output, "content") and tool_output.content is not None:
            output = str(tool_output.content)
        else:
            output = str(tool_output)

        event_run_id = str(event.get("run_id", ""))
        tool_input = state.tool_input_by_run_id.pop(event_run_id, None)
        tool_call_id = (
            state.tool_call_id_by_run_id.pop(event_run_id, None)
            or self._new_tool_call_id()
        )
        tool_input_for_log = tool_input if tool_input is not None else ""
        tool_name = event.get("name", "")
        status, error = self._infer_tool_result_status(output)
        self._run_tracker.record_tool_end(
            self._turn.run_id,
            tool_name,
            output,
            error=error,
            tool_call_id=tool_call_id,
        )
        self._audit_logger.log_tool_call(
            self._request.agent_id,
            self._turn.run_id,
            tool_name,
            tool_input_for_log,
            output,
            tool_call_id=tool_call_id,
            status=status,
            error=error,
        )
        state.tool_calls_log.append(
            {
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "status": status,
                "input": tool_input_for_log,
                "output": output,
                "error": error,
            }
        )
        lifecycle_hooks = self._get_lifecycle_hooks()
        if lifecycle_hooks:
            await lifecycle_hooks.on_after_tool_call(
                self._request.agent_id,
                self._turn.run_id,
                tool_name,
                tool_input_for_log,
                output,
            )
        if is_untrusted_source_tool(tool_name):
            state.recent_untrusted_content = True
            mark_recent_untrusted_content(True)
        yield {
            "type": "tool_end",
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "status": status,
            "error": error,
            "output": output[:2000],
        }

        loop_warning = self._loop_detector.record(
            tool_name,
            tool_input_for_log,
            output,
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

        if tool_name in ("exec", "process_kill"):
            safe_input = (
                str(tool_input_for_log)[:200]
                if tool_input_for_log
                else ""
            )
            self._emit_event(
                self._request.agent_id,
                Events.tool_dangerous_executed(
                    tool=tool_name,
                    input_preview=safe_input,
                ),
            )
