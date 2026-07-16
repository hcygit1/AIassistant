"""Execute one Agent turn against one concrete model."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable

from infra.event_bus import Events
from llm.model_selection import ModelCandidateError
from llm.models_config import ModelRef
from runtime.security_context import (
    mark_recent_untrusted_content,
    runtime_security_context,
)
from runtime.source_sink_guard import (
    contains_untrusted_marker,
    is_untrusted_source_tool,
)
from runtime.tool_call_parser import (
    parse_text_tool_calls,
    strip_tool_call_patterns,
)
from runtime.tool_execution import invoke_tool_async
from runtime.turn_models import TurnExecutionRequest
from sandbox.loop_detection import LoopDetector
from tools.error_utils import format_tool_error


logger = logging.getLogger(__name__)


class _TerminalTurnError(RuntimeError):
    pass


def should_persist_input_message(persist_input_role: str) -> bool:
    return bool((persist_input_role or "").strip())


def _new_tool_call_id() -> str:
    return f"tc_{uuid.uuid4().hex[:12]}"


def _infer_tool_result_status(
    output: str,
) -> tuple[str, str | None]:
    text = output or ""
    lowered = text.lower()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            status = str(parsed.get("status") or "").lower()
            if status == "error":
                return (
                    "error",
                    str(parsed.get("error") or "")[:500] or None,
                )
    except Exception:
        pass
    if (
        "命令被拒绝" in text
        or "command rejected" in lowered
        or "已拒绝" in text
    ):
        return "denied", text[:500]
    if "timed out" in lowered or "超时" in text:
        return "timeout", text[:500]
    if (
        "execution error" in lowered
        or "执行错误" in text
        or "执行出错" in text
    ):
        return "error", text[:500]
    return "success", None


def _loop_warning_is_breaker(warning: str) -> bool:
    return "全局熔断" in warning or "circuit" in warning.lower()


class TurnExecutor:
    def __init__(
        self,
        *,
        create_llm: Callable[[ModelRef], Any],
        build_messages: Callable[
            [list[dict[str, Any]], str],
            list[Any],
        ],
        get_lifecycle_hooks: Callable[[], Any],
        get_run_tracker: Callable[[], Any],
        get_audit_logger: Callable[[], Any],
        save_message: Callable[..., Any],
        write_skills_snapshot: Callable[[str], None],
        emit_event: Callable[[str, dict[str, Any]], None],
        count_tokens: Callable[[str], int],
        incremental_ingest: Callable[
            [str, str, str, str],
            Awaitable[None],
        ],
        get_pending_tasks: Callable[[], set[asyncio.Task]],
        maybe_auto_compact: Callable[..., Awaitable[None]],
    ) -> None:
        self._create_llm = create_llm
        self._build_messages = build_messages
        self._get_lifecycle_hooks = get_lifecycle_hooks
        self._get_run_tracker = get_run_tracker
        self._get_audit_logger = get_audit_logger
        self._save_message = save_message
        self._write_skills_snapshot = write_skills_snapshot
        self._emit_event = emit_event
        self._count_tokens = count_tokens
        self._incremental_ingest = incremental_ingest
        self._get_pending_tasks = get_pending_tasks
        self._maybe_auto_compact = maybe_auto_compact

    async def execute(
        self,
        request: TurnExecutionRequest,
    ) -> AsyncGenerator[dict[str, Any], None]:
        ref = ModelRef(
            provider=request.provider,
            model=request.model,
        )
        try:
            llm = self._create_llm(ref)
        except Exception as error:
            raise ModelCandidateError(
                str(error)
            ) from error

        try:
            from langgraph.prebuilt import create_react_agent

            agent = create_react_agent(
                model=llm,
                tools=request.tools,
                prompt=request.system_prompt,
            )
        except ImportError:
            yield {
                "type": "error",
                "error": "langgraph 未安装",
            }
            return
        except Exception as error:
            yield Events.turn_error(error=str(error))
            yield {
                "type": "error",
                "error": str(error),
            }
            return

        turn = None
        run_tracker = None
        audit_logger = None
        try:
            messages = self._build_messages(
                request.history,
                request.message,
            )
            run_tracker = self._get_run_tracker()
            audit_logger = self._get_audit_logger()
            turn = run_tracker.start_turn(
                request.agent_id,
                request.session_id,
            )
            audit_logger.log_turn_start(
                request.agent_id,
                turn.run_id,
                request.session_id,
            )
        except Exception as error:
            error_text = str(error)
            if turn is not None and run_tracker is not None:
                run_tracker.error_turn(
                    turn.run_id,
                    error_text,
                )
                if audit_logger is not None:
                    try:
                        audit_logger.log_turn_error(
                            request.agent_id,
                            turn.run_id,
                            error_text,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to audit setup error "
                            "for turn %s",
                            turn.run_id,
                        )
            yield Events.turn_error(error=error_text)
            yield {
                "type": "error",
                "error": error_text,
            }
            return
        turn_start_event = Events.turn_start(
            run_id=turn.run_id,
            model=str(ref),
        )
        turn_start_emitted = False

        full_response = ""
        tool_calls_log: list[dict[str, Any]] = []
        tool_input_by_run_id: dict[str, Any] = {}
        tool_call_id_by_run_id: dict[str, str] = {}
        streaming_model_run_id: str | None = None
        step_count = 0
        content_refresh_sent = False
        recent_untrusted_content = any(
            contains_untrusted_marker(
                str(message.get("content", ""))
            )
            for message in request.history[-4:]
        )
        loop_detector = LoopDetector()

        async def stream_model_events():
            iterator = agent.astream_events(
                {"messages": messages},
                version="v2",
                config={
                    "recursion_limit": request.recursion_limit
                },
            ).__aiter__()
            while True:
                try:
                    yield await anext(iterator)
                except StopAsyncIteration:
                    return
                except Exception as error:
                    raise ModelCandidateError(
                        str(error)
                    ) from error

        try:
            with runtime_security_context(
                request.message,
                recent_untrusted_content=recent_untrusted_content,
            ):
                async for event in stream_model_events():
                    if not turn_start_emitted:
                        yield turn_start_event
                        turn_start_emitted = True
                    kind = event.get("event", "")
                    if kind == "on_chat_model_stream":
                        event_run_id = event.get("run_id", "")
                        if streaming_model_run_id is None:
                            streaming_model_run_id = event_run_id
                        elif event_run_id != streaming_model_run_id:
                            continue

                        chunk = event.get("data", {}).get("chunk")
                        if (
                            chunk
                            and hasattr(chunk, "content")
                            and chunk.content
                        ):
                            content = chunk.content
                            if isinstance(content, str):
                                full_response += content
                                yield {
                                    "type": "token",
                                    "content": content,
                                }
                            elif isinstance(content, list):
                                for block in content:
                                    if (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ):
                                        text = block.get("text", "")
                                        if text:
                                            full_response += text
                                            yield {
                                                "type": "token",
                                                "content": text,
                                            }

                        if (
                            chunk
                            and hasattr(chunk, "usage_metadata")
                            and chunk.usage_metadata
                        ):
                            usage = chunk.usage_metadata
                            input_details = getattr(
                                usage,
                                "input_token_details",
                                {},
                            )
                            run_tracker.record_tokens(
                                turn.run_id,
                                input_tokens=getattr(
                                    usage,
                                    "input_tokens",
                                    0,
                                ),
                                output_tokens=getattr(
                                    usage,
                                    "output_tokens",
                                    0,
                                ),
                                cache_read=(
                                    input_details.get(
                                        "cache_read",
                                        0,
                                    )
                                    if hasattr(
                                        usage,
                                        "input_token_details",
                                    )
                                    else 0
                                ),
                            )

                    elif kind == "on_chat_model_end":
                        if (
                            event.get("run_id")
                            == streaming_model_run_id
                        ):
                            streaming_model_run_id = None

                    elif kind == "on_tool_start":
                        if (
                            not content_refresh_sent
                            and full_response
                            and parse_text_tool_calls(full_response)
                        ):
                            yield {
                                "type": "content_refresh",
                                "content": strip_tool_call_patterns(
                                    full_response
                                ),
                            }
                            content_refresh_sent = True

                        tool_name = event.get("name", "")
                        tool_input = (
                            event.get("data", {}).get("input") or {}
                        )
                        if not isinstance(tool_input, dict):
                            tool_input = {}
                        lifecycle_hooks = (
                            self._get_lifecycle_hooks()
                        )
                        if lifecycle_hooks:
                            await lifecycle_hooks.on_before_tool_call(
                                request.agent_id,
                                turn.run_id,
                                tool_name,
                                tool_input,
                            )
                        step_count += 1
                        event_run_id = str(
                            event.get("run_id", "")
                        )
                        tool_call_id = _new_tool_call_id()
                        if event_run_id:
                            tool_input_by_run_id[
                                event_run_id
                            ] = tool_input
                            tool_call_id_by_run_id[
                                event_run_id
                            ] = tool_call_id
                        run_tracker.record_tool_start(
                            turn.run_id,
                            tool_name,
                            tool_input,
                            tool_call_id=tool_call_id,
                        )
                        yield {
                            "type": "tool_start",
                            "tool_call_id": tool_call_id,
                            "tool": tool_name,
                            "input": tool_input,
                            "step": step_count,
                            "max_steps": request.recursion_limit,
                        }

                    elif kind == "on_tool_end":
                        tool_output = (
                            event.get("data", {}).get("output", "")
                        )
                        if isinstance(tool_output, str):
                            output = tool_output
                        elif (
                            hasattr(tool_output, "content")
                            and tool_output.content is not None
                        ):
                            output = str(tool_output.content)
                        else:
                            output = str(tool_output)

                        event_run_id = str(
                            event.get("run_id", "")
                        )
                        tool_input = tool_input_by_run_id.pop(
                            event_run_id,
                            None,
                        )
                        tool_call_id = (
                            tool_call_id_by_run_id.pop(
                                event_run_id,
                                None,
                            )
                            or _new_tool_call_id()
                        )
                        tool_input_for_log = (
                            tool_input
                            if tool_input is not None
                            else ""
                        )
                        tool_name = event.get("name", "")
                        status, error = _infer_tool_result_status(
                            output
                        )
                        run_tracker.record_tool_end(
                            turn.run_id,
                            tool_name,
                            output,
                            error=error,
                            tool_call_id=tool_call_id,
                        )
                        audit_logger.log_tool_call(
                            request.agent_id,
                            turn.run_id,
                            tool_name,
                            tool_input_for_log,
                            output,
                            tool_call_id=tool_call_id,
                            status=status,
                            error=error,
                        )
                        tool_calls_log.append(
                            {
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "status": status,
                                "input": tool_input_for_log,
                                "output": output,
                                "error": error,
                            }
                        )
                        lifecycle_hooks = (
                            self._get_lifecycle_hooks()
                        )
                        if lifecycle_hooks:
                            await lifecycle_hooks.on_after_tool_call(
                                request.agent_id,
                                turn.run_id,
                                tool_name,
                                tool_input_for_log,
                                output,
                            )
                        if is_untrusted_source_tool(tool_name):
                            recent_untrusted_content = True
                            mark_recent_untrusted_content(True)
                        yield {
                            "type": "tool_end",
                            "tool_call_id": tool_call_id,
                            "tool": tool_name,
                            "status": status,
                            "error": error,
                            "output": output[:2000],
                        }

                        loop_warning = loop_detector.record(
                            tool_name,
                            tool_input_for_log,
                            output,
                        )
                        if loop_warning:
                            audit_logger.log_tool_loop_warning(
                                request.agent_id,
                                turn.run_id,
                                tool_name,
                                loop_warning,
                                tool_call_id=tool_call_id,
                            )
                            loop_event = Events.tool_loop_warning(
                                run_id=turn.run_id,
                                tool=tool_name,
                                warning=loop_warning,
                                tool_call_id=tool_call_id,
                            )
                            self._emit_event(
                                request.agent_id,
                                loop_event,
                            )
                            yield loop_event
                            if _loop_warning_is_breaker(
                                loop_warning
                            ):
                                raise _TerminalTurnError(
                                    loop_warning
                                )

                        if tool_name in (
                            "exec",
                            "process_kill",
                        ):
                            safe_input = (
                                str(tool_input_for_log)[:200]
                                if tool_input_for_log
                                else ""
                            )
                            self._emit_event(
                                request.agent_id,
                                Events.tool_dangerous_executed(
                                    tool=tool_name,
                                    input_preview=safe_input,
                                ),
                            )
            if not turn_start_emitted:
                yield turn_start_event
                turn_start_emitted = True
        except Exception as error:
            root_error = (
                error.__cause__
                if isinstance(error, ModelCandidateError)
                and error.__cause__ is not None
                else error
            )
            error_text = str(root_error)
            is_recursion = (
                "recursion" in error_text.lower()
                or "GraphRecursionError"
                in type(root_error).__name__
            )
            run_tracker.error_turn(turn.run_id, error_text)
            audit_logger.log_turn_error(
                request.agent_id,
                turn.run_id,
                error_text,
            )
            if is_recursion:
                if not turn_start_emitted:
                    yield turn_start_event
                    turn_start_emitted = True
                yield Events.recursion_limit_reached(
                    step=step_count,
                    max_steps=request.recursion_limit,
                )
                yield {
                    "type": "error",
                    "error": (
                        "Agent 达到最大迭代次数 "
                        f"({request.recursion_limit})，已自动停止。"
                        f"已执行 {step_count} 步工具调用。"
                    ),
                }
            elif (
                isinstance(error, _TerminalTurnError)
                or not isinstance(error, ModelCandidateError)
            ):
                if not turn_start_emitted:
                    yield turn_start_event
                    turn_start_emitted = True
                yield Events.turn_error(error=error_text)
                yield {
                    "type": "error",
                    "error": error_text,
                }
            else:
                raise
            return

        try:
            parsed_calls = parse_text_tool_calls(full_response)
            if parsed_calls and not tool_calls_log:
                if not content_refresh_sent:
                    yield {
                        "type": "content_refresh",
                        "content": strip_tool_call_patterns(
                            full_response
                        ),
                    }
                    content_refresh_sent = True
                tools_by_name = {
                    getattr(tool, "name", ""): tool
                    for tool in request.tools
                }
                for tool_name, tool_args in parsed_calls:
                    matched_tool = tools_by_name.get(tool_name)
                    if not matched_tool:
                        continue
                    step_count += 1
                    args = (
                        dict(tool_args)
                        if tool_args
                        else {}
                    )
                    if (
                        tool_name == "read"
                        and not args.get("path")
                    ):
                        args["path"] = "IDENTITY.md"
                        logger.info(
                            "Fallback read: 无 path 参数，"
                            "使用默认 IDENTITY.md"
                        )
                    tool_call_id = _new_tool_call_id()
                    run_tracker.record_tool_start(
                        turn.run_id,
                        tool_name,
                        args,
                        tool_call_id=tool_call_id,
                    )
                    logger.info(
                        "Fallback tool call: %s(%s)",
                        tool_name,
                        args,
                    )
                    yield {
                        "type": "tool_start",
                        "tool_call_id": tool_call_id,
                        "tool": tool_name,
                        "input": args,
                        "step": step_count,
                        "max_steps": request.recursion_limit,
                    }
                    try:
                        result = (
                            await invoke_tool_async(
                                matched_tool,
                                args,
                                user_message=request.message,
                                recent_untrusted_content=(
                                    recent_untrusted_content
                                ),
                            )
                        )[:2000]
                    except Exception as error:
                        result = format_tool_error(
                            tool_name,
                            error,
                        )
                    if is_untrusted_source_tool(tool_name):
                        recent_untrusted_content = True
                    status, tool_error = (
                        _infer_tool_result_status(result)
                    )
                    run_tracker.record_tool_end(
                        turn.run_id,
                        tool_name,
                        result,
                        error=tool_error,
                        tool_call_id=tool_call_id,
                    )
                    audit_logger.log_tool_call(
                        request.agent_id,
                        turn.run_id,
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
                    tool_calls_log.append(
                        {
                            "tool_call_id": tool_call_id,
                            "tool": tool_name,
                            "status": status,
                            "input": args,
                            "output": result,
                            "error": tool_error,
                        }
                    )
                    loop_warning = loop_detector.record(
                        tool_name,
                        args,
                        result,
                    )
                    if loop_warning:
                        audit_logger.log_tool_loop_warning(
                            request.agent_id,
                            turn.run_id,
                            tool_name,
                            loop_warning,
                            tool_call_id=tool_call_id,
                        )
                        loop_event = Events.tool_loop_warning(
                            run_id=turn.run_id,
                            tool=tool_name,
                            warning=loop_warning,
                            tool_call_id=tool_call_id,
                        )
                        self._emit_event(
                            request.agent_id,
                            loop_event,
                        )
                        yield loop_event
                        if _loop_warning_is_breaker(
                            loop_warning
                        ):
                            raise _TerminalTurnError(
                                loop_warning
                            )
                full_response = strip_tool_call_patterns(
                    full_response
                )

            done_content = (
                strip_tool_call_patterns(full_response)
                if parse_text_tool_calls(full_response)
                else full_response
            )
            turn_tokens = (
                self._count_tokens(request.message)
                + self._count_tokens(full_response)
            )
            for tool_call in tool_calls_log:
                turn_tokens += self._count_tokens(
                    str(tool_call.get("output", ""))
                )
            total_context = (
                request.prompt_tokens
                + request.history_tokens
                + turn_tokens
            )
            context_utilization = (
                round(
                    total_context / request.active_tokens,
                    3,
                )
                if request.active_tokens
                else 0
            )

            if should_persist_input_message(
                request.persist_input_role
            ):
                self._save_message(
                    request.session_id,
                    request.agent_id,
                    request.persist_input_role,
                    request.message,
                )
            content_to_save = (
                strip_tool_call_patterns(full_response)
                if parse_text_tool_calls(full_response)
                else full_response
            )
            self._save_message(
                request.session_id,
                request.agent_id,
                "assistant",
                content_to_save,
                tool_calls=(
                    tool_calls_log
                    if tool_calls_log
                    else None
                ),
            )
            self._write_skills_snapshot(request.agent_id)
            completed = run_tracker.complete_turn(
                turn.run_id
            )
        except Exception as error:
            error_text = str(error)
            run_tracker.error_turn(
                turn.run_id,
                error_text,
            )
            audit_logger.log_turn_error(
                request.agent_id,
                turn.run_id,
                error_text,
            )
            yield Events.turn_error(error=error_text)
            yield {
                "type": "error",
                "error": error_text,
            }
            return

        if completed:
            try:
                request.state.record_turn(
                    completed.input_tokens,
                    completed.output_tokens,
                )
                audit_logger.log_turn_end(
                    request.agent_id,
                    turn.run_id,
                    request.session_id,
                    tokens={
                        "input": completed.input_tokens,
                        "output": completed.output_tokens,
                    },
                    tool_calls=len(tool_calls_log),
                    duration_ms=completed.duration_ms,
                )
            except Exception:
                logger.exception(
                    "Failed to record completed turn %s",
                    turn.run_id,
                )

        usage_info: dict[str, Any] = {}
        if completed:
            usage_info = {
                "input_tokens": completed.input_tokens,
                "output_tokens": completed.output_tokens,
                "total_tokens": completed.total_tokens,
                "duration_ms": completed.duration_ms,
                "model": str(ref),
            }
        yield Events.turn_end(
            run_id=turn.run_id,
            usage=usage_info,
        )
        yield {
            "type": "done",
            "content": done_content,
            "session_id": request.session_id,
            "usage": usage_info,
            "context_utilization": context_utilization,
        }

        try:
            ingested_user_content = (
                request.message
                if request.persist_input_role == "user"
                else ""
            )
            pending_tasks = self._get_pending_tasks()
            task = asyncio.create_task(
                self._incremental_ingest(
                    request.agent_id,
                    request.session_id,
                    ingested_user_content,
                    done_content,
                )
            )
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
        except Exception:
            logger.exception(
                "Failed to schedule turn ingestion for %s",
                turn.run_id,
            )

        try:
            await self._maybe_auto_compact(
                request.session_id,
                request.agent_id,
                overhead_tokens=(
                    request.prompt_tokens
                    + request.summary_tokens
                ),
            )
        except Exception:
            logger.exception(
                "Failed to auto-compact after turn %s",
                turn.run_id,
            )
