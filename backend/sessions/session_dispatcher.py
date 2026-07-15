"""统一会话调度器 — 用户消息与系统消息同队列、按优先级 + aging 串行消费

这个模块负责“会话正式工作项”的排队与执行。

这里的 SessionWorkItem 可以理解为 conversation work item：

- user：用户正式对话
- announce：子 Agent 向 requester 的正式汇报
- heartbeat：心跳检查形成的正式会话工作
- cron：带提醒内容的系统工作

职责：

- 维护每个 session 的正式工作队列
- 按优先级和 aging 串行执行
- 复用同一把 session 级锁，避免同一会话并发写入

非职责：

- 不保存用户 turn 的外部状态查询模型（那是 UserTurnCoordinator / user turn runtime 视图）
- 不负责前端 UI 广播（那是 event_bus）
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ANNOUNCE_TIMEOUT_SEC = 60
SYSTEM_TIMEOUT_SEC = 5

# 用户优先于所有系统任务；系统 aging 不低于 PRIORITY_MIN_SYSTEM（避免压过用户）
PRIORITY_USER = -10
PRIORITY_MIN_SYSTEM = -9
PRIORITY_ANNOUNCE = 0
PRIORITY_CRON = 2
PRIORITY_HEARTBEAT = 3

AGING_INTERVAL_SEC = 30.0
MAX_AGING_BONUS = 3.0


def _parse_sse_event(line: str) -> tuple[str, dict[str, Any]]:
    event_type = ""
    payload: dict[str, Any] = {}
    for raw_line in line.splitlines():
        if raw_line.startswith("event: "):
            event_type = raw_line[7:].strip()
        elif raw_line.startswith("data: "):
            try:
                parsed = json.loads(raw_line[6:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
    return event_type, payload


@dataclass(order=False)
class SessionWorkItem:
    """会话正式工作项。

    这是调度器内部真正要执行的载体，而不是 API 暴露给前端的 turn 状态。
    对用户消息来说，它与 user turn runtime/status 指向同一条业务请求的两个侧面：

    - user turn runtime/status：状态与可观测性
    - SessionWorkItem：调度与执行
    """

    kind: str
    priority: int
    content: str
    agent_id: str
    session_id: str
    prompt_mode: str = "minimal"
    persist_role: str = "system"
    run_id: str | None = None
    work_id: str | None = None
    created_at: float = field(default_factory=time.time)
    result_handler: Callable[[str], Awaitable[None]] | None = None
    on_success: Callable[[], Any] | None = None
    on_failure: Callable[[], Any] | None = None
    on_failure_async: Callable[[Exception], Awaitable[None]] | None = None
    turn_id: str | None = None
    stream_queue: asyncio.Queue[str | None] | None = None

class SessionDispatcher:
    """每个 session 一个实例，后台单协程按优先级 + aging 消费队列。"""

    def __init__(self, lock: asyncio.Lock):
        self._lock = lock
        self._queue: list[SessionWorkItem] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._current_executing_turn_id: str | None = None

    @property
    def current_executing_turn_id(self) -> str | None:
        return self._current_executing_turn_id

    def _effective_priority(self, task: SessionWorkItem) -> float:
        now = time.time()
        if task.kind == "user":
            return float(PRIORITY_USER)
        age = now - task.created_at
        bonus = min(age / AGING_INTERVAL_SEC, MAX_AGING_BONUS)
        eff = float(task.priority) - bonus
        return max(eff, float(PRIORITY_MIN_SYSTEM))

    def _sort_key(self, t: SessionWorkItem) -> tuple[float, float]:
        return (self._effective_priority(t), t.created_at)

    def submit(self, task: SessionWorkItem) -> int:
        self._queue.append(task)
        self._queue.sort(key=self._sort_key)
        self._wake.set()
        return len(self._queue)

    def turn_queue_position(self, turn_id: str) -> int | None:
        """整队列中 1-based 位置；若不在队列中则 None。"""
        self._queue.sort(key=self._sort_key)
        for i, t in enumerate(self._queue):
            if t.turn_id == turn_id:
                return i + 1
        return None

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._consume_loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    async def _consume_loop(self) -> None:
        while True:
            if not self._queue:
                self._wake.clear()
                await self._wake.wait()

            while self._queue:
                self._queue.sort(key=self._sort_key)
                task = self._queue.pop(0)
                try:
                    await self._execute(task)
                finally:
                    del task

    async def _execute(self, task: SessionWorkItem) -> None:
        if task.kind == "user":
            await self._execute_user(task)
        else:
            await self._execute_system(task)

    async def _execute_user(self, task: SessionWorkItem) -> None:
        from runtime.user_turn_stream import iter_user_turn_sse
        from turns.coordinator import user_turn_coordinator

        if not task.turn_id or not task.stream_queue:
            logger.error("user task missing turn_id or stream_queue")
            return

        lock_acquired = False

        try:
            await self._lock.acquire()
            lock_acquired = True
        except asyncio.CancelledError:
            user_turn_coordinator.set_error(task.turn_id, "cancelled before lock")
            await task.stream_queue.put(None)
            return

        self._current_executing_turn_id = task.turn_id
        user_turn_coordinator.set_running(task.turn_id)

        async def _run_user_inner() -> tuple[str, str | None]:
            terminal_status = "done"
            terminal_error: str | None = None
            async for line in iter_user_turn_sse(
                task.content,
                task.session_id,
                task.agent_id,
                task.turn_id,
            ):
                await task.stream_queue.put(line)
                event_type, payload = _parse_sse_event(line)
                if terminal_status == "done" and event_type == "error":
                    terminal_status = "error"
                    terminal_error = str(
                        payload.get("error") or "user turn failed"
                    )
                elif terminal_status == "done" and event_type == "aborted":
                    terminal_status = "cancelled"
            return terminal_status, terminal_error

        inner = asyncio.create_task(_run_user_inner())
        user_turn_coordinator.bind_execution_task(task.turn_id, inner)

        try:
            terminal_status, terminal_error = await inner
        except asyncio.CancelledError:
            user_turn_coordinator.set_cancelled(task.turn_id)
        except Exception as e:
            logger.error("User turn execution failed: %s", e)
            err_line = json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False)
            await task.stream_queue.put(f"event: error\ndata: {err_line}\n\n")
            user_turn_coordinator.set_error(task.turn_id, str(e))
        else:
            if terminal_status == "error":
                user_turn_coordinator.set_error(
                    task.turn_id,
                    terminal_error or "user turn failed",
                )
            elif terminal_status == "cancelled":
                user_turn_coordinator.set_cancelled(task.turn_id)
            else:
                user_turn_coordinator.set_done(task.turn_id)
        finally:
            self._current_executing_turn_id = None
            user_turn_coordinator.bind_execution_task(task.turn_id, None)
            try:
                await task.stream_queue.put(None)
            except Exception:
                pass
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass

    async def _execute_system(self, task: SessionWorkItem) -> None:
        from runtime.agent import agent_manager
        from sessions.session_work_store import session_work_store

        timeout = ANNOUNCE_TIMEOUT_SEC if task.kind == "announce" else SYSTEM_TIMEOUT_SEC
        lock_acquired = False
        if task.work_id:
            session_work_store.mark_running(task.work_id)

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            lock_acquired = True
        except asyncio.TimeoutError as te:
            logger.warning(
                "Dispatcher timeout acquiring lock for %s (priority=%d, timeout=%ds)",
                task.kind, task.priority, timeout,
            )
            if task.on_failure_async:
                try:
                    await task.on_failure_async(te)
                except Exception as e2:
                    logger.warning("on_failure_async (lock timeout): %s", e2)
            elif task.on_failure:
                _safe_call(task.on_failure)
            if task.work_id:
                session_work_store.mark_failed(task.work_id, str(te))
            return

        try:
            response_parts: list[str] = []
            done_content: str | None = None
            async for event in agent_manager.astream(
                message=task.content,
                session_id=task.session_id,
                agent_id=task.agent_id,
                prompt_mode=task.prompt_mode,
                persist_input_role=task.persist_role,
            ):
                et = event.get("type")
                if et == "token":
                    response_parts.append(event.get("content", ""))
                elif et == "done":
                    c = event.get("content")
                    if isinstance(c, str) and c.strip():
                        done_content = c

            response = "".join(response_parts).strip()
            if done_content is not None and done_content.strip():
                response = done_content.strip()

            if task.result_handler:
                await task.result_handler(response)

            if task.on_success:
                _safe_call(task.on_success)
            if task.work_id:
                session_work_store.mark_done(task.work_id)

        except Exception as e:
            logger.error("Dispatcher execution failed for %s: %s", task.kind, e)
            if task.on_failure_async:
                try:
                    await task.on_failure_async(e)
                except Exception as e2:
                    logger.warning("on_failure_async failed: %s", e2)
            elif task.on_failure:
                _safe_call(task.on_failure)
            if task.work_id:
                session_work_store.mark_failed(task.work_id, str(e))
        finally:
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass


def _safe_call(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception as e:
        logger.warning("Dispatcher callback error: %s", e)


class DispatcherManager:
    """全局管理器：每个 agent:session 一个 dispatcher。"""

    def __init__(self):
        self._dispatchers: dict[str, SessionDispatcher] = {}

    def get(self, agent_id: str, session_id: str, lock: asyncio.Lock) -> SessionDispatcher:
        key = f"{agent_id}:{session_id}"
        if key not in self._dispatchers:
            d = SessionDispatcher(lock=lock)
            d.start()
            self._dispatchers[key] = d
        return self._dispatchers[key]

    def cleanup(self, agent_id: str, session_id: str) -> None:
        key = f"{agent_id}:{session_id}"
        d = self._dispatchers.pop(key, None)
        if d:
            d.stop()


dispatcher_manager = DispatcherManager()
