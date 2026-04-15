"""聊天 API — submit / status / stream / pending-turn / abort（统一队列）

本文件只负责 API 入口，不负责真正执行对话。

一条用户消息进入系统时会同时生成两样东西：

1. 用户 turn 状态 / 运行时视图
   由用户 turn 协调层维护，
   代表“这条用户 turn 当前处于什么状态”，
   供 status / stream / pending-turn / abort 等 API 查询。
2. SessionWorkItem
   提交给 SessionDispatcher，代表“这条消息要如何进入会话级调度器执行”。

可以把这两者理解为：

- user turn runtime/status = 状态侧 / 可观测侧
- SessionWorkItem = 执行侧 / 调度侧
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatSubmitRequest(BaseModel):
    message: str
    session_id: str = ""
    agent_id: str = "main"


class ChatAbortRequest(BaseModel):
    session_id: str = ""
    agent_id: str = "main"
    user_initiated: bool = True
    turn_id: str = ""


@router.post("/chat/submit")
async def chat_submit(req: ChatSubmitRequest):
    from sessions.session_manager import session_manager
    from turns.service import user_turn_service

    session_id = req.session_id or session_manager.resolve_main_session_id(req.agent_id)
    result = await user_turn_service.submit(req.message, req.agent_id, session_id)
    return JSONResponse(status_code=202, content=result)


@router.get("/chat/turn/{turn_id}/status")
async def chat_turn_status(turn_id: str):
    from turns.service import user_turn_service

    return await user_turn_service.status(turn_id)


@router.get("/chat/turn/{turn_id}/stream")
async def chat_turn_stream(turn_id: str):
    from turns.service import user_turn_service

    return StreamingResponse(
        user_turn_service.stream(turn_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/pending-turn")
async def chat_pending_turn(
    session_id: str = Query(""),
    agent_id: str = Query("main"),
):
    from sessions.session_manager import session_manager
    from turns.service import user_turn_service

    sid = session_id or session_manager.resolve_main_session_id(agent_id)
    return await user_turn_service.pending(agent_id, sid)


@router.post("/chat/abort")
async def abort_chat(req: ChatAbortRequest):
    from sessions.session_manager import session_manager
    from turns.service import user_turn_service

    session_id = req.session_id or session_manager.resolve_main_session_id(req.agent_id)
    return await user_turn_service.abort(
        req.agent_id,
        session_id,
        turn_id=req.turn_id,
        user_initiated=req.user_initiated,
    )


# 旧版一体 SSE 已移除；保留占位说明（可选）
@router.post("/chat")
async def chat_deprecated():
    raise HTTPException(
        status_code=410,
        detail="Use POST /api/chat/submit then GET /api/chat/turn/{id}/status and /stream",
    )
