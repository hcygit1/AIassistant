"""对话压缩 API — compress + post-compaction"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_agent_manager, get_session_manager

router = APIRouter()


@router.post("/agents/{agent_id}/sessions/{session_id}/compress")
async def compress_session(
    agent_id: str,
    session_id: str,
    agent_manager: Any = Depends(get_agent_manager),
    session_manager: Any = Depends(get_session_manager),
):
    """
    压缩流程：
    1. Compress — 压缩旧消息为摘要
    2. Post-Compaction Context — 注入上下文提醒 Agent 重新执行启动序列
    """
    data = session_manager.load_session(session_id, agent_id)
    if data is None:
        raise HTTPException(404, "会话不存在")

    messages = data.get("messages", [])
    if len(messages) < 4:
        raise HTTPException(400, "消息数量不足（至少需要 4 条）")

    try:
        result = await agent_manager.compress_session(session_id, agent_id)
    except Exception as e:
        raise HTTPException(500, f"压缩失败: {e}")

    if "error" in result:
        raise HTTPException(400, result["error"])

    return result
