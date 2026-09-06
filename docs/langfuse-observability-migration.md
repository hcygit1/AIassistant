# Langfuse 观测迁移进度

## 已完成

- 在 LangGraph `astream_events` 中接入 Langfuse callback。
- Langfuse 未配置、初始化失败或上报失败时不阻塞 Agent。
- 每轮 Agent 正常结束或异常结束时调用 `flush()`。
- 统一 Trace 名称为 `pipixia-agent-turn`。
- Trace Metadata 包含 `pipixia_run_id`、`agent_id`、`langfuse_session_id`、`provider` 和 `model`。
- 同一会话的多轮 Trace 可通过 `langfuse_session_id` 在 Sessions 页面聚合。
- `RunTracker` 增加 `max_history`，默认只保留最近 200 条本地短期历史。
- Langfuse 负责长期历史观测，RunTracker 继续保障实时状态和现有前端接口。
- 已验证 Langfuse 凭证、百炼模型切换和 Trace 展示正常。
- Langfuse/RunTracker 相关测试 6 项通过，`git diff --check` 通过。

## 当前保留

- `RunTracker`：活动 Turn、实时 Token、前端运行状态和兼容接口。
- `AuditLogger`：安全、生命周期、心跳、压缩、工具告警和本地审计兜底。
- SSE：前端实时接收回复、Token、工具调用和生命周期事件。

这些模块目前不能直接删除，因为它们不是 Langfuse 的长期历史存储替代品，而是运行时控制、实时通信和故障兜底。

## 待完成

1. 将 Skill 名称、Skill 版本、任务族、Candidate 版本和父版本写入 Trace。
2. 统一业务结果字段：`success`、`failed`、`partial`、`cancelled`，并记录失败类型。
3. 确认 Langfuse 数据完整后，删除 AuditLogger 中重复的普通模型和工具历史记录。
4. 将前端历史 Token、成本和模型分析入口迁移到 Langfuse；实时状态仍走本地 API。
5. 增加 Langfuse Trace 到 Skill 进化离线样本格式的导出流程。
6. 验证网络故障、上报失败、服务重启、多轮 Session 和 Token 差异场景。

## 推荐顺序

```text
补充业务元数据
→ 统一结果和失败类型
→ 验证 Langfuse 数据完整性
→ 收缩 AuditLogger 重复记录
→ 前端历史查询转向 Langfuse
→ 导出 Skill 进化样本
```

## 结论

第一阶段已经完成：Langfuse 可以记录 Agent 的历史运行过程并按会话归并，RunTracker 已收敛为短期缓存。当前不应直接删除 RunTracker、AuditLogger 或 SSE；下一阶段重点是补充业务结果和 Skill 版本信息。
