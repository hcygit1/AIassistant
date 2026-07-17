# PIPIXIA（中文简洁版）

PIPIXIA 是一个本地优先的 AI Agent 系统，基于 Python + LangChain/LangGraph 构建。

---

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.11+ · FastAPI · LangChain / LangGraph |
| 前端 | Next.js · React · TypeScript |
| 存储 | SQLite（FTS5 + sqlite-vec）· 本地文件系统 |

---

## 快速开始

```bash
# 一键启动
python3 scripts/dev.py

# 或单独启动
cd backend && python3 -m pip install -r requirements.txt && python3 cli.py start
cd frontend && npm install && npm run dev
```

浏览器打开：<http://localhost:3000>

质量检查：后端运行 `python3 -m pytest backend/tests`；前端依次运行 `npm test`、`npx tsc --noEmit`、`npm run lint` 和 `npm run build`。

---

## 核心功能

### 记忆系统
- 写入-索引-召回三阶段架构
- SQLite FTS5 全文索引 + sqlite-vec 向量索引双路检索
- 异步摘要 + SHA-256 去重，小模型执行保证低成本
- 瀑布式召回，40,000 字符预算内按优先级检索
- 技能演化：从历史对话中自动提炼可复用技能

### 上下文管理
- `ContextBudget` frozen dataclass 统一管理所有预算参数
- 三级压缩：JIT 裁剪 → 滑动摘要（80%）→ 强制压缩（95%）
- 换模型只改一个配置值，所有比率自动等比缩放

### 子 Agent 协作
- 独立会话和工具集，支持嵌套 spawn
- 显式状态机管理运行和投递生命周期
- 标准化事件总线（23 种事件类型，`Events` 工厂统一构建）

### 工具与安全
- 文件、命令、网络、记忆、会话类工具
- 路径与执行策略约束，危险操作审批流
- 工具结果落盘与 JIT 截断

### 其他
- Heartbeat 后台巡检与 Cron 定时任务
- 会话命令系统（`/new`、`/compact`、`/status`）
- SSE 事件流 + REST API 可观测

## License

MIT
