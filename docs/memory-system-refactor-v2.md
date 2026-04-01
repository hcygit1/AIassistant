# 记忆系统改进方案 v2

基于 v1 重构方案的实施经验，本文档聚焦三个核心改进：**召回策略**、**压缩策略**、**入库时机**。

> v1 方案见 `docs/memory-system-refactor.md`，本文档为增量改进，不重复 v1 中存储层、Skill 层等已稳定的设计。

---

## 一、问题总结

### 1.1 当前召回：每轮自动 recall，浪费且有噪音

```
当前流程（每轮触发）：
  用户发消息
    → _needs_recall() 规则过滤（跳过问候/极短消息）
    → MemRecall.search(用户原始消息)
    → 结果注入 history（作为 assistant 消息）
    → Agent 开始推理
```

问题：

| 问题 | 影响 |
|------|------|
| 多轮追问时每轮都 recall | 重复注入相同记忆，浪费 token |
| 用户原始消息作为 query | "这个怎么弄的"无法有效检索 |
| 连续对话中注入无关记忆 | 噪音干扰 Agent 判断 |
| 每轮 ~100-300ms 向量搜索 | 无论是否需要，都有延迟开销 |

### 1.2 当前压缩：一次性全切，上下文断裂

```
当前流程：
  token 超阈值（80%）
    → _calc_compress_count()：按 keepRecentTokens 从尾部保留
    → messages[:n] 发给 LLM 生成纯文本摘要
    → compressed_context += "---\n" + 新摘要
    → 原始消息移入 archive JSON 文件
    → 注入 post_compaction_context 系统消息
```

问题：

| 问题 | 影响 |
|------|------|
| 80% 才触发，一次压缩大量消息 | 压缩前后上下文断裂严重 |
| 纯文本摘要不断拼接 | `摘要1---摘要2---摘要3` 越来越长，本身成 token 负担 |
| 摘要是追加而非更新 | 无法反映会话目标的演进，只有叠加 |
| 压缩时才入库 | 崩溃/断网 → 未压缩的对话全部丢失 |

### 1.3 当前入库：时机过晚，数据丢失风险高

```
当前入库时机：
  ① /new（session 结束）→ _batch_ingest_messages(session_end=True)
  ② compress_session() → _batch_ingest_messages(被压缩的消息)
```

正常对话过程中不入库。如果用户关掉浏览器、服务崩溃，对话数据不会进入记忆系统。

---

## 二、改进方案

### 2.1 召回策略：清场 + 工具主导

**核心原则：新会话不自动注入任何会话记忆，所有历史记忆由 Agent 按需通过工具获取。**

#### 清场原则

新会话 = 清掉上一段对话的堆积。用户执行 `/new` 就是要干净地开始。

```
固定注入（每轮都有，不受清场影响）：
  System Prompt / SOUL / IDENTITY / USER / AGENTS / TOOLS / Skill 元数据
  → 身份、规则、工具目录、用户画像 — 这些不是「上一段聊天的残留」

不自动注入（清场清掉的）：
  上一轮 session 的结构化摘要
  长期记忆 recall 结果
  任何历史对话片段

Agent 按需获取（工具调用）：
  memory_search(query)   — 语义检索历史记忆
  memory_get(chunk_id)   — 读取完整记忆原文
  memory_timeline(chunk_id, window) — 展开前后对话上下文
```

#### Agent 的 system prompt 中说明：

> 你拥有 memory_search 工具，当你发现当前上下文不足以回答用户问题时，
> 主动搜索历史记忆。不要每次对话都调用，只在确实需要回忆过往事件、
> 决策、配置等具体信息时使用。

#### 与当前实现的对比：

```
当前：
  每轮 → _needs_recall() → MemRecall.search() → 注入 history

改进后：
  所有轮次 → 无自动 recall
  Agent 需要时 → 主动调用 memory_search 工具
```

### 2.2 压缩策略：滑动窗口 + 结构化摘要

#### 压缩后的上下文结构：

```
┌──────────────────────────────────────────────────┐
│  System Prompt + Skills + 固定注入                │  ~固定开销
├──────────────────────────────────────────────────┤
│  结构化会话摘要（从 DB 读取，可更新）              │  ~2000-4000 tokens
│  ┌────────────────────────────────────────────┐  │
│  │ 会话目标：用户在做 XX 项目的部署            │  │
│  │ 关键决策：选择了方案 A，原因是 YY           │  │
│  │ 当前进展：已完成步骤 1-3，正在做步骤 4      │  │
│  │ 待办事项：还需要配置 ZZ                     │  │
│  │ 关键实体：Redis 6.2, Docker 24.0, nginx    │  │
│  └────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────┤
│  保留的近期原始消息（最近 N 轮完整对话）          │  ~8000-12000 tokens
│  → 保证上下文连贯性，Agent 能接着对话            │
└──────────────────────────────────────────────────┘
```

#### 结构化摘要 vs 当前纯文本摘要：

| | 当前（纯文本拼接） | 改进（结构化摘要） |
|---|---|---|
| 格式 | `摘要1\n---\n摘要2\n---\n摘要3` | 结构化 JSON，每次压缩整体更新 |
| 增长 | 线性增长，永远追加 | 固定上限，每次覆盖更新 |
| 信息密度 | 低（重复内容不合并） | 高（决策/进展/待办持续演进） |
| 可读性 | 对 Agent 来说是大段文本 | 结构清晰，Agent 快速定位所需 |

#### 结构化摘要的生成提示词要点：

让 LLM 输出 JSON 格式（用 structured output / JSON mode）：

```json
{
  "goal": "用户在做什么，总体目标",
  "decisions": [
    "选择了方案 A，原因是 XX",
    "放弃了方案 B，因为 YY"
  ],
  "progress": "当前进展到哪一步",
  "open_items": [
    "还需要配置 XX",
    "用户提到稍后要处理 YY"
  ],
  "entities": ["Redis 6.2", "Docker 24.0", "nginx"],
  "user_preferences": [
    "偏好中文回复",
    "喜欢先给方案再解释"
  ]
}
```

每次压缩时，将**上一版摘要 + 本次被压缩的消息**一起交给 LLM，要求生成**更新后的**摘要，而非追加新摘要。

#### 压缩不需要再触发 recall：

结构化摘要已经是当前会话最精准的上下文总结。压缩后无需自动 recall：
- 摘要覆盖了当前会话的核心信息
- 最近 N 轮原始消息保证了对话连续性
- 如果 Agent 发现摘要不够用，它自己会调 `memory_search`

### 2.3 压缩时机：渐进式替代一次性

#### 三级触发机制：

```
token 占用率
  │
  │  0%──────60%──────75%──────90%──────100%
  │  │       │        │        │        │
  │  │  正常  │  预入库  │ 滑动压缩│ 强制压缩│
  │  │       │        │        │        │
```

| 阶段 | token 占用 | 动作 | 说明 |
|------|-----------|------|------|
| 正常 | < 60% | 无 | 每轮异步入库照常进行 |
| 预入库 | 60%-75% | 确认早期消息已入库 | 确保数据安全，不做压缩 |
| 滑动压缩 | 75%-90% | 压缩最早的 30-40% 消息 | 生成/更新结构化摘要，保留最近 N 轮原始消息 |
| 强制压缩 | > 90% | 激进压缩，只保留最近 3-5 轮 | 紧急情况，优先保证系统可用 |

#### 与当前实现的对比：

```
当前：
  token > 80%
    → 一次性压缩到 keepRecentTokens
    → 纯文本摘要追加到 compressed_context

改进后：
  token 到 75%
    → 滑动压缩最早 30-40% 消息
    → 结构化摘要覆盖更新（非追加）
    → 保留最近 N 轮完整对话

  token 到 90%（多次滑动压缩仍不够）
    → 强制压缩到只保留 3-5 轮
    → 结构化摘要依然覆盖更新
```

渐进式的好处：
- 每次只压缩一部分，上下文不断裂
- 结构化摘要持续演进，不丢关键信息
- 多次小压缩 > 一次大压缩

### 2.4 入库时机：每轮异步增量

#### 改进后的入库时机：

| 时机 | 触发条件 | 行为 |
|------|----------|------|
| 每轮对话后 | `save_message` 完成后 | 异步将本轮 user + assistant 消息入队 |
| 压缩时 | 滑动压缩触发 | 确认被压缩消息已入库（hash 去重跳过已入库的） |
| Session 结束 | `/new`、归档 | 确认全部消息已入库 + 标记 `session_end` 触发 Task finalize |

#### 每轮入库的实现要点：

```python
# agent.py astream() 末尾
async for event in agent.astream(...):
    # ... 处理 event ...
    pass

# 本轮结束后，异步入库（非阻塞）
asyncio.create_task(
    self._incremental_ingest(agent_id, session_id, [user_msg, assistant_msg])
)
```

- 异步非阻塞，不影响响应延迟
- hash 去重保证幂等：压缩/结束时的批量入库自动跳过已入库消息
- recall 时按 `session_id` 排除当前 session，不会重复注入

---

## 三、数据库 Schema 变更

### 3.1 新增：session_summaries 表

```sql
CREATE TABLE session_summaries (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    version         INTEGER DEFAULT 1,    -- 每次压缩 version+1
    goal            TEXT,                  -- 会话目标
    decisions       TEXT,                  -- 关键决策（JSON array）
    progress        TEXT,                  -- 当前进展
    open_items      TEXT,                  -- 待办/未解决问题（JSON array）
    entities        TEXT,                  -- 关键实体（JSON array）
    user_preferences TEXT,                 -- 用户偏好（JSON array）
    raw_summary     TEXT,                  -- 兜底：LLM 完整自然语言摘要
    token_count     INTEGER,              -- 摘要注入 context 的 token 数
    created_at      REAL,
    updated_at      REAL,
    UNIQUE(session_id, agent_id)
);
```

每次压缩时 `UPSERT`：`version += 1`，各字段覆盖更新。

### 3.2 已有表无变更

`chunks`、`tasks`、`skills` 及其 FTS/向量表沿用 v1 设计，不修改。

---

## 四、完整流程图

### 4.1 单轮对话的完整生命周期

```
用户发消息
    │
    ├─ 固定注入（System Prompt / SOUL / IDENTITY / USER / AGENTS / TOOLS / Skill 元数据）
    │
    ├─ 是否有结构化摘要？（session_summaries 表）
    │   ├─ 是 → 格式化摘要注入 context 头部
    │   └─ 否 → 跳过（新会话首轮走这里，清场不注入历史记忆）
    │
    ├─ 加载保留的近期原始消息（history）
    │
    ├─ 构建完整 prompt → 发给 LLM
    │
    ├─ Agent 推理（按需调用 memory_search 等工具获取历史记忆）
    │
    ├─ 返回响应给用户
    │
    ├─ save_message（持久化到 session JSON）
    │
    ├─ 异步入库（本轮 user + assistant → MemWorker 队列）
    │   └─ TaskProcessor 判断话题边界（每批入库后触发）
    │
    └─ 检查是否需要压缩（渐进式阈值检测）
        ├─ < 75% → 无操作
        ├─ 75-90% → 滑动压缩
        │   ├─ 确认早期消息已入库
        │   ├─ LLM 生成/更新结构化摘要 → UPSERT session_summaries
        │   ├─ 删除已压缩的原始消息
        │   └─ 保留最近 N 轮原始消息
        └─ > 90% → 强制压缩（只保留 3-5 轮）
```

### 4.2 Session 生命周期

```
Session 创建（清场：只有固定注入，无历史记忆）
    │
    ├─ 对话轮次 1..N（所有轮次行为一致）
    │   ├─ 无自动 recall（Agent 按需调用 memory_search）
    │   ├─ Agent 推理 + 响应
    │   ├─ 异步入库（每轮 2 条 chunk → MemWorker）
    │   └─ TaskProcessor 判断话题边界
    │       ├─ SAME → chunk 挂到当前 active task
    │       └─ NEW → finalize 旧 task + 创建新 task + 触发 Skill 评估
    │
    ├─ [可能] 滑动压缩（75%）
    │   ├─ 结构化摘要 v1 写入 DB
    │   └─ 继续对话...
    │
    ├─ [可能] 再次滑动压缩（又到 75%）
    │   ├─ 结构化摘要 v2 覆盖更新
    │   └─ 继续对话...
    │
    └─ Session 结束（/new）
        ├─ 确认全部消息入库
        ├─ 当前 active task → finalize → 触发 Skill 评估
        └─ session_summary 留存（不删，不参与跨 session 检索）
```

---

## 五、需要修改的文件

### 5.1 agent.py

| 改动点 | 说明 |
|--------|------|
| `astream()` 中的 recall 逻辑 | 移除全部自动 recall 代码（`_needs_recall`、`_format_recall_as_prompt` 等），recall 完全由 Agent 工具调用驱动 |
| `astream()` 末尾新增增量入库 | 每轮结束后异步入库本轮消息 |
| `_maybe_auto_compact()` | 阈值从 80% 单级改为 75%/90% 两级 |
| `compress_session()` | 纯文本摘要改为结构化摘要，UPSERT 到 DB |
| `_calc_compress_count()` | 从按 token 保留改为按轮次保留 + 滑动比例 |
| context 组装 | 读取 `session_summaries` 注入 context 头部 |

### 5.2 session_manager.py

| 改动点 | 说明 |
|--------|------|
| `compress_history()` | `compressed_context` 字段不再拼接纯文本，改为存 summary version 引用 |
| `get_compressed_context()` | 改为从 DB `session_summaries` 表读取结构化摘要 |

### 5.3 token_counter.py

| 改动点 | 说明 |
|--------|------|
| `should_compact()` | 新增返回压缩级别（none / sliding / forced） |
| `resolve_compaction_threshold()` | 返回两级阈值（sliding: 75%, forced: 90%） |

### 5.4 mem/store.py

| 改动点 | 说明 |
|--------|------|
| 新增 `session_summaries` 表 DDL | 建表 + UPSERT 方法 |
| `upsert_session_summary()` | 写入/更新结构化摘要 |
| `get_session_summary()` | 读取当前会话摘要 |

### 5.5 memory_tools.py

| 改动点 | 说明 |
|--------|------|
| `MemSearchTool.description` | 更新描述，强调 Agent 应按需调用 |

### 5.6 prompt_builder.py

| 改动点 | 说明 |
|--------|------|
| `build_post_compaction_context()` | 不再需要提醒 Agent 重新执行启动序列，结构化摘要已自动注入 |
| 新增结构化摘要的格式化方法 | 将 DB 中的 JSON 字段渲染为 Agent 可读的文本 |

---

## 六、配置项变更

```yaml
# config.yaml

agents:
  defaults:
    compaction:
      enabled: true
      slidingThreshold: 0.75       # 滑动压缩触发（新增，替代原 threshold: 0.8）
      forcedThreshold: 0.90        # 强制压缩触发（新增）
      keepRecentTurns: 8           # 滑动压缩保留最近 N 轮完整对话（新增，替代 keepRecentTokens）
      forcedKeepRecentTurns: 4     # 强制压缩保留轮次（新增）
      summaryMaxTokens: 3000       # 结构化摘要注入 context 的 token 上限（新增）

mem:
  recall:
    # 无自动 recall 配置（清场策略：所有 recall 由 Agent 工具调用驱动）
    max_task_results: 5
    chunks_per_task: 2
    budget_chars: 4000
    rrf_k: 60
    recency_half_life_days: 14

  ingest:
    per_turn_enabled: true           # 每轮异步入库开关（新增）
    queue_max_size: 500              # 入库队列最大长度（新增）
    batch_flush_interval_sec: 5      # 队列批量刷写间隔（新增）
```

---

## 七、关键设计决策记录

### Q1: 为什么不在压缩后自动 recall？

结构化摘要已经是当前会话最精准的上下文总结。压缩后再 recall 等于：
- 搜出来的跟摘要重复 → 浪费 token
- 搜出来的是其他 session 的 → 未必相关，引入噪音
- 搜出来的是刚被压缩的 chunk → 白压缩了

Agent 手里有 `memory_search`，如果摘要不够用，它自己会判断并调用。

### Q2: 为什么用结构化摘要替代纯文本拼接？

纯文本拼接的问题是**只增不减**。第 3 次压缩后，`compressed_context` 里有 3 段摘要，大量信息重复（比如"用户在做 XX 项目"在每段摘要里都会出现）。

结构化摘要是**覆盖更新**：每次压缩把上一版摘要 + 新消息交给 LLM，生成一个整体更新的摘要。信息密度恒定，不膨胀。

### Q3: 为什么每轮入库而不是压缩时才入库？

数据安全。当前系统在压缩或 `/new` 时才入库，正常对话过程中的数据只在 session JSON 文件里。如果服务崩溃、用户关浏览器、磁盘满，这些对话不会进入记忆系统。

每轮异步入库 + hash 去重保证：
- 数据实时安全
- 幂等不重复
- 不影响响应延迟

### Q4: 滑动压缩按轮次保留还是按 token 保留？

按轮次。原因：
- 按 token 切可能从一轮对话中间截断（user 留下但 assistant 被切掉）
- 按轮次保证每轮对话完整，Agent 的工作记忆连贯
- 轮次更直觉：`keepRecentTurns: 8` 比 `keepRecentTokens: 8000` 更易理解

token 预算仍然作为兜底：如果 8 轮对话的 token 数超出预算，则减少保留轮数。

### Q5: 为什么新会话不自动注入历史记忆（清场原则）？

用户执行 `/new` 就是要清掉上一段对话的堆积，开始一个干净的新会话。如果系统自动注入历史 recall 结果或上一轮 session 摘要，会与"清场"预期冲突。

区分两类注入：
- **固定注入**（身份/规则/工具/用户画像/Skill 索引）：这些是产品层面的"人设"，不是上一段聊天的残留，每轮都注入
- **会话记忆**（recall 结果/session 摘要/历史片段）：这些是可变的、会话态的，新会话不自动带入

Agent 手里有 `memory_search`，当用户问到需要历史信息的问题时，Agent 自己判断并调用。这比系统替 Agent 猜测"该不该注入"要精准得多。

### Q6: session_summary 与长期记忆（chunks → tasks → skills）的关系？

两者是**并行流**，不是转换关系：

```
每轮消息 ──→ chunks 表（每轮异步入库）──→ tasks（session 结束时 finalize）──→ skills
                                                      ↑
压缩时 ──→ session_summary（当前 session 的工作便签）──┘ Task finalize 时可选读取作为辅助输入
```

- **chunks** 是长期记忆的数据源头，不是 session_summary
- **session_summary** 只服务于当前 session 的上下文续接（压缩后 Agent 能接着对话）
- **Task finalize** 时综合 chunks + session_summary 生成 Task summary，session_summary 作为辅助参考可减少 LLM 重复工作
- **session_summary 不参与跨 session 检索**，session 结束后留存供审计，但不会被 recall 命中

### Q7: 为什么 session_summaries 不与 tasks 表合并？

虽然两者有信息重叠，但合并有四个冲突点：

1. **生命周期不兼容**：Task 是 write-once（finalize 后不再修改），session_summary 是 write-many（每次压缩覆盖更新）
2. **粒度不同**：一个 session 可能有多个 Task（话题切换），但只有一个 session_summary
3. **语义污染**：Task summary 是回顾性的（"做了什么"），session_summary 是状态性的（"进展到哪了"），混在一起会干扰 recall 检索质量
4. **Skill 触发被打破**：SkillEvolver 依赖 Task 是"已完成的任务"来评估是否生成 Skill，混入进行中的摘要会破坏触发逻辑

`session_summaries` 是一张极简表（每个活跃 session 只有 1 行），存储和检索开销几乎为零。

---

## 八、实施步骤

```
Step 1: 入库时机前移
  修改 agent.py → astream() 末尾新增异步入库
  新增 _incremental_ingest() 方法
  hash 去重保证幂等
  验证：正常对话 → 检查 DB 中有实时入库的 chunk

Step 2: 结构化摘要
  新增 session_summaries 表（mem/store.py）
  修改 compress_session() → LLM 输出结构化 JSON
  新增摘要格式化注入方法（prompt_builder.py）
  验证：压缩后 context 中出现结构化摘要

Step 3: 渐进式压缩
  修改 token_counter.py → 两级阈值
  修改 _maybe_auto_compact() → 区分 sliding / forced
  修改 _calc_compress_count() → 按轮次保留
  验证：长对话中观察到多次小压缩而非一次大压缩

Step 4: 召回策略切换（清场 + 工具主导）
  移除 astream() 中全部自动 recall 代码（_needs_recall / _format_recall_as_prompt / 首轮预取分支）
  移除 config 中 auto_recall_on_first_turn / first_turn_max_results / first_turn_budget_chars
  更新 memory_search 工具描述（确保 Agent 知道何时应该调用）
  更新 system prompt 中的记忆工具使用指引
  验证：任何轮次都不触发自动 recall；Agent 按需主动调用 memory_search
```

Step 1-2 可以并行开发，Step 3 依赖 Step 2，Step 4 独立。
