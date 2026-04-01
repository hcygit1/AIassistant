# 记忆系统重构方案

## 一、现有系统分析

### 1.1 架构概述

当前 clawchain 的记忆系统由以下模块组成：

| 模块 | 位置 | 职责 |
|------|------|------|
| `MemorySearchEngine` | `graph/memory_search_engine.py` | SQLite FTS5 全文检索 + 可选 sentence-transformers 向量检索 |
| `MemoryIndexer` | `graph/memory_indexer.py` | 关键词分块索引，RAG prefetch 用 |
| `memory_tools.py` | `tools/memory_tools.py` | 提供 `memory_search`、`memory_get` 工具 |
| 记忆文件 | `workspace/memory/*.md` | Markdown 文件作为长期记忆存储 |
| memory flush | `graph/agent.py` | 压缩前让模型主动写入记忆文件 |
| session 归档 | `api/sessions.py` | `/new` 或关闭时 LLM 摘要写入 `memory/YYYY-MM-DD.md` |

### 1.2 当前写入流程

```
对话结束（手动 /compact 或超阈值）
    ↓
memory flush：模型主动调 file_tools 写入 memory/*.md
    ↓
session 归档（/new）：LLM 摘要追加到日期文件
    ↓
MemoryIndexer 重建关键词索引
```

### 1.3 当前读取流程

```
用户发消息
    ↓
RAG prefetch（MemoryIndexer）：关键词匹配，结果注入 system prompt
    ↓
模型可主动调 memory_search（FTS5 + 可选向量）
    ↓
模型可调 memory_get 读取指定行
```

### 1.4 已识别的问题

| 问题 | 影响 |
|------|------|
| 存储为 Markdown 文件，无结构化语义索引 | 召回精度差，依赖关键词命中 |
| 向量检索为可选项，默认关闭 | 语义召回能力弱 |
| 无任务层（Task）和技能层（Skill）的组织 | 记忆扁平，无法按事件/技能分层检索 |
| 入库时机依赖模型主动写文件 | 信息丢失风险高，写入质量不稳定 |
| 无去重机制 | 重复记忆堆积 |
| 无分层召回 | 数据量增大后只能全量扫描，性能线性劣化 |

---

## 二、新系统设计目标

1. **自动捕获**：对话结束后系统自动入库，不依赖模型主动写文件
2. **分层存储**：Chunk（原文块）→ Task（任务摘要）→ Skill（可复用技能）
3. **混合检索**：FTS5 关键词 + sqlite-vec 向量 ANN，RRF 融合
4. **分层召回**：Task 搜索（历史语义）+ 近期 Chunk 搜索（新鲜内容）
5. **无重复**：入库时精确 hash 去重 + 语义去重；召回时排除当前 session
6. **可扩展**：数据量增大后性能不劣化（ANN 索引 O(log n)）

---

## 三、新系统架构

### 3.1 存储层（单 SQLite 文件）

```
mem.db（路径由配置 mem.storage.db_path 指定，示例 {data_dir}/mem/mem.db）
├── chunks              原始对话块
│   ├── id, session_key, turn_id, seq
│   ├── role, content, summary, kind
│   ├── task_id, skill_id
│   ├── owner, dedup_status
│   └── created_at, updated_at
├── chunks_fts          FTS5 虚拟表（summary + content）
├── vec_chunks          sqlite-vec 向量表（ANN 索引）
│
├── tasks               任务摘要
│   ├── id, session_key, owner
│   ├── title, summary, status（active/completed/skipped）
│   └── started_at, ended_at
├── tasks_fts           FTS5 虚拟表（title + summary）
├── vec_tasks           sqlite-vec 向量表
│
├── skills              技能元数据
│   ├── id, name, description, dirPath
│   ├── version, status, installed, owner, visibility
│   └── quality_score, created_at
├── skills_fts          FTS5 虚拟表（name + description）
└── vec_skills          sqlite-vec 向量表（description 向量）
```

技能内容文件存储在磁盘：

```
stateDir/skills-store/技能名/
    ├── SKILL.md
    ├── scripts/
    ├── references/
    └── evals/

workspace/skills/          （已安装，开局常驻）
    ├── memos-memory-guide/SKILL.md
    ├── skill-creation-guide/SKILL.md
    └── 已安装的技能/SKILL.md
```

### 3.2 模块划分

| 文件 | 职责 |
|------|------|
| `graph/mem_store.py` | SQLite schema 管理、sqlite-vec 初始化、所有 CRUD 方法 |
| `graph/mem_embedder.py` | 向量嵌入（sentence-transformers 或 API provider） |
| `graph/mem_recall.py` | 查询扩展、Task 搜索、近期 Chunk 搜索、RRF、MMR、时间衰减 |
| `graph/mem_worker.py` | 异步入库队列、hash 去重、语义去重、LLM 摘要生成 |
| `graph/mem_task_processor.py` | 任务边界检测、finalize、Task embedding 生成 |
| `graph/mem_skill_evolver.py` | Skill 评估、生成、升级、安装 |
| `tools/memory_tools.py` | 替换现有工具实现 |
| `graph/agent.py` | 接入 before-turn 召回、入库时机钩子 |

---

## 四、写入流程

### 4.1 触发时机

**不在每轮对话结束后立刻入库**，而是：

```
① 压缩触发时（手动 /compact 或超阈值）
   → 被压缩掉的旧消息 → 批量入库

② Session 结束时（/new、归档、子 agent 运行完）
   → 全部未入库消息 → 批量入库
```

设计理由：
- 当前 session 的内容已在对话历史中，不需要从记忆系统重复召回
- 压缩/结束时入库，天然避免召回与会话历史重复的问题
- 子 agent 通过消息传递与主 agent 通信，不依赖记忆实时可见，延迟入库无影响

### 4.2 入库流程

```
待入库消息列表
    ↓
① 精确 hash 去重
   same session + role + content hash → 直接跳过

② LLM 生成 summary（异步，可并行）
   提示词要求：
   - 保留所有数字、配置值、版本号、代码标识符（不得概括为"某个值"）
   - 保留结论性判断（成功/失败、原因）
   - 控制在 120 字以内

③ Embedder 对 summary 做向量嵌入

④ sqlite-vec ANN 语义去重（替代原暴力扫描）
   Top-5 相似 chunk → LLM 判断（输入：新 chunk 的 summary + content 前 300 字，
                                    候选 chunk 的 summary + content 前 300 字）：
   DUPLICATE → 跳过（仅当内容完全相同或无新增信息时）
   UPDATE    → 合并摘要；新 chunk 的完整 content 追加写入，不覆盖旧 content
   NEW       → 正常写入

⑤ INSERT INTO chunks + vec_chunks

⑥ TaskProcessor.on_chunks_ingested()
   │
   ├─ 任务边界检测
   │   ├─ session 变了 → finalize 旧 Task，创建新 Task
   │   ├─ 超 2 小时空闲 → finalize 旧 Task，创建新 Task
   │   └─ LLM 判断话题切换 → finalize 旧 Task，创建新 Task
   │
   └─ finalize Task：
       ├─ LLM 生成结构化摘要（目标/步骤/结论/关键细节）
       ├─ 摘要 embedding → INSERT INTO vec_tasks
       └─ SkillEvolver.on_task_completed()
           ├─ 规则过滤（chunk 数量、摘要长度等）
           ├─ LLM 评估：是否值得提炼为 Skill？
           └─ 值得 → 四步流水线生成 Skill
               ① 读取 skill-creation-guide/SKILL.md 指导
               ② LLM 生成 SKILL.md
               ③ 并行提取脚本 + 参考文档
               ④ 生成测试用例 + 质量评分
               → 写入磁盘 + metadata 存数据库 + description embedding
```

---

## 五、读取流程

### 5.1 查询预处理

```python
def expand_query(query: str) -> list[str]:
    """纯规则拆分，不调 LLM，耗时约 0ms"""
    results = [query]  # 原始 query 保留

    # 按连接词、标点拆分子查询
    parts = re.split(r'[，。？！,?!]|在.+里|和|以及|跟|与', query)
    parts = [p.strip() for p in parts if len(p.strip()) > 2]

    if len(parts) > 1:
        results.extend(parts)

    return results[:4]  # 最多 4 个，控制 embedding 调用次数
```

**FTS 不做扩展**（天然多关键词并行匹配），**向量搜索做扩展**（一个向量是加权混合，多方向查询需要多个子向量）。

### 5.2 完整召回流程

```
用户发消息（before_agent_start 钩子）
    │
    ▼
expand_query() → [原始 query, 子查询1, 子查询2, ...]
    │
    ┌───────────── 并行执行 ─────────────┐
    │                                   │
    ▼                                   ▼
Task 搜索                         近期 Chunk 搜索
（历史语义层）                      （新鲜内容层）
    │                                   │
    ├─ FTS：tasks_fts                    ├─ FTS：chunks_fts
    │   原始 query                       │   原始 query
    │   对象：status = 'completed'        │   过滤：非 completed Task
    │                                   │   排除：当前 session_id
    ├─ 向量：vec_tasks（ANN）             │
    │   多个子查询并行搜索                 ├─ 向量：vec_chunks（ANN）
    │   结果取最高分合并                   │   多个子查询并行搜索
    │                                   │   结果取最高分合并
    ├─ RRF 融合                          │
    ├─ Top-5 Task                        ├─ RRF 融合
    │                                   ├─ MMR 去重
    └─ 每个 Task：                        └─ Top-N 近期 Chunk
        摘要 + Task 内 Top-2 Chunk
        （在 task_id 范围内子查询）
    │                                   │
    └───────────── 合并 ────────────────┘
                    │
                    ▼
            预算分配（默认 4000 字符）

            判断模式：
            if Task 命中 >= 2 且近期命中 < 3:
                task_heavy：Task × 3 + 近期 Chunk × 2
            elif 近期命中 >= 3 且 Task 命中 < 2:
                recent_heavy：Task × 1 + 近期 Chunk × 6
            elif 两边都无命中:
                全局兜底：全量 chunks 搜索 Top-8
            else:
                均衡：Task × 2 + 近期 Chunk × 4
                    │
                    ▼
            注入 system prompt
```

### 5.3 注入格式

```
## 历史任务记忆

### 任务：配置 Redis 缓存（2026-01-15）
目标：将 session 存储迁移到 Redis
关键步骤：pip install redis → 配置连接池 → 修改 settings.py
结论：成功上线，响应时间降低 40%
关键细节：max_connections=50, socket_timeout=5, decode_responses=True

相关片段：
  1. [user] Redis 连接池应该设多少... (score: 0.94)
     chunkId="abc123" taskId="task-001"
  2. [assistant] max_connections=50 是个合理起点... (score: 0.89)
     chunkId="def456" taskId="task-001"

---

### 任务：Docker 部署配置（2026-01-20）
...

## 近期对话（其他 session）

  3. [user] 我现在想把连接池改成 100... (score: 0.91)
     chunkId="ghi789"
  4. [assistant] 好的，需要同时修改... (score: 0.87)
     chunkId="jkl012"

如需更多上下文：
→ memory_timeline(chunkId) 展开前后对话
→ memory_get(chunkId) 读取完整原文
→ memory_search(query) 换角度搜索
→ skill_search(query) 搜索相关技能指南
```

### 5.4 模型后续工具调用（ReAct 循环）

```
模型判断 prompt 里的信息是否足够
    ├─ 足够 → 直接回答
    ├─ 需要上下文（因果顺序/前后消息）
    │   → memory_timeline(chunkId, window=3)
    ├─ 需要完整原文
    │   → memory_get(chunkId)
    ├─ 需要另一方向的记忆（多跳）
    │   → memory_search(query="新角度关键词")
    ├─ 需要操作指南
    │   → skill_search(query) → skill_get(skillId)
    └─ 信息确实不存在 → 告知用户
```

---

## 六、Skill 层设计

### 6.1 两类 Skill

**通用 Skill（开局常驻）**

系统启动时写入 `workspace/skills/`，OpenClaw/agent 系统自动扫描加载，每轮 name + description 在 system prompt 中：

| Skill | 职责 |
|-------|------|
| `memos-memory-guide` | 记忆工具使用说明，模型使用记忆系统的行为规范 |
| `skill-creation-guide` | Skill 生成的知识指南，指导 LLM 生成高质量 SKILL.md |

**Task 生成的 Skill（按需加载）**

- 存于 `stateDir/skills-store/技能名/`
- 数据库只存 metadata + description embedding
- 模型通过 `skill_search` 发现，`skill_get` 获取完整内容
- `skill_install` 后复制到 `workspace/skills/`，升级为常驻

### 6.2 Skill 生成流水线

```
Task 完成
    → SkillEvolver.on_task_completed(task)
    │
    ├─ 规则过滤（硬性门槛）
    │   chunks >= 6
    │   summary 长度 >= 100
    │   有用户消息 + 有助手回复
    │   task.status != 'skipped'
    │
    ├─ LLM 评估（软性判断）
    │   输入：task.title + task.summary
    │   判断：可复用性、可迁移性、技术深度
    │   输出：shouldGenerate + suggestedName + confidence
    │
    └─ shouldGenerate = true → 四步流水线
        │
        ├─ Step 1：读取 skill-creation-guide/SKILL.md 作为指导
        │          LLM 生成 SKILL.md（含 frontmatter + 各章节）
        │
        ├─ Step 2：并行提取
        │   ├─ 脚本（scripts/*.sh 等）
        │   └─ 参考文档（references/*.md）
        │
        ├─ Step 3：生成测试用例（evals/evals.json）
        │          3-4 个真实触发场景
        │
        └─ Step 4：验证 + 质量评分（0-10）
                   < 6 分标记为 draft
                   写入数据库 + description embedding

Skill 升级（相似 Task 再次出现时）：
    → 搜索相似 Skill（vec_skills ANN）
    → LLM 判断：值得升级？升级类型？
    → SkillUpgrader 生成新版本 SKILL.md
    → 追加 skill_versions 记录
    → 若已安装，自动同步到 workspace/skills/
```

### 6.3 skill-creation-guide 的自我进化

`skill-creation-guide/SKILL.md` 本身也是一个 Skill，受 `SkillEvolver` 管理。当系统观察到"生成的 Skill 质量评分普遍偏低"或"用户频繁修改生成的 Skill"时，可触发对该指南本身的升级，实现方法论层面的自我进化。

---

## 七、性能分析

### 7.1 写入性能

| 操作 | 当前（暴力扫描）| 新系统（sqlite-vec ANN）|
|------|---------------|----------------------|
| 语义去重（5000 条）| ~50ms/条 | ~5ms/条 |
| 语义去重（50000 条）| ~500ms/条 | ~5ms/条 |
| Task 摘要 embedding | 无 | ~30ms（一次性）|

### 7.2 读取性能

| 操作 | 当前（FTS 全量）| 新系统 |
|------|--------------|-------|
| Task 搜索 | 无此功能 | FTS ~1ms + ANN ~5ms |
| 近期 Chunk 向量搜索（5000 条）| ~50ms | ANN ~5ms |
| 近期 Chunk 向量搜索（50000 条）| ~500ms | ANN ~5ms |
| 查询扩展 | 无 | 纯规则 ~0ms |
| 并行 embedding（4 个子查询）| 无 | ~30ms（并行）|
| **自动召回总耗时** | **~100ms** | **~50ms** |

---

## 八、与现有系统的对接

### 8.1 需要修改的文件

| 文件 | 改动 |
|------|------|
| `graph/agent.py` | ① `initialize()`：初始化 MemStore、MemRecall、MemWorker；② `astream()` 开头：before-turn 召回替换 RAG prefetch；③ `_maybe_auto_compact()` 和 `save_session_memory()`：触发入库；④ **移除 memory flush**（不再调用 `run_memory_flush` 等） |
| `tools/memory_tools.py` | 全部替换，接入 MemRecall 和 MemStore |
| `graph/prompt_builder.py` | 记忆注入格式改为新的分层格式；移除 memory flush 相关 prompt 与写 markdown 的指引 |
| `config.py` / `config_schema.py` | 新增 `mem` 配置节（db_path、embedding、recall 参数等）|

### 8.2 可以移除的文件

| 文件 | 原因 |
|------|------|
| `graph/memory_search_engine.py` | 由 `mem_recall.py` + `mem_store.py` 替代 |
| `graph/memory_indexer.py` | RAG prefetch 由新召回流程替代 |

### 8.3 不需要改动的文件

- `graph/session_manager.py`：对话历史管理不变
- `graph/token_counter.py`：压缩阈值判断不变
- `graph/compaction.py`：压缩逻辑不变，只在压缩后触发入库
- `graph/subagent_registry.py`：子 agent 管理不变
- `tools/file_tools.py`：文件操作不变

### 8.4 入库钩子接入点

```python
# graph/agent.py

# 钩子 1：压缩时入库
async def _maybe_auto_compact(self, session_id, agent_id):
    # ... 原有压缩逻辑 ...
    compressed_messages = session_manager.compress_history(...)

    # 新增：被压缩的消息入库
    self.mem_worker.enqueue(
        messages=compressed_messages,
        session_id=session_id,
        agent_id=agent_id
    )

# 钩子 2：session 结束时入库
async def save_session_memory(self, session_id, agent_id):
    messages = session_manager.get_history(session_id)

    # 新增：全部未入库消息入库
    self.mem_worker.enqueue(
        messages=messages,
        session_id=session_id,
        agent_id=agent_id,
        session_end=True  # 标记 session 结束，触发 Task finalize
    )
```

---

## 九、实施规划（阶段一至五）

### 9.1 前提决定

| 项 | 决定 |
|----|------|
| 旧记忆迁移 | **不做**：不将现有 `workspace/memory/*.md` 导入新库 |
| memory flush | **不保留**：移除 `run_memory_flush` 及压缩前让模型写记忆文件的流程 |
| 向量检索 | **sqlite-vec ANN**：Python 侧 `pip install sqlite-vec`，连接 SQLite 后 `sqlite_vec.load(conn)`，向量表使用 `vec0` 虚拟表；入库与召回均走 ANN，**不**采用全量暴力余弦扫描 |
| 模块前缀 | 统一使用 `mem_`（如 `mem_store.py`），**不**使用 `memos_` 前缀 |

### 9.2 模块命名对照

| 文档/旧称 | 实际文件名 |
|-----------|------------|
| memos_store | `graph/mem_store.py` |
| memos_embedder | `graph/mem_embedder.py` |
| memos_worker | `graph/mem_worker.py` |
| memos_recall | `graph/mem_recall.py` |
| memos_task_processor | `graph/mem_task_processor.py` |
| memos_skill_evolver | `graph/mem_skill_evolver.py` |

---

### 阶段一：存储层

**新建：** `graph/mem_store.py`、`graph/mem_embedder.py`  
**依赖：** `sqlite-vec`（PyPI）

**`MemStore`（`mem_store.py`）**

- 单 SQLite 文件；连接时 `enable_load_extension(True)` → `sqlite_vec.load(conn)` → `enable_load_extension(False)`。
- Schema 与本文 **§3.1** 一致：`chunks`、`chunks_fts`（FTS5）、`vec_chunks`（sqlite-vec）、`tasks`、`tasks_fts`、`vec_tasks`、`skills`、`skills_fts`、`vec_skills`。
- `vec_*` 表使用 sqlite-vec `vec0` 虚拟表；向量维度与 embedding 配置一致（如 1536）；写入时使用 `serialize_float32` 或等价 BLOB 格式。
- 关键方法：`insert_chunk`、`upsert_embedding`（chunk / task / skill 对应各 vec 表）、`fts_search`、`ann_search`（`MATCH` + `ORDER BY distance LIMIT k`）、`find_active_chunk_by_hash`、`get_chunk`、`get_chunks_in_range`（供 `memory_timeline`）、Task/Skill CRUD、`mark_dedup_status` 等。

**`MemEmbedder`（`mem_embedder.py`）**

- 统一接口：`embed(texts: list[str])`、`embed_query(text: str)` → `list[float]`。
- Provider：`openai` / `openai_compatible` / `local`（本地 sentence-transformers 可与现有 `memory_search_engine` 逻辑对齐后迁移）。
- 维度与 `mem.embedding.dimensions` 一致。

**验证：** 建表成功；FTS 命中；ANN 查询返回合理排序；无旧数据迁移步骤。

---

### 阶段二：处理层

**新建：** `graph/mem_worker.py`、`graph/mem_recall.py`

**`MemWorker`（`mem_worker.py`）**

- 异步入库队列；按本文 **§4.2** 流程：hash 去重 → LLM summary（§4.2 提示词约束）→ 对 summary 嵌入 → **sqlite-vec ANN** 取 Top-5 候选 → LLM judgeDedup（输入含新/旧 **summary + content 前 300 字**）→ DUPLICATE / UPDATE / NEW → `insert_chunk` + 写 `vec_chunks`。
- `session_end=True` 时通知 TaskProcessor（阶段四接线）。
- LLM 调用复用后端 `llm_factory` 等现有能力；prompt 可参考 `memos-local-openclaw` 中 summarizer / dedup 实现并翻译为 Python。

**`MemRecall`（`mem_recall.py`）**

- 本文 **§5.1** `expand_query`（纯规则）。
- **阶段二可先实现单层 Chunk 搜索**：`chunks_fts` + `vec_chunks` ANN → RRF（`rrf_k`）→ MMR（`mmr_lambda`）→ 时间衰减（`recency_half_life_days`）→ `budget_chars` 截断；排除当前 `session_id`。
- Task 分层搜索与 **§5.2** 完整双路并行在 **阶段四** 与 `MemTaskProcessor` 一并接通。

**验证：** 入库若干条后 `search` 能召回；当前 session 被排除；去重走 ANN + LLM，而非全表余弦。

---

### 阶段三：接入层

**修改：** `graph/agent.py`、`tools/memory_tools.py`、`graph/prompt_builder.py`、`config.py` / `config_schema.py`  
**删除：** `graph/memory_search_engine.py`、`graph/memory_indexer.py`

**`agent.py`**

- 初始化：`MemStore`、`MemEmbedder`、`MemWorker`、`MemRecall`（按 agent 维度放入 dict）。
- before-turn：用 `MemRecall.search` 替换原 `MemoryIndexer.retrieve` + RAG 注入；格式见 **§5.3**。
- 钩子：压缩后、session 结束时调用 `mem_worker.enqueue`（见 **§8.4**）。
- **移除** memory flush 全流程及 `prompt_builder` 中 flush 相关构建。

**`memory_tools.py`**

- `memory_search` → `MemRecall.search`
- `memory_get` → `MemStore.get_chunk`
- `memory_timeline` → `MemStore.get_chunks_in_range`（或等价 API）

**`prompt_builder.py`**

- 新增分层记忆注入；移除指导写 `MEMORY.md` / 日期文件的旧指引（与 **§9.1** 一致）。

**验证：** 端到端对话 → 压缩或结束 session → 入库 → 新 session 自动召回 + 工具可用。

**里程碑：** 阶段一至三完成后，基础记忆能力已替换旧实现，可独立上线。

---

### 阶段四：任务层

**新建：** `graph/mem_task_processor.py`

**`MemTaskProcessor`**

- 在 `MemWorker` 入库路径末尾调用 `on_chunks_ingested`。
- 任务边界：session 变化、空闲超时（`task.idle_timeout_hours`）、LLM 话题切换（见 **§4.2** ⑥）。
- finalize：LLM 结构化 Task 摘要 → 嵌入 → `tasks` + `tasks_fts` + `vec_tasks`。

**`MemRecall` 升级**

- 接通 **§5.2**：Task 搜索（`tasks_fts` + `vec_tasks`）与近期 Chunk 搜索并行；预算分配与 **§5.2** 模式（task_heavy / recent_heavy / 兜底 / 均衡）一致。

**验证：** 多 session 后 Task 表有记录；召回含「历史任务记忆」+「近期对话」结构。

---

### 阶段五：技能层

**新建：** `graph/mem_skill_evolver.py`（内部可拆 Evaluator / Generator / Upgrader / Validator / Installer，逻辑见 **§6**）

**接线**

- `MemTaskProcessor` 在 Task finalize 后调用 `MemSkillEvolver.on_task_completed`（规则过滤 → LLM 评估 → 四步流水线 → 磁盘 + DB + `vec_skills`）。
- 按需新增 `skill_search` / `skill_get` / `skill_install` 等工具（与 **§5.3、§5.4、§6** 一致）。

**验证：** Task 完成后可生成 Skill；相似 Task 可走升级路径；可选 `skill-creation-guide` 资源。

**里程碑：** 阶段五完成后，Chunk → Task → Skill 三层闭环可用。

---

### 9.3 依赖关系小结

```
阶段一  mem_store.py + mem_embedder.py
  ▼
阶段二  mem_worker.py + mem_recall.py（Chunk 单层召回）
  ▼
阶段三  agent.py + memory_tools.py + prompt_builder.py + config
        删除 memory_search_engine.py、memory_indexer.py
        ✓ 基础记忆替换完成，可上线
  ▼
阶段四  mem_task_processor.py + mem_recall 分层召回
        ✓ 分层召回生效
  ▼
阶段五  mem_skill_evolver.py + 可选 skill 工具
        ✓ 三层记忆完整
```

前三个阶段完成后系统即可正常运行；第四、五阶段为增强能力，可分期合并。

---

## 十、配置项设计

```yaml
# config.yaml 新增 mem 节
mem:
  storage:
    db_path: "{data_dir}/mem/mem.db"

  embedding:
    provider: "openai"          # openai / openai_compatible / local
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimensions: 1536

  recall:
    max_task_results: 5         # Task 搜索返回最大数量
    chunks_per_task: 2          # 每个 Task 附带的 Chunk 数
    max_recent_chunks: 8        # 近期 Chunk 最大数量
    budget_chars: 4000          # 注入 prompt 的字符预算
    min_task_score: 0.3         # Task 搜索最低分阈值
    rrf_k: 60                   # RRF 融合参数
    mmr_lambda: 0.7             # MMR 多样性参数
    recency_half_life_days: 14  # 时间衰减半衰期

  dedup:
    similarity_threshold: 0.60  # 语义去重相似度阈值

  task:
    idle_timeout_hours: 2       # 空闲超时触发新任务的时间

  skill_evolution:
    enabled: true
    auto_evaluate: true
    min_chunks_for_eval: 6
    min_confidence: 0.7
    auto_install: false
```
