# PIPIXIA Frontend

基于 Next.js + React + TypeScript 的 PIPIXIA Web 客户端。

## 技术栈

- **框架**: Next.js (App Router)
- **UI**: React + Tailwind CSS
- **状态管理**: React Context + Hooks
- **Markdown 渲染**: react-markdown + remark-gfm
- **代码编辑**: @monaco-editor/react
- **图标**: Lucide React

## 开发运行

### 前置条件

- Node.js 20+
- 后端服务已启动（默认 `http://localhost:8002`）

### 启动

```bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建
npm run build

# 生产预览
npm run start

# ESLint 检查
npm run lint

# TypeScript 检查
npx tsc --noEmit

# Playwright 测试
npm test
```

## 目录结构

```
frontend/
├── src/
│   ├── app/              # Next.js App Router 入口
│   │   ├── page.tsx      # 主页面
│   │   ├── layout.tsx    # 根布局
│   │   └── globals.css   # 全局样式
│   ├── components/       # React 组件
│   │   ├── chat/         # 聊天相关组件
│   │   ├── inspector/    # Inspector 面板组件
│   │   ├── layout/       # 布局组件
│   │   └── editor/       # 编辑器组件
│   ├── lib/              # 工具库
│   │   ├── api.ts        # API 客户端
│   │   ├── store.tsx     # App Context 组合层
│   │   ├── chatState.ts  # 按 Agent 隔离的聊天状态与运行时
│   │   ├── chatStreamEvents.ts # SSE 消息归约
│   │   ├── chatTurnRecovery.ts # 未完成 turn 恢复
│   │   ├── hooks/        # 自定义 Hooks
│   │   │   ├── useChat.ts
│   │   │   ├── useSubagents.ts
│   │   │   ├── useInspectorState.ts
│   │   │   └── useAppUiState.ts
│   │   └── i18n/         # 国际化
├── tests/                # Playwright 单元与结构回归测试
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── postcss.config.mjs
```

## 核心组件

### ChatPanel
聊天主面板，处理消息展示、输入、流式输出

### InspectorPanel
侧边检查面板，支持文件浏览、技能管理、子 Agent 状态、审计日志等

### Navbar
顶部导航栏，包含 Agent 切换、主题切换、菜单等

### ConfigModal
配置中心弹窗，支持 Provider/模型配置

## 状态管理

`AppProvider` 仅负责组合并对外提供 Context；具体职责由独立模块承担：
- `useChat` 与 `chatState`：按 Agent 隔离消息、活跃 turn 和本地队列
- `useSubagents`：子 Agent 树、运行状态、实时轨迹和唯一轮询/SSE 状态源
- `useInspectorState`：Inspector 布局、文件读取与保存
- `useAppUiState`：主题、国际化、弹窗和全局通知

## API 通信

所有后端请求封装在 `src/lib/api.ts`：
- 聊天：`submitChat` → `waitUntilTurnRunning`（若 turn 已结束则直接拉消息）→ `streamTurn`；刷新后 `fetchPendingTurn` 自动续接未完成的 turn
- 中断对话 (`abortChat`，可带 `turnId`)
- Agent/Session 管理
- 配置读写
- 文件读写
- 子 Agent 状态订阅

## 国际化

支持 `zh-CN` 和 `en-US`，配置文件位于 `src/lib/i18n/locales.ts`。
用户偏好存储在 `localStorage`。

## 主题

支持 `system` / `light` / `dark` 三种模式，使用 CSS 变量实现。

## 许可证

MIT
