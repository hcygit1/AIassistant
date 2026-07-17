import type { SSEEvent, SubagentTreeItem } from "./api";

export interface LiveTraceEntry {
  ts: number;
  type: string;
  text: string;
}

export interface SubagentState {
  scopeKey: string;
  tree: SubagentTreeItem[];
  flat: SubagentTreeItem[];
  traceMap: Record<string, LiveTraceEntry[]>;
  loading: boolean;
}

export interface RunningSubagent {
  run_id: string;
  task: string;
  status: string;
}

export interface MappedSubagentEvent {
  runId: string;
  trace: LiveTraceEntry;
  shouldRefresh: boolean;
  doneDelayMs: number | null;
}

export type SubagentStateAction =
  | { type: "scope"; scopeKey: string }
  | { type: "loading"; scopeKey: string }
  | { type: "success"; scopeKey: string; tree: SubagentTreeItem[]; flat: SubagentTreeItem[] }
  | { type: "failure"; scopeKey: string }
  | { type: "trace"; scopeKey: string; runId: string; trace: LiveTraceEntry };

export function deriveSubagentViews(flat: SubagentTreeItem[], runId?: string) {
  const runningSubagents: RunningSubagent[] = flat
    .filter((item) => (item.state || item.status) === "running")
    .map((item) => ({
      run_id: item.run_id,
      task: item.task?.slice(0, 60) || "",
      status: item.state || item.status,
    }));

  return {
    runningSubagents,
    inline: runId ? flat.find((item) => item.run_id === runId) ?? null : null,
  };
}

function announceLabel(state: string): string {
  switch (state) {
    case "queued":
      return "等待主会话队列";
    case "delivering":
      return "主Agent正在融合结果";
    case "delivered":
      return "已合并进主会话";
    case "dropped":
      return "结果合并失败，已写入兜底消息";
    case "retrying":
      return "结果通知重试中";
    default:
      return state || "结果通知状态更新";
  }
}

export function mapSubagentEvent(event: SSEEvent, timestamp: number): MappedSubagentEvent | null {
  if (!event.type.startsWith("subagent_")) return null;
  const runId = String(event.run_id || "").trim();
  if (!runId) return null;

  let type = "";
  let text = "";
  let shouldRefresh = true;
  let doneDelayMs: number | null = null;

  switch (event.type) {
    case "subagent_start": {
      const task = String(event.task || "").slice(0, 200);
      type = "start";
      text = task ? `开始执行：${task}` : "开始执行";
      break;
    }
    case "subagent_tool": {
      const tool = String(event.tool || "").trim();
      type = "tool";
      text = tool ? `调用工具：${tool}` : "调用工具";
      break;
    }
    case "subagent_tool_end": {
      const tool = String(event.tool || "").trim();
      const preview = String(event.output_preview || "").trim();
      type = "tool_end";
      text = preview ? `工具完成：${tool} -> ${preview}` : `工具完成：${tool}`;
      break;
    }
    case "subagent_progress":
      type = "progress";
      text = `执行中：${Number(event.elapsed_s || 0)}s，输出 ${Number(event.chars || 0)} chars`;
      shouldRefresh = false;
      break;
    case "subagent_done": {
      const result = String(event.result || "").trim();
      type = "done";
      text = result ? `执行完成，等待结果合并：${result.slice(0, 160)}` : "执行完成，等待结果合并";
      break;
    }
    case "subagent_error": {
      const error = String(event.error || "").trim();
      type = "error";
      text = error ? `执行失败：${error.slice(0, 160)}` : "执行失败";
      break;
    }
    case "subagent_killed":
      type = "killed";
      text = "已终止";
      break;
    case "subagent_announce": {
      const state = String(event.result_delivery_state || "").trim();
      type = "announce";
      text = announceLabel(state);
      if (state === "delivered") doneDelayMs = 250;
      if (state === "dropped") doneDelayMs = 100;
      break;
    }
    default:
      return null;
  }

  return {
    runId,
    trace: { ts: timestamp, type, text },
    shouldRefresh,
    doneDelayMs,
  };
}

export function createSubagentState(scopeKey: string): SubagentState {
  return { scopeKey, tree: [], flat: [], traceMap: {}, loading: false };
}

export function subagentStateReducer(state: SubagentState, action: SubagentStateAction): SubagentState {
  if (action.type === "scope") {
    return action.scopeKey === state.scopeKey ? state : createSubagentState(action.scopeKey);
  }
  if (action.type === "loading") {
    if (action.scopeKey !== state.scopeKey) {
      return { ...createSubagentState(action.scopeKey), loading: true };
    }
    return state.loading ? state : { ...state, loading: true };
  }

  if (action.scopeKey !== state.scopeKey) return state;

  if (action.type === "success") {
    return { ...state, tree: action.tree, flat: action.flat, loading: false };
  }
  if (action.type === "failure") {
    return { ...state, tree: [], flat: [], loading: false };
  }

  const previous = state.traceMap[action.runId] || [];
  return {
    ...state,
    traceMap: {
      ...state.traceMap,
      [action.runId]: [...previous, action.trace].slice(-50),
    },
  };
}
