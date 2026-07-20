"use client";

import { useState } from "react";
import { useApp } from "@/lib/store";
import { useSubagentContext } from "@/lib/subagentContext";
import { useUi } from "@/lib/uiContext";
import * as api from "@/lib/api";
import type { SubagentTreeItem } from "@/lib/api";
import type { LiveTraceEntry } from "@/lib/subagentState";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, ChevronRight, CheckCircle, XCircle, Loader2, Wrench, Square, Send } from "lucide-react";

const STEER_TEMPLATES = [
  "请先总结当前已完成的步骤，再继续执行下一步。",
  "请先验证上一轮工具结果是否可靠，若不可靠请换数据源重试。",
  "请用三条要点给出当前结论，并标注不确定性。",
];

interface SubagentMessage {
  role: string;
  content: string;
  tool_calls?: { tool: string; input: any; output: string }[];
}

export default function SubagentPanel() {
  const {
    currentAgentId,
    loadMainSession,
  } = useApp();
  const { tree, traceMap, loading, refreshSubagents } = useSubagentContext();
  const { showNotice, t } = useUi();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [killAllBusy, setKillAllBusy] = useState(false);
  const [lastSteerPrompt, setLastSteerPrompt] = useState("");

  const toggleExpand = (runId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  if (loading && tree.length === 0) {
    return <div className="text-center text-xs py-8" style={{ color: "var(--text-tertiary)" }}>{t.loading}</div>;
  }

  if (tree.length === 0) {
    return <div className="text-center text-xs py-8" style={{ color: "var(--text-tertiary)" }}>{t.noSubagents}</div>;
  }

  const runningCount = countRunning(tree);

  return (
    <div className="space-y-1.5">
      {runningCount > 0 && (
        <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg text-[10px] glass-card">
          <span style={{ color: "var(--text-secondary)" }}>running: {runningCount}</span>
          <button
            type="button"
            disabled={killAllBusy}
            onClick={async () => {
              const confirmed = window.confirm(t.confirmKillSubagents.replace("{count}", String(runningCount)));
              if (!confirmed) return;
              setKillAllBusy(true);
              try {
                const res = await api.killSubagent(currentAgentId, "all");
                if (res?.ok) {
                  showNotice({ kind: "success", text: t.killedSubagents.replace("{count}", String(res.killed ?? 0)) });
                } else {
                  showNotice({ kind: "error", text: res?.error || "终止失败" });
                }
                await refreshSubagents();
                await loadMainSession();
              } finally {
                setKillAllBusy(false);
              }
            }}
            className="btn-outline" style={{ padding: "2px 8px", fontSize: 10 }}
          >
            <Square className="w-2.5 h-2.5" />
            Kill all
          </button>
        </div>
      )}
      {tree.map((sa) => (
        <SubagentTreeNode
          key={sa.run_id}
          node={sa}
          depth={0}
          expanded={expanded}
          toggleExpand={toggleExpand}
          onNotice={showNotice}
          onRefresh={refreshSubagents}
          traceMap={traceMap}
          lastSteerPrompt={lastSteerPrompt}
          setLastSteerPrompt={setLastSteerPrompt}
        />
      ))}
    </div>
  );
}

function countRunning(nodes: SubagentTreeItem[]): number {
  let count = 0;
  for (const n of nodes) {
    if ((n.state || n.status) === "running") count += 1;
    if (n.children?.length) count += countRunning(n.children);
  }
  return count;
}

function SubagentTreeNode({
  node, depth, expanded, toggleExpand, onNotice, onRefresh, traceMap, lastSteerPrompt, setLastSteerPrompt,
}: {
  node: SubagentTreeItem;
  depth: number;
  expanded: Set<string>;
  toggleExpand: (runId: string) => void;
  onNotice: (notice: { kind: "success" | "error" | "info"; text: string }) => void;
  onRefresh: () => Promise<void>;
  traceMap: Record<string, LiveTraceEntry[]>;
  lastSteerPrompt: string;
  setLastSteerPrompt: (v: string) => void;
}) {
  const { currentAgentId, loadMainSession } = useApp();
  const { t } = useUi();
  const status = node.state || node.status;
  const isRunning = status === "running";
  const isExpanded = expanded.has(node.run_id);
  const hasChildren = node.children && node.children.length > 0;
  const traces = traceMap[node.run_id] || [];
  const [busy, setBusy] = useState(false);
  const [showSteerInput, setShowSteerInput] = useState(false);
  const [steerText, setSteerText] = useState("");

  return (
    <div className="space-y-1" style={{ marginLeft: depth > 0 ? `${depth * 12}px` : 0 }}>
      <div className="glass-card overflow-hidden">
        <div
          role="button"
          tabIndex={0}
          onClick={() => toggleExpand(node.run_id)}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleExpand(node.run_id); } }}
          className="w-full flex items-center gap-2 px-2.5 py-2 text-left transition-colors cursor-pointer"
          onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
        >
          <ChevronRight className={`w-3 h-3 flex-shrink-0 transition-transform duration-200 ${isExpanded ? "rotate-90" : ""}`}
            style={{ color: "var(--text-secondary)" }} />
          <StatusIcon status={status} />
          {(depth > 0 || (node.spawn_depth != null && node.spawn_depth >= 1)) && (
            <span className="text-[9px] flex-shrink-0" style={{ color: "var(--text-tertiary)" }}>
              D{node.spawn_depth ?? depth + 1}
            </span>
          )}
          <div className="flex-1 min-w-0 overflow-hidden">
            <div className="text-xs font-medium text-[var(--text)] truncate">{node.label}</div>
            <div className="text-[10px] truncate" style={{ color: "var(--text-secondary)" }}>{node.task}</div>
            {!isRunning && (
              <div className="text-[10px] truncate" style={{ color: "var(--text-tertiary)" }}>
                {formatTerminalMeta(node)}
              </div>
            )}
          </div>
          {node.elapsed != null && (
            <span className="text-[10px] flex-shrink-0 tabular-nums" style={{ color: "var(--text-secondary)" }}>
              {node.elapsed}s
            </span>
          )}
          {isRunning && (
            <div className="flex items-center gap-1 flex-shrink-0">
              <button
                type="button"
                disabled={busy}
                onClick={async (e) => {
                  e.stopPropagation();
                  const confirmed = window.confirm(t.confirmKillOne.replace("{id}", node.run_id));
                  if (!confirmed) return;
                  setBusy(true);
                  try {
                    const res = await api.killSubagent(currentAgentId, node.run_id);
                    if (res?.ok) {
                      onNotice({ kind: "success", text: `已终止 ${node.run_id}` });
                    } else {
                      onNotice({ kind: "error", text: res?.error || "终止失败" });
                    }
                    await onRefresh();
                    await loadMainSession();
                  } finally {
                    setBusy(false);
                  }
                }}
                className="btn-outline" style={{ padding: "2px 6px", fontSize: 10 }}
              >
                <Square className="w-2.5 h-2.5" /> Kill
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowSteerInput((v) => !v);
                }}
                className="btn-outline" style={{ padding: "2px 6px", fontSize: 10 }}
              >
                <Send className="w-2.5 h-2.5" /> Steer
              </button>
            </div>
          )}
        </div>

        <div className={`grid transition-all duration-200 ease-out ${isExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
          <div className="overflow-hidden">
            <div className="max-h-[500px] overflow-y-auto" style={{ borderTop: "1px solid var(--border)", background: "var(--bg-inset)" }}>
              {showSteerInput && isRunning && (
                <div className="mx-2 mt-2 p-2.5 rounded-lg glass-card">
                  <div className="text-[10px] mb-1" style={{ color: "var(--text-secondary)" }}>新的引导指令</div>
                  <div className="mb-1 flex flex-wrap gap-1">
                    {lastSteerPrompt && (
                      <button type="button" onClick={() => setSteerText(lastSteerPrompt)} className="chip" style={{ fontSize: 10 }}>
                        复用上次指令
                      </button>
                    )}
                    {STEER_TEMPLATES.map((tpl, idx) => (
                      <button key={idx} type="button" onClick={() => setSteerText(tpl)} className="chip" style={{ fontSize: 10 }}>
                        模板{idx + 1}
                      </button>
                    ))}
                  </div>
                  <textarea
                    value={steerText}
                    onChange={(e) => setSteerText(e.target.value)}
                    placeholder="输入 steer 指令..."
                    className="input min-h-16 text-[11px]"
                  />
                  <div className="mt-1.5 flex items-center justify-end gap-1.5">
                    <button type="button" onClick={() => setShowSteerInput(false)} className="btn-ghost" style={{ fontSize: 10 }}>
                      取消
                    </button>
                    <button
                      type="button"
                      disabled={busy || !steerText.trim()}
                      onClick={async () => {
                        const msg = steerText.trim();
                        if (!msg) return;
                        setBusy(true);
                        try {
                          const res = await api.steerSubagent(currentAgentId, node.run_id, msg);
                          if (res?.ok) {
                            onNotice({ kind: "success", text: `已引导 ${node.run_id} -> ${res.run_id}` });
                            setLastSteerPrompt(msg);
                            setSteerText("");
                            setShowSteerInput(false);
                          } else {
                            onNotice({ kind: "error", text: res?.error || "引导失败" });
                          }
                          await onRefresh();
                          await loadMainSession();
                        } finally {
                          setBusy(false);
                        }
                      }}
                      className="btn-primary" style={{ fontSize: 10 }}
                    >
                      确认引导
                    </button>
                  </div>
                </div>
              )}
              {traces.length > 0 && (
                <div className="mx-2 mt-2 rounded-lg px-2.5 py-2 text-[10px]"
                  style={{ background: "var(--glass-subtle)", border: "1px solid var(--border)" }}>
                  <div className="mb-1 font-medium" style={{ color: "var(--text-secondary)" }}>实时过程</div>
                  <div className="space-y-0.5">
                    {traces.slice(-12).map((tr, i) => (
                      <div key={`${tr.ts}-${i}`} className="flex items-start gap-1.5">
                        <span style={{ color: "var(--text-tertiary)" }}>
                          {new Date(tr.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                        </span>
                        <span className="flex-1 break-words" style={{ color: "var(--text)" }}>{tr.text}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {node.messages.length === 0 ? (
                <div className="text-[10px] text-center py-4" style={{ color: "var(--text-tertiary)" }}>
                  {isRunning ? (traces.length > 0 ? "正在执行中..." : "等待输出...") : "无对话记录"}
                </div>
              ) : (
                <div className="p-2 space-y-2">
                  {node.messages.map((msg, i) => (
                    <ChatBubble key={i} msg={msg} />
                  ))}
                  {isRunning && (
                    <div className="flex items-center gap-1 py-1 px-2">
                      <span className="w-1 h-1 rounded-full animate-pulse-dot" style={{ background: "var(--accent)", animationDelay: "0ms" }} />
                      <span className="w-1 h-1 rounded-full animate-pulse-dot" style={{ background: "var(--accent)", animationDelay: "150ms" }} />
                      <span className="w-1 h-1 rounded-full animate-pulse-dot" style={{ background: "var(--accent)", animationDelay: "300ms" }} />
                    </div>
                  )}
                </div>
              )}
              {node.result_summary && !isRunning && (
                <div className="mx-2 mb-2 text-[11px] rounded-lg px-2.5 py-1.5"
                  style={{ background: "var(--success-bg)", color: "var(--success)", border: "1px solid var(--border)" }}>
                  <span className="font-medium">结果: </span>
                  <span className="break-words">{node.result_summary.slice(0, 300)}</span>
                </div>
              )}
              {node.result_delivery_state && (
                <div className="mx-2 mb-2 text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                  delivery: {node.result_delivery_state}
                  {(node.announce_retry_count || 0) > 0 ? ` (retry ${node.announce_retry_count})` : ""}
                  {node.descendants_active_count ? ` | descendants running ${node.descendants_active_count}` : ""}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      {hasChildren && (isExpanded || node.children!.some((c) => (c.state || c.status) === "running")) && (
        <div className="space-y-1">
          {node.children!.map((child) => (
            <SubagentTreeNode
              key={child.run_id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              toggleExpand={toggleExpand}
              onNotice={onNotice}
              onRefresh={onRefresh}
              traceMap={traceMap}
              lastSteerPrompt={lastSteerPrompt}
              setLastSteerPrompt={setLastSteerPrompt}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function formatTerminalMeta(node: SubagentTreeItem): string {
  const bits: string[] = [];
  if (node.state && node.state !== "running") bits.push(node.state);
  if (node.terminal_reason) bits.push(node.terminal_reason);
  if (node.archive_at_ms) {
    const left = Math.max(0, Math.floor((node.archive_at_ms - Date.now()) / 1000));
    bits.push(`archive in ${left}s`);
  }
  return bits.join(" | ");
}

function StatusIcon({ status }: { status: string }) {
  if (status === "running") return <Loader2 className="w-3.5 h-3.5 animate-spin flex-shrink-0" style={{ color: "var(--accent)" }} />;
  if (["completed", "completed-empty", "completed-with-errors", "succeeded"].includes(status)) {
    return <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--success)" }} />;
  }
  return <XCircle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--error)" }} />;
}

function ChatBubble({ msg }: { msg: SubagentMessage }) {
  const { t } = useUi();
  const isUser = msg.role === "user";
  const isAssistant = msg.role === "assistant";
  const hasToolCalls = msg.tool_calls && msg.tool_calls.length > 0;
  const isEmpty = !msg.content && !hasToolCalls;
  const isTextToolCall = isAssistant && msg.content?.startsWith("functions.");

  return (
    <div className={`flex gap-1.5 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
        style={{ background: isUser ? "var(--accent-muted)" : "var(--hover)" }}>
        {isUser
          ? <User className="w-2.5 h-2.5" style={{ color: "var(--accent)" }} />
          : <Bot className="w-2.5 h-2.5" style={{ color: "var(--text-secondary)" }} />}
      </div>

      <div className="flex-1 min-w-0 space-y-1">
        {hasToolCalls && (
          <div className="space-y-0.5">
            {msg.tool_calls!.map((tc, j) => (
              <ToolCallItem key={j} tc={tc} />
            ))}
          </div>
        )}

        {isEmpty && !isUser && (
          <span className="text-[10px] italic" style={{ color: "var(--text-tertiary)" }}>({t.noReply})</span>
        )}

        {isTextToolCall && !hasToolCalls && (
          <div className="flex items-center gap-1 text-[10px]" style={{ color: "var(--text-secondary)" }}>
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>{t.parsingTools}</span>
          </div>
        )}

        {!isEmpty && !isTextToolCall && msg.content && (
          <div className="text-[11px] rounded-lg px-2 py-1.5 inline-block max-w-full"
            style={isUser ? {
              background: "var(--accent)",
              color: "var(--text-on-accent)",
            } : {
              background: "var(--glass-heavy)",
              border: "1px solid var(--border)",
              color: "var(--text)",
            }}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap break-words">{msg.content}</p>
            ) : (
              <div className="prose prose-xs max-w-none [&_p]:my-0.5 [&_li]:my-0 [&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs [&_code]:text-[10px]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ToolCallItem({ tc }: { tc: { tool: string; input: any; output: string } }) {
  const [open, setOpen] = useState(false);
  const inputPreview = typeof tc.input === "string"
    ? tc.input.slice(0, 50)
    : JSON.stringify(tc.input).slice(0, 50);

  return (
    <div className="rounded-lg overflow-hidden text-[10px]"
      style={{ border: "1px solid var(--border)", background: "var(--glass-subtle)" }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-1 px-2 py-1 transition-colors text-left"
        onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <ChevronRight className={`w-2.5 h-2.5 flex-shrink-0 transition-transform duration-150 ${open ? "rotate-90" : ""}`}
          style={{ color: "var(--text-secondary)" }} />
        <Wrench className="w-2.5 h-2.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
        <span className="font-mono font-medium" style={{ color: "var(--text)" }}>{tc.tool}</span>
        {!open && <span className="truncate ml-1 min-w-0" style={{ color: "var(--text-secondary)" }}>{inputPreview}</span>}
      </button>
      <div className={`grid transition-all duration-150 ease-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden">
          <div className="px-2 pb-1.5 pt-0.5 space-y-1" style={{ borderTop: "1px solid var(--border)" }}>
            <div>
              <span className="text-[9px] uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>Input</span>
              <pre className="mt-0.5 glass-inset rounded p-1.5 overflow-x-auto whitespace-pre-wrap text-[10px]" style={{ color: "var(--text)" }}>
                {typeof tc.input === "string" ? tc.input : JSON.stringify(tc.input, null, 2)}
              </pre>
            </div>
            {tc.output && (
              <div>
                <span className="text-[9px] uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>Output</span>
                <pre className="mt-0.5 glass-inset rounded p-1.5 overflow-x-auto whitespace-pre-wrap text-[10px] max-h-32 overflow-y-auto" style={{ color: "var(--text)" }}>
                  {tc.output}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
