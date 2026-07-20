"use client";

import { useEffect, useState } from "react";
import { useSubagentContext } from "@/lib/subagentContext";
import { useUi } from "@/lib/uiContext";
import { deriveSubagentViews } from "@/lib/subagentState";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronRight, Loader2, CheckCircle, XCircle, Wrench, Bot, User } from "lucide-react";

interface SubagentMessage {
  role: string;
  content: string;
  tool_calls?: { tool: string; input: any; output: string }[];
}

interface Props {
  runId: string;
  task: string;
}

export default function SubagentInlineCard({ runId, task }: Props) {
  const { flat: subagents, loading: subagentsLoading } = useSubagentContext();
  const [expanded, setExpanded] = useState(true);
  const match = deriveSubagentViews(subagents, runId).inline;
  const messages = (match?.messages || []) as SubagentMessage[];
  const status = match?.state || match?.status || (subagentsLoading ? "running" : "archived");
  const elapsed = match?.elapsed ?? null;
  const resultSummary = match?.result_summary || "";
  const label = match?.label || "子 Agent";
  const announceState = match?.result_delivery_state || "pending";
  const terminalReason = match?.terminal_reason || "";

  useEffect(() => {
    if (status !== "running" && !["pending", "queued", "delivering", "retrying"].includes(announceState)) {
      const timer = setTimeout(() => setExpanded(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [status, announceState]);

  const isExecuting = status === "running";
  const isMerging = !isExecuting && ["pending", "queued", "delivering", "retrying"].includes(announceState);
  const isRunning = isExecuting || isMerging;
  const isComplete = ["completed", "completed-empty", "completed-with-errors", "succeeded"].includes(status);
  const isError =
    status.startsWith("error") ||
    ["failed", "timed_out", "timeout", "cancelled", "killed", "interrupted", "orphaned"].includes(status);
  const statusText = isExecuting
    ? "执行中"
    : isMerging
      ? `等待主会话融合 (${formatAnnounceState(announceState)})`
      : formatAnnounceState(announceState);

  return (
    <div className="glass-card overflow-hidden my-1.5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left transition-colors"
        onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <ChevronRight className={`w-3 h-3 text-[var(--text-secondary)] flex-shrink-0 transition-transform duration-200 ${expanded ? "rotate-90" : ""}`} />
        {isRunning && <Loader2 className="w-3.5 h-3.5 animate-spin flex-shrink-0" style={{ color: isExecuting ? "var(--accent)" : "var(--warning)" }} />}
        {!isRunning && isComplete && <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--success)" }} />}
        {!isRunning && isError && <XCircle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--error)" }} />}
        <span className="text-xs font-medium text-[var(--text)] flex-shrink-0">{label}</span>
        <span className="text-[11px] text-[var(--text-secondary)] truncate min-w-0">{task}</span>
        <span className="text-[10px] text-[var(--text-tertiary)] flex-shrink-0">{statusText}</span>
        {elapsed != null && (
          <span className="text-[10px] text-[var(--text-secondary)] flex-shrink-0 tabular-nums ml-auto">{elapsed}s</span>
        )}
      </button>

      <div className={`grid transition-all duration-300 ease-out ${expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden">
          <div className="px-3 py-2 space-y-2 max-h-[400px] overflow-y-auto"
            style={{ borderTop: "1px solid var(--border)", background: "var(--bg-inset)" }}>
            {messages.length === 0 && isRunning && (
              <div className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] py-2 justify-center">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>{isExecuting ? "等待输出..." : "已完成，等待主会话融合..."}</span>
              </div>
            )}

            {messages.map((msg, i) => (
              <InlineMessage key={i} msg={msg} />
            ))}

            {isRunning && messages.length > 0 && (
              <div className="flex items-center gap-1 py-0.5">
                <span className="w-1 h-1 rounded-full animate-pulse-dot" style={{ background: "var(--accent)", animationDelay: "0ms" }} />
                <span className="w-1 h-1 rounded-full animate-pulse-dot" style={{ background: "var(--accent)", animationDelay: "200ms" }} />
                <span className="w-1 h-1 rounded-full animate-pulse-dot" style={{ background: "var(--accent)", animationDelay: "400ms" }} />
              </div>
            )}

            {resultSummary && !isRunning && (
              <div className="text-[11px] rounded-lg px-2.5 py-1.5"
                style={{ background: "var(--success-bg)", color: "var(--success)", border: "1px solid var(--border)" }}>
                {resultSummary.slice(0, 300)}
              </div>
            )}
            {!isExecuting && (
              <div className="text-[10px] text-[var(--text-tertiary)]">
                announce: {formatAnnounceState(announceState)}{terminalReason ? ` | ${terminalReason}` : ""}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function formatAnnounceState(state: string): string {
  switch (state) {
    case "pending":
      return "pending";
    case "queued":
      return "queued";
    case "delivering":
      return "delivering";
    case "retrying":
      return "retrying";
    case "delivered":
      return "delivered";
    case "dropped":
      return "dropped";
    default:
      return state || "unknown";
  }
}

function InlineMessage({ msg }: { msg: SubagentMessage }) {
  const { t } = useUi();
  const isUser = msg.role === "user";
  const isAssistant = msg.role === "assistant";
  const hasToolCalls = msg.tool_calls && msg.tool_calls.length > 0;
  const isEmpty = !msg.content && !hasToolCalls;
  const isTextToolCall = isAssistant && msg.content?.startsWith("functions.");

  return (
    <div className={`flex gap-1.5 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5`}
        style={{ background: isUser ? "var(--accent-muted)" : "var(--hover)" }}>
        {isUser ? <User className="w-2.5 h-2.5" style={{ color: "var(--accent)" }} /> : <Bot className="w-2.5 h-2.5" style={{ color: "var(--text-secondary)" }} />}
      </div>

      <div className="flex-1 min-w-0 space-y-0.5">
        {hasToolCalls && (
          <div className="space-y-px">
            {msg.tool_calls!.map((tc, j) => (
              <InlineToolCall key={j} tc={tc} />
            ))}
          </div>
        )}

        {isEmpty && !isUser && (
          <span className="text-[10px] text-[var(--text-secondary)] italic">({t.noReply})</span>
        )}

        {isTextToolCall && !hasToolCalls && (
          <div className="flex items-center gap-1 text-[10px] text-[var(--text-secondary)]">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>{t.parsingTools}</span>
          </div>
        )}

        {!isEmpty && !isTextToolCall && msg.content && (
          <div className={`text-[11px] rounded-lg px-2 py-1 inline-block max-w-full ${isUser ? "" : ""
            }`}
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
              <div className="prose prose-xs max-w-none [&_p]:my-0.5 [&_li]:my-0">
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

function InlineToolCall({ tc }: { tc: { tool: string; input: any; output: string } }) {
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
        <ChevronRight className={`w-2.5 h-2.5 text-[var(--text-secondary)] flex-shrink-0 transition-transform duration-150 ${open ? "rotate-90" : ""}`} />
        <Wrench className="w-2.5 h-2.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
        <span className="font-medium text-[var(--text)] flex-shrink-0">{tc.tool}</span>
        {!open && <span className="text-[var(--text-secondary)] truncate ml-1 min-w-0">{inputPreview}</span>}
      </button>
      {open && (
        <div className="px-2 pb-1.5 pt-0 space-y-1" style={{ borderTop: "1px solid var(--border)" }}>
          <pre className="glass-inset rounded p-1.5 overflow-x-auto whitespace-pre-wrap text-[var(--text)] text-[10px]">
            {typeof tc.input === "string" ? tc.input : JSON.stringify(tc.input, null, 2)}
          </pre>
          {tc.output && (
            <pre className="glass-inset rounded p-1.5 overflow-x-auto whitespace-pre-wrap text-[var(--text)] text-[10px] max-h-24 overflow-y-auto">
              {tc.output}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
