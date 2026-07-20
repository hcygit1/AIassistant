"use client";

import { useRef, useEffect } from "react";
import { useApp } from "@/lib/store";
import { useChatState } from "@/lib/chatContext";
import { useInspectorContext } from "@/lib/inspectorContext";
import { useUi } from "@/lib/uiContext";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import RetrievalCard from "./RetrievalCard";
import { Activity } from "lucide-react";
import PIPIXIAMark from "@/components/icons/PipixiaMark";

export default function ChatPanel() {
  const {
    currentSessionId,
    currentAgentId, agents, runningSubagents,
  } = useApp();
  const { messages, sessionError } = useChatState();
  const { setInspectorTab } = useInspectorContext();
  const { t } = useUi();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const currentAgent = agents.find((a: any) => a.id === currentAgentId);

  if (!currentSessionId || messages.length === 0) {
    return (
      <div
        className="flex h-full flex-col"
        style={{
          background: "linear-gradient(180deg, color-mix(in srgb, var(--bg) 92%, var(--bg-elevated) 8%) 0%, var(--bg) 100%)",
        }}
        data-testid="chat-panel"
      >
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-fade-in-up px-8 text-center max-w-lg">
            <div
              className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-2xl"
              style={{ background: "var(--accent-muted)", color: "var(--accent)" }}
            >
              <PIPIXIAMark className="h-5 w-5" strokeWidth={1.9} />
            </div>
            <h2 className="mb-2 text-2xl font-semibold text-[var(--text)]">
              {currentAgent?.name || "PIPIXIA"}
            </h2>
            <p className="mb-5 text-sm leading-relaxed text-[var(--text-secondary)]">
              {currentAgent?.description || t.agentDescription}
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {["记住这个", "查看今天的记忆", "帮我写代码", "搜索网页"].map((hint) => (
                <span key={hint} className="chip" style={{ background: "transparent" }}>{hint}</span>
              ))}
            </div>
            {sessionError && (
              <p className="mt-5 text-xs px-4 py-2.5 rounded-lg inline-block"
                style={{ background: "var(--error-bg)", color: "var(--error)" }}>
                {sessionError}
              </p>
            )}
          </div>
        </div>
        <ChatInput />
      </div>
    );
  }

  const lastAssistantIdx = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  })();

  return (
    <div
      className="flex h-full flex-col"
      style={{
        background: "linear-gradient(180deg, color-mix(in srgb, var(--bg) 94%, var(--bg-elevated) 6%) 0%, var(--bg) 100%)",
      }}
      data-testid="chat-panel"
    >
      {/* Subagent running banner */}
      {runningSubagents.length > 0 && (
        <div className="px-4 py-2 flex items-center gap-2 text-xs font-medium"
          style={{ background: "var(--warning-bg)", color: "var(--warning)", borderBottom: "1px solid var(--border)" }}>
          <Activity className="w-3.5 h-3.5" />
          <span>{runningSubagents.length} 个子 Agent 运行中</span>
          <button
            type="button"
            onClick={() => setInspectorTab("subagents")}
            className="ml-1 underline underline-offset-2 hover:no-underline transition-all"
          >
            查看详情
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-7">
        <div className="mx-auto max-w-[860px] space-y-5">
          {messages.map((msg, i) => {
            const prev = i > 0 ? messages[i - 1] : null;
            const isContinuation = prev?.role === "assistant" && msg.role === "assistant";

            return (
              <div key={msg.id} className="animate-fade-in">
                {msg.retrievals && msg.retrievals.length > 0 && (
                  <div className="mb-2">
                    <RetrievalCard results={msg.retrievals} />
                  </div>
                )}
                <ChatMessage
                  message={msg}
                  hideAvatar={isContinuation}
                  isLast={i === lastAssistantIdx}
                />
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>
      </div>

      <ChatInput />
    </div>
  );
}
