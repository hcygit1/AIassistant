"use client";

import { useEffect, useRef } from "react";
import * as api from "@/lib/api";
import { useApp } from "@/lib/store";
import { useChatState } from "@/lib/chatContext";
import { useUi } from "@/lib/uiContext";
import Navbar from "@/components/layout/Navbar";
import ChatPanel from "@/components/chat/ChatPanel";
import InspectorPanel from "@/components/editor/InspectorPanel";
import ConfigModal from "@/components/layout/ConfigModal";
import ApprovalModal from "@/components/layout/ApprovalModal";
import MemoryModal from "@/components/memory/MemoryModal";
import ResizeHandle from "@/components/layout/ResizeHandle";
import WorkspaceRail from "@/components/layout/WorkspaceRail";

export default function HomePage() {
  const {
    loadAgents,
    loadMainSession,
    inspectorWidth,
    setInspectorWidth,
    isCompactLayout,
    inspectorPanelMode,
  } = useApp();
  const { sessionError } = useChatState();
  const { uiNotice, clearNotice, setShowConfigModal } = useUi();
  const initCheckedRef = useRef(false);

  useEffect(() => {
    loadAgents();
    loadMainSession();
  }, [loadAgents, loadMainSession]);

  useEffect(() => {
    if (initCheckedRef.current) return;
    initCheckedRef.current = true;

    api.fetchInitStatus().then((status) => {
      if (!status.config_ready) {
        setShowConfigModal(true);
      }
    }).catch(() => {});
  }, [setShowConfigModal]);

  useEffect(() => {
    if (!uiNotice) return;
    const timer = setTimeout(() => clearNotice(), 3500);
    return () => clearTimeout(timer);
  }, [uiNotice, clearNotice]);

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: "var(--bg)" }}>
      <Navbar />
      <ConfigModal />
      <ApprovalModal />
      <MemoryModal />

      {/* Toast：左侧弹出，与聊天框保持间距 */}
      {uiNotice && (
        <div className={`toast toast--${uiNotice.kind} animate-slide-in-left`}>
          {uiNotice.text}
        </div>
      )}

      {/* Session error */}
      {sessionError && (
        <div className="px-4 py-2 text-xs font-medium"
          style={{ background: "var(--error-bg)", color: "var(--error)", borderBottom: "1px solid var(--border)" }}>
          {sessionError}
        </div>
      )}

      {/* Main workspace with glass inset */}
      <div
        className="relative m-2 mt-1 flex flex-1 overflow-hidden rounded-[24px]"
        style={{
          background: "var(--bg-elevated)",
          border: "1px solid var(--border)",
          boxShadow: "var(--shadow-sm)",
        }}
      >
        <WorkspaceRail />
        <div
          className="w-[1px] h-full flex-shrink-0"
          style={{ background: "linear-gradient(to bottom, transparent, var(--border), transparent)" }}
        />
        {inspectorPanelMode === "docked" && !isCompactLayout && (
          <>
            <div
              style={{ width: inspectorWidth, minWidth: 280, borderRight: "1px solid var(--border)" }}
              className="hidden flex-shrink-0 overflow-hidden bg-[var(--bg)] md:flex md:flex-col"
            >
              <InspectorPanel />
            </div>
            <div className="hidden md:block">
              <ResizeHandle onResize={(delta) => setInspectorWidth(Math.max(280, inspectorWidth + delta))} />
            </div>
          </>
        )}
        <div className="flex-1 min-w-0 overflow-hidden">
          <ChatPanel />
        </div>
        <div
          className={`absolute inset-y-0 z-20 overflow-hidden flex flex-col transition-all duration-300 ease-out ${
            inspectorPanelMode === "overlay"
              ? "translate-x-0 opacity-100 pointer-events-auto"
              : "-translate-x-full opacity-0 pointer-events-none"
          }`}
          style={{
            left: "4rem",
            width: `min(${inspectorWidth}px, calc(100% - 4rem))`,
            minWidth: 0,
            background: "var(--bg)",
            borderRight: "1px solid var(--border)",
            boxShadow: "var(--shadow-xl)",
          }}
        >
          <InspectorPanel />
        </div>
      </div>
    </div>
  );
}
