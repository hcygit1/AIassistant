"use client";

import { useEffect, useRef } from "react";
import * as api from "@/lib/api";
import { useApp } from "@/lib/store";
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
    inspectorPanelMode,
    uiNotice,
    clearNotice,
    sessionError,
    setShowConfigModal,
    showNotice,
  } = useApp();
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
        showNotice({
          kind: "info",
          text: "检测到尚未完成初始化，请先在配置中心添加 Provider 并设置默认模型。",
        });
      }
    }).catch(() => {});
  }, [setShowConfigModal, showNotice]);

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
        {inspectorPanelMode === "docked" && (
          <>
            <div
              style={{ width: inspectorWidth, minWidth: 280, borderRight: "1px solid var(--border)" }}
              className="flex-shrink-0 overflow-hidden flex flex-col bg-[var(--bg)]"
            >
              <InspectorPanel />
            </div>
            <ResizeHandle onResize={(delta) => setInspectorWidth(Math.max(280, inspectorWidth + delta))} />
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
            width: inspectorWidth,
            minWidth: 280,
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
