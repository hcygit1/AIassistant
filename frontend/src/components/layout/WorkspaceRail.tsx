"use client";

import {
  BrainCircuit,
  FolderOpen,
  Heart,
  ListTodo,
  PanelRightClose,
  PanelRightOpen,
  Settings2,
  Sparkles,
  Users,
  Wrench,
} from "lucide-react";
import { useApp } from "@/lib/store";
import { resolveInspectorOpenMode } from "@/lib/inspectorLayout";

type InspectorTabId = "files" | "tools" | "skills" | "subagents" | "heartbeat" | "tasks";

const INSPECTOR_ITEMS: { id: InspectorTabId; label: (t: any) => string; icon: any }[] = [
  { id: "files", label: (t) => t.tabFiles, icon: FolderOpen },
  { id: "tools", label: (t) => t.tabTools, icon: Wrench },
  { id: "skills", label: (t) => t.tabSkills, icon: Sparkles },
  { id: "subagents", label: (t) => t.tabSubagents, icon: Users },
  { id: "heartbeat", label: (t) => t.tabHeartbeat, icon: Heart },
  { id: "tasks", label: (t) => t.tabTasks, icon: ListTodo },
];

function RailButton({
  label,
  active = false,
  onClick,
  children,
}: {
  label: string;
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="group relative flex h-10 w-10 items-center justify-center rounded-2xl transition-all duration-200"
      style={{
        background: active ? "var(--accent-muted)" : "transparent",
        color: active ? "var(--accent)" : "var(--text-secondary)",
        border: active ? "1px solid transparent" : "1px solid transparent",
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.background = "var(--hover)";
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.background = "transparent";
      }}
    >
      {children}
      {active && (
        <span
          className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full"
          style={{ background: "var(--accent)" }}
        />
      )}
    </button>
  );
}

export default function WorkspaceRail() {
  const {
    inspectorPanelMode,
    isCompactLayout,
    inspectorTab,
    setInspectorPanelMode,
    setInspectorTab,
    setShowMemoryModal,
    setShowConfigModal,
    t,
  } = useApp();

  const openInspectorTab = (tab: InspectorTabId) => {
    setInspectorTab(tab);
    if (isCompactLayout) {
      setInspectorPanelMode(resolveInspectorOpenMode(true));
    } else if (inspectorPanelMode === "hidden") {
      setInspectorPanelMode(resolveInspectorOpenMode(false));
    }
  };

  return (
    <aside
      className="flex h-full w-16 flex-shrink-0 flex-col items-center justify-between px-2 py-3"
      style={{
        background: "linear-gradient(180deg, var(--bg-elevated) 0%, color-mix(in srgb, var(--bg-elevated) 85%, var(--bg) 15%) 100%)",
      }}
    >
      <div className="flex w-full flex-col items-center gap-1.5">
        {INSPECTOR_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = inspectorPanelMode !== "hidden" && inspectorTab === item.id;
          return (
            <RailButton
              key={item.id}
              label={item.label(t)}
              active={active}
              onClick={() => openInspectorTab(item.id)}
            >
              <Icon className="h-4 w-4" />
            </RailButton>
          );
        })}
      </div>

      <div className="flex w-full flex-col items-center gap-1.5">
        <RailButton label="记忆看板" onClick={() => setShowMemoryModal(true)}>
          <BrainCircuit className="h-4 w-4" />
        </RailButton>
        <RailButton label="配置中心" onClick={() => setShowConfigModal(true)}>
          <Settings2 className="h-4 w-4" />
        </RailButton>
        <RailButton
          label={inspectorPanelMode === "hidden" ? "展开侧栏" : "收起侧栏"}
          onClick={() => setInspectorPanelMode(
            inspectorPanelMode === "hidden"
              ? resolveInspectorOpenMode(isCompactLayout)
              : "hidden",
          )}
        >
          {inspectorPanelMode === "hidden" ? (
            <PanelRightOpen className="h-4 w-4" />
          ) : (
            <PanelRightClose className="h-4 w-4" />
          )}
        </RailButton>
      </div>
    </aside>
  );
}
