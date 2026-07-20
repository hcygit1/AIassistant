"use client";

import React, { createContext, useContext, useMemo } from "react";

import type { useInspectorState } from "./hooks/useInspectorState";

type InspectorSource = ReturnType<typeof useInspectorState>;

export type InspectorContextState = Omit<InspectorSource, "resetInspector">;

const InspectorContext = createContext<InspectorContextState | null>(null);

export function InspectorProvider({
  inspector,
  children,
}: {
  inspector: InspectorSource;
  children: React.ReactNode;
}) {
  const value = useMemo<InspectorContextState>(() => ({
    inspectorWidth: inspector.inspectorWidth,
    setInspectorWidth: inspector.setInspectorWidth,
    isCompactLayout: inspector.isCompactLayout,
    inspectorPanelMode: inspector.inspectorPanelMode,
    setInspectorPanelMode: inspector.setInspectorPanelMode,
    inspectorTab: inspector.inspectorTab,
    setInspectorTab: inspector.setInspectorTab,
    inspectorFile: inspector.inspectorFile,
    inspectorFileLoading: inspector.inspectorFileLoading,
    openFile: inspector.openFile,
    saveInspectorFile: inspector.saveInspectorFile,
  }), [
    inspector.inspectorWidth,
    inspector.setInspectorWidth,
    inspector.isCompactLayout,
    inspector.inspectorPanelMode,
    inspector.setInspectorPanelMode,
    inspector.inspectorTab,
    inspector.setInspectorTab,
    inspector.inspectorFile,
    inspector.inspectorFileLoading,
    inspector.openFile,
    inspector.saveInspectorFile,
  ]);

  return (
    <InspectorContext.Provider value={value}>
      {children}
    </InspectorContext.Provider>
  );
}

export function useInspectorContext(): InspectorContextState {
  const context = useContext(InspectorContext);
  if (!context) {
    throw new Error("useInspectorContext must be used inside InspectorProvider");
  }
  return context;
}
