"use client";

import React, { createContext, useContext, useMemo } from "react";

import type { useSubagents } from "./hooks/useSubagents";

export type SubagentContextState = ReturnType<typeof useSubagents>;

const SubagentContext = createContext<SubagentContextState | null>(null);

export function SubagentProvider({
  subagents,
  children,
}: {
  subagents: SubagentContextState;
  children: React.ReactNode;
}) {
  const value = useMemo<SubagentContextState>(() => ({
    tree: subagents.tree,
    flat: subagents.flat,
    runningSubagents: subagents.runningSubagents,
    traceMap: subagents.traceMap,
    loading: subagents.loading,
    refreshSubagents: subagents.refreshSubagents,
  }), [
    subagents.tree,
    subagents.flat,
    subagents.runningSubagents,
    subagents.traceMap,
    subagents.loading,
    subagents.refreshSubagents,
  ]);

  return (
    <SubagentContext.Provider value={value}>
      {children}
    </SubagentContext.Provider>
  );
}

export function useSubagentContext(): SubagentContextState {
  const context = useContext(SubagentContext);
  if (!context) {
    throw new Error("useSubagentContext must be used inside SubagentProvider");
  }
  return context;
}
