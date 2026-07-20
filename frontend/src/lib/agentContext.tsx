"use client";

import React, { createContext, useContext, useMemo } from "react";

export interface AgentContextState {
  agents: any[];
  currentAgentId: string;
  currentSessionId: string | null;
  currentModel: any | null;
  loadAgents: () => Promise<void>;
  switchAgent: (agentId: string) => Promise<void>;
  loadMainSession: () => Promise<void>;
}

const AgentContext = createContext<AgentContextState | null>(null);

export function AgentProvider({
  agent,
  children,
}: {
  agent: AgentContextState;
  children: React.ReactNode;
}) {
  const value = useMemo<AgentContextState>(() => ({
    agents: agent.agents,
    currentAgentId: agent.currentAgentId,
    currentSessionId: agent.currentSessionId,
    currentModel: agent.currentModel,
    loadAgents: agent.loadAgents,
    switchAgent: agent.switchAgent,
    loadMainSession: agent.loadMainSession,
  }), [
    agent.agents,
    agent.currentAgentId,
    agent.currentSessionId,
    agent.currentModel,
    agent.loadAgents,
    agent.switchAgent,
    agent.loadMainSession,
  ]);

  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  );
}

export function useAgentContext(): AgentContextState {
  const context = useContext(AgentContext);
  if (!context) {
    throw new Error("useAgentContext must be used inside AgentProvider");
  }
  return context;
}
