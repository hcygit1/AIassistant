"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import * as api from "../api";
import { useChat } from "./useChat";
import { useSubagents } from "./useSubagents";

interface AgentWorkspaceOptions {
  currentAgentId: string;
  setCurrentAgentId: Dispatch<SetStateAction<string>>;
  formatCommandResponse: (raw: string) => string;
  onTurnComplete: () => void;
  resetInspector: () => void;
}

export function useAgentWorkspace({
  currentAgentId,
  setCurrentAgentId,
  formatCommandResponse,
  onTurnComplete,
  resetInspector,
}: AgentWorkspaceOptions) {
  const [agents, setAgents] = useState<any[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<any | null>(null);
  const loadMainSessionReqRef = useRef(0);
  const currentAgentIdRef = useRef("main");
  const chatRef = useRef<ReturnType<typeof useChat> | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const saved = window.localStorage.getItem("pipixia.agent");
      if (saved && typeof saved === "string" && saved.trim()) {
        const agentId = saved.trim();
        currentAgentIdRef.current = agentId;
        setCurrentAgentId(agentId);
      }
    } catch {
      // Ignore unavailable local storage.
    }
  }, [setCurrentAgentId]);

  useEffect(() => {
    currentAgentIdRef.current = currentAgentId;
  }, [currentAgentId]);

  const loadAgents = useCallback(async () => {
    const data = await api.fetchAgents();
    setAgents(data);
  }, []);

  const loadMainSession = useCallback(async () => {
    const chat = chatRef.current;
    if (!chat) return;
    const reqId = ++loadMainSessionReqRef.current;
    try {
      const session = await api.fetchMainSession(currentAgentId);
      if (reqId !== loadMainSessionReqRef.current) return;
      setCurrentSessionId(session.session_id);
      await chat.loadMessages(currentAgentId, session.session_id);
      if (reqId !== loadMainSessionReqRef.current) return;
      const model = await api.fetchCurrentModel(currentAgentId);
      if (reqId !== loadMainSessionReqRef.current) return;
      setCurrentModel(model);
    } catch {
      if (reqId !== loadMainSessionReqRef.current) return;
      chat.setMessages([]);
    }
  }, [currentAgentId]);

  const handleSessionCompacted = useCallback((agentId: string) => {
    if (currentAgentIdRef.current === agentId) {
      void loadMainSession();
    }
  }, [loadMainSession]);

  const chatOptions = useMemo(() => ({
    onAgentCreated: loadAgents,
    onSessionCompacted: handleSessionCompacted,
    onTurnComplete,
    formatCommandResponse,
  }), [
    formatCommandResponse,
    handleSessionCompacted,
    loadAgents,
    onTurnComplete,
  ]);

  const chat = useChat(
    currentAgentId,
    currentSessionId,
    setCurrentSessionId,
    chatOptions,
  );
  useEffect(() => {
    chatRef.current = chat;
  }, [chat]);

  const subagents = useSubagents(
    currentAgentId,
    currentSessionId,
    loadMainSession,
  );

  const switchAgent = useCallback(async (agentId: string) => {
    currentAgentIdRef.current = agentId;
    const reqId = ++loadMainSessionReqRef.current;
    setCurrentAgentId(agentId);
    try {
      window.localStorage.setItem("pipixia.agent", agentId);
    } catch {
      // Ignore unavailable local storage.
    }
    setCurrentSessionId(null);
    resetInspector();

    const currentChat = chatRef.current;
    if (!currentChat) return;
    try {
      const session = await api.fetchMainSession(agentId);
      if (
        reqId !== loadMainSessionReqRef.current
        || currentAgentIdRef.current !== agentId
      ) return;
      setCurrentSessionId(session.session_id);
      await currentChat.loadMessages(agentId, session.session_id);
      if (
        reqId !== loadMainSessionReqRef.current
        || currentAgentIdRef.current !== agentId
      ) return;
      const model = await api.fetchCurrentModel(agentId);
      if (
        reqId !== loadMainSessionReqRef.current
        || currentAgentIdRef.current !== agentId
      ) return;
      setCurrentModel(model);
    } catch {
      if (
        reqId === loadMainSessionReqRef.current
        && currentAgentIdRef.current === agentId
      ) {
        currentChat.clearAgent(agentId);
      }
    }
  }, [resetInspector, setCurrentAgentId]);

  return {
    agents,
    currentAgentId,
    currentSessionId,
    currentModel,
    loadAgents,
    loadMainSession,
    switchAgent,
    chat,
    ...subagents,
  };
}
