"use client";

import React, { createContext, useContext, useMemo } from "react";

import type { TokenUsage } from "./api";
import { useLifecycleNotices } from "./hooks/useLifecycleNotices";
import type { ChatMessage, LifecycleEvent } from "./hooks/useChat";
import { useUi } from "./uiContext";

export interface ChatState {
  messages: ChatMessage[];
  isStreaming: boolean;
  lifecycleEvents: LifecycleEvent[];
  lastUsage: TokenUsage | null;
  contextUtilization: number | null;
  sessionError: string | null;
  sendMessage: (text: string) => Promise<void>;
  stopStreaming: () => void;
}

const ChatContext = createContext<ChatState | null>(null);

export function ChatProvider({
  chat,
  children,
}: {
  chat: ChatState;
  children: React.ReactNode;
}) {
  const { showNotice } = useUi();
  useLifecycleNotices(chat.lifecycleEvents, showNotice);

  const value = useMemo<ChatState>(() => ({
    messages: chat.messages,
    isStreaming: chat.isStreaming,
    lifecycleEvents: chat.lifecycleEvents,
    lastUsage: chat.lastUsage,
    contextUtilization: chat.contextUtilization,
    sessionError: chat.sessionError,
    sendMessage: chat.sendMessage,
    stopStreaming: chat.stopStreaming,
  }), [
    chat.messages,
    chat.isStreaming,
    chat.lifecycleEvents,
    chat.lastUsage,
    chat.contextUtilization,
    chat.sessionError,
    chat.sendMessage,
    chat.stopStreaming,
  ]);

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChatState(): ChatState {
  const context = useContext(ChatContext);
  if (!context) throw new Error("useChatState must be used inside ChatProvider");
  return context;
}
