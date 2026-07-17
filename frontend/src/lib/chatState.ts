import type { TokenUsage } from "./api";

export interface ChatToolCall {
  tool?: string;
  name?: string;
  input?: any;
  output?: string;
  result?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "command";
  content: string;
  createdAt: number;
  finishedAt?: number;
  streamDurationMs?: number;
  toolCalls?: ChatToolCall[];
  retrievals?: any[];
  isStreaming?: boolean;
  usage?: TokenUsage;
}

export interface LifecycleEvent {
  type: string;
  event: string;
  run_id?: string;
  timestamp: number;
  data?: any;
}

export interface AgentChatState {
  messages: ChatMessage[];
  isStreaming: boolean;
  lifecycleEvents: LifecycleEvent[];
  lastUsage: TokenUsage | null;
  contextUtilization: number | null;
  sessionError: string | null;
}

export interface ChatState {
  byAgent: Record<string, AgentChatState>;
}

type MessageUpdater = ChatMessage[] | ((messages: ChatMessage[]) => ChatMessage[]);

export type ChatStateAction =
  | { type: "messages"; agentId: string; updater: MessageUpdater }
  | { type: "patch"; agentId: string; patch: Partial<AgentChatState> }
  | { type: "lifecycle"; agentId: string; event: LifecycleEvent }
  | { type: "clear"; agentId: string };

export interface AgentChatRuntime {
  controller: AbortController | null;
  userStopped: boolean;
  assistantMessageId: string | null;
  turnId: string | null;
  sessionId: string | null;
  segmentToolCalls: ChatToolCall[];
}

export type ChatRuntimeRegistry = Map<string, AgentChatRuntime>;

const EMPTY_AGENT_CHAT_STATE: AgentChatState = {
  messages: [],
  isStreaming: false,
  lifecycleEvents: [],
  lastUsage: null,
  contextUtilization: null,
  sessionError: null,
};

function createAgentChatState(): AgentChatState {
  return {
    messages: [],
    isStreaming: false,
    lifecycleEvents: [],
    lastUsage: null,
    contextUtilization: null,
    sessionError: null,
  };
}

export function createChatState(): ChatState {
  return { byAgent: {} };
}

export function selectAgentChatState(state: ChatState, agentId: string): AgentChatState {
  return state.byAgent[agentId] || EMPTY_AGENT_CHAT_STATE;
}

export function appendTurnMessages(
  messages: ChatMessage[],
  userMessage: ChatMessage | null,
  assistantMessage: ChatMessage,
  displayedUserMessageId?: string,
): ChatMessage[] {
  if (displayedUserMessageId) {
    const userIndex = messages.findIndex((message) => message.id === displayedUserMessageId);
    if (userIndex >= 0) {
      return [
        ...messages.slice(0, userIndex + 1),
        assistantMessage,
        ...messages.slice(userIndex + 1),
      ];
    }
  }
  return userMessage
    ? [...messages, userMessage, assistantMessage]
    : [...messages, assistantMessage];
}

export function chatStateReducer(state: ChatState, action: ChatStateAction): ChatState {
  if (action.type === "clear") {
    if (!state.byAgent[action.agentId]) return state;
    const byAgent = { ...state.byAgent };
    delete byAgent[action.agentId];
    return { byAgent };
  }

  const previous = state.byAgent[action.agentId] || createAgentChatState();
  let next: AgentChatState;
  if (action.type === "messages") {
    next = {
      ...previous,
      messages: typeof action.updater === "function"
        ? action.updater(previous.messages)
        : action.updater,
    };
  } else if (action.type === "lifecycle") {
    next = { ...previous, lifecycleEvents: [...previous.lifecycleEvents, action.event] };
  } else {
    next = { ...previous, ...action.patch };
  }

  return {
    byAgent: {
      ...state.byAgent,
      [action.agentId]: next,
    },
  };
}

function createAgentChatRuntime(): AgentChatRuntime {
  return {
    controller: null,
    userStopped: false,
    assistantMessageId: null,
    turnId: null,
    sessionId: null,
    segmentToolCalls: [],
  };
}

export function getAgentChatRuntime(
  registry: ChatRuntimeRegistry,
  agentId: string,
): AgentChatRuntime {
  const existing = registry.get(agentId);
  if (existing) return existing;
  const runtime = createAgentChatRuntime();
  registry.set(agentId, runtime);
  return runtime;
}

export function clearAgentChatRuntime(
  registry: ChatRuntimeRegistry,
  agentId: string,
): void {
  registry.delete(agentId);
}
