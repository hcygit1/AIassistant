import type { SSEEvent } from "./api";
import type {
  AgentChatRuntime,
  AgentChatState,
  ChatMessage,
} from "./chatState";

export interface ChatStreamState {
  doneReceived: boolean;
  terminalErrorReceived: boolean;
}

export interface ChatStreamEventDependencies {
  assistantMessageId: string;
  runtime: AgentChatRuntime;
  streamState: ChatStreamState;
  updateMessages: (updater: (messages: ChatMessage[]) => ChatMessage[]) => void;
  patchState: (patch: Partial<AgentChatState>) => void;
  addLifecycleEvent?: (event: SSEEvent) => void;
  onSessionCompacted?: () => void;
  formatCommandResponse?: (raw: string) => string;
}

export function createChatStreamEventHandler({
  assistantMessageId,
  runtime,
  streamState,
  updateMessages,
  patchState,
  addLifecycleEvent,
  onSessionCompacted,
  formatCommandResponse,
}: ChatStreamEventDependencies) {
  runtime.segmentToolCalls = [];

  return (event: SSEEvent) => {
    switch (event.type) {
      case "token":
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          content: message.content + (event.content || ""),
        })));
        break;

      case "clear_content":
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          content: "",
        })));
        break;

      case "content_refresh":
        if (typeof event.content === "string") {
          updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
            ...message,
            content: event.content!,
          })));
        }
        break;

      case "tool_start": {
        const toolCall = {
          tool: event.tool || event.name || "",
          input: event.input ?? event.args ?? {},
          output: "",
        };
        runtime.segmentToolCalls = [...runtime.segmentToolCalls, toolCall];
        const toolCalls = runtime.segmentToolCalls;
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          toolCalls,
        })));
        break;
      }

      case "tool_end": {
        const output = event.output || event.result || "";
        const toolName = event.tool || event.name || "";
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => {
          if (!message.toolCalls?.length) return message;
          const targetIndex = message.toolCalls.findIndex(
            (toolCall) => !(toolCall.output ?? toolCall.result)
              && (!toolName || (toolCall.tool || toolCall.name) === toolName),
          );
          const fallbackIndex = targetIndex >= 0
            ? targetIndex
            : message.toolCalls.findIndex((toolCall) => !(toolCall.output ?? toolCall.result));
          const index = fallbackIndex >= 0 ? fallbackIndex : message.toolCalls.length - 1;
          return {
            ...message,
            toolCalls: message.toolCalls.map((toolCall, currentIndex) => (
              currentIndex === index ? { ...toolCall, output } : toolCall
            )),
          };
        }));

        if (runtime.segmentToolCalls.length > 0) {
          const targetIndex = runtime.segmentToolCalls.findIndex(
            (toolCall) => !toolCall.output && (!toolName || toolCall.tool === toolName),
          );
          const fallbackIndex = targetIndex >= 0
            ? targetIndex
            : runtime.segmentToolCalls.findIndex((toolCall) => !toolCall.output);
          const index = fallbackIndex >= 0 ? fallbackIndex : runtime.segmentToolCalls.length - 1;
          runtime.segmentToolCalls = runtime.segmentToolCalls.map((toolCall, currentIndex) => (
            currentIndex === index ? { ...toolCall, output } : toolCall
          ));
        }
        break;
      }

      case "new_response":
        runtime.segmentToolCalls = [];
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          isStreaming: false,
        })));
        break;

      case "retrieval":
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          retrievals: event.results || [],
        })));
        break;

      case "command_response": {
        const response = formatCommandResponse
          ? formatCommandResponse(event.response || "")
          : (event.response || "");
        const text = response.trim();
        if (!text) break;
        updateMessages((messages) => {
          const index = messages.length - 1;
          const last = messages[index];
          const commandMessage: ChatMessage = {
            id: `command-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            role: "command",
            content: text,
            createdAt: Date.now(),
            isStreaming: false,
          };
          if (last?.role === "assistant" && !last.content.trim()) {
            const updated = messages.slice();
            updated[index] = commandMessage;
            return updated;
          }
          return [...messages, commandMessage];
        });
        break;
      }

      case "session_reset": {
        patchState({ lifecycleEvents: [], lastUsage: null });
        const nextAssistantId = `assistant-${Date.now()}`;
        runtime.assistantMessageId = nextAssistantId;
        updateMessages((messages) => {
          const last = messages[messages.length - 1];
          if (last?.role !== "command") return messages;
          return [...messages, {
            id: nextAssistantId,
            role: "assistant",
            content: "",
            createdAt: Date.now(),
            isStreaming: true,
          }];
        });
        break;
      }

      case "session_compacted":
        onSessionCompacted?.();
        break;

      case "lifecycle":
        addLifecycleEvent?.(event);
        break;

      case "title":
        break;

      case "done":
        streamState.doneReceived = true;
        patchState({
          ...(event.usage ? { lastUsage: event.usage } : {}),
          ...(event.context_utilization != null
            ? { contextUtilization: event.context_utilization }
            : {}),
        });
        updateMessages((messages) => updateTargetMessage(
          messages,
          runtime.assistantMessageId || assistantMessageId,
          (message) => {
            if (message.role !== "assistant" && message.role !== "command") return message;
            const finishedAt = Date.now();
            const duration = event.usage?.duration_ms && event.usage.duration_ms > 0
              ? event.usage.duration_ms
              : Math.max(0, finishedAt - (message.createdAt || finishedAt));
            return {
              ...message,
              isStreaming: false,
              finishedAt,
              streamDurationMs: duration,
              ...(event.usage ? { usage: event.usage } : {}),
              ...(typeof event.content === "string" ? { content: event.content } : {}),
            };
          },
        ));
        break;

      case "aborted":
        streamState.doneReceived = true;
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          content: typeof event.content === "string" && event.content.length > 0
            ? event.content
            : message.content,
          isStreaming: false,
          finishedAt: Date.now(),
        })));
        break;

      case "error": {
        streamState.terminalErrorReceived = true;
        streamState.doneReceived = true;
        const error = event.error || "";
        const friendly = error.includes("401") || error.includes("invalid") || error.includes("Authentication")
          ? `**API Authentication Failed**: Please check the apiKey for the corresponding provider in config.json.\n\nOriginal error: ${error}`
          : `**Error:** ${error}`;
        updateMessages((messages) => updateAssistant(messages, runtime, assistantMessageId, (message) => ({
          ...message,
          content: `${message.content}\n\n${friendly}`,
          isStreaming: false,
        })));
        break;
      }
    }
  };
}

function updateAssistant(
  messages: ChatMessage[],
  runtime: AgentChatRuntime,
  fallbackAssistantId: string,
  update: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  return updateTargetMessage(
    messages,
    runtime.assistantMessageId || fallbackAssistantId,
    (message) => message.role === "assistant" ? update(message) : message,
  );
}

function updateTargetMessage(
  messages: ChatMessage[],
  targetId: string,
  update: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  const index = messages.findIndex((message) => message.id === targetId);
  if (index < 0) return messages;
  const previous = messages[index];
  const next = update(previous);
  if (next === previous) return messages;
  const updated = messages.slice();
  updated[index] = next;
  return updated;
}
