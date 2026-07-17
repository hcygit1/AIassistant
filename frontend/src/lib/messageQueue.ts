/**
 * 前端本地消息队列 — agent 忙时缓冲用户消息，空闲时自动出队发送。
 *
 * 入队：纯内存 + sessionStorage 持久化（防刷新丢失）
 * 出队：由 useChat 在 turn 结束后自动消费
 */

export interface QueuedMessage {
  agentId: string;
  text: string;
  messageId?: string;
  timestamp: number;
}

const STORAGE_KEY = "pipixia_msg_queue";

let queue: QueuedMessage[] = [];

// 启动时从 sessionStorage 恢复
try {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved) {
    const parsed = JSON.parse(saved);
    if (Array.isArray(parsed)) {
      queue = parsed
        .filter((item) => item && typeof item.text === "string")
        .map((item) => ({
          agentId: typeof item.agentId === "string" && item.agentId ? item.agentId : "main",
          text: item.text,
          ...(typeof item.messageId === "string" && item.messageId ? { messageId: item.messageId } : {}),
          timestamp: Number(item.timestamp) || Date.now(),
        }));
    }
  }
} catch {
  // ignore
}

function persist() {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(queue));
  } catch {
    // quota exceeded or unavailable
  }
}

export function enqueue(agentId: string, text: string, messageId?: string): number {
  queue.push({
    agentId,
    text: text.trim(),
    ...(messageId ? { messageId } : {}),
    timestamp: Date.now(),
  });
  persist();
  return queue.filter((item) => item.agentId === agentId).length;
}

export function dequeue(agentId: string): QueuedMessage | null {
  const index = queue.findIndex((item) => item.agentId === agentId);
  if (index < 0) return null;
  const [item] = queue.splice(index, 1);
  persist();
  return item || null;
}

export function clear(agentId?: string): void {
  queue = agentId ? queue.filter((item) => item.agentId !== agentId) : [];
  persist();
}
