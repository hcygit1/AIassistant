/**
 * 前端本地消息队列 — agent 忙时缓冲用户消息，空闲时自动出队发送。
 *
 * 入队：纯内存 + sessionStorage 持久化（防刷新丢失）
 * 出队：由 useChat 在 turn 结束后自动消费
 */

export interface QueuedMessage {
  text: string;
  timestamp: number;
}

const STORAGE_KEY = "clawchain_msg_queue";

let queue: QueuedMessage[] = [];

// 启动时从 sessionStorage 恢复
try {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved) queue = JSON.parse(saved);
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

export function enqueue(text: string): number {
  queue.push({ text: text.trim(), timestamp: Date.now() });
  persist();
  return queue.length;
}

export function dequeue(): QueuedMessage | null {
  const item = queue.shift() || null;
  persist();
  return item;
}

export function clear(): void {
  queue = [];
  persist();
}
