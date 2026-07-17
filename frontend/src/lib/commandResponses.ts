import type { Messages } from "./i18n/locales";

export function formatCommandResponse(raw: string, messages: Messages): string {
  const value = (raw || "").trim();
  if (!value) return value;

  const helpLines = [
    `## ${messages.helpListTitle}`,
    `- \`/new\` — ${messages.cmdNewDesc}`,
    `- \`/reset\` — ${messages.cmdResetDesc}`,
    `- \`/compact\` — ${messages.cmdCompactDesc}`,
    `- \`/help\` — ${messages.cmdHelpDesc}`,
    `- \`/status\` — ${messages.cmdStatusDesc}`,
    `- \`/context\` — ${messages.cmdContextDesc}`,
    `- \`/usage\` — ${messages.cmdUsageDesc}`,
    `- \`/model\` — ${messages.cmdModelDesc}`,
    `- \`/subagents\` — ${messages.cmdSubagentsDesc}`,
    `- \`/whoami\` — ${messages.cmdWhoamiDesc}`,
  ].join("\n");

  if (
    value.startsWith("## 可用命令")
    || value.startsWith("## Available Commands")
    || (value.includes("`/new`") && value.includes("`/reset`") && value.includes("`/help`"))
  ) {
    return helpLines;
  }

  const lower = value.toLowerCase();
  if (value.includes("正在执行压缩") || lower.includes("compaction in progress")) {
    return messages.cmdCompactProgress;
  }
  if (value.startsWith("压缩未执行：") || lower.startsWith("compaction skipped:")) {
    const lines = value.split("\n");
    const first = lines[0] || value;
    const rawReason = first.split("：").slice(1).join("：").trim()
      || first.split(":").slice(1).join(":").trim();
    let reason = rawReason;
    if (rawReason.includes("消息过少")) reason = messages.cmdCompactReasonTooFewMessages;
    else if (rawReason.includes("无足够消息可压缩")) reason = messages.cmdCompactReasonNoEnoughCompressible;
    else if (rawReason.includes("会话不存在")) reason = messages.cmdCompactReasonSessionMissing;
    const details = lines.slice(1).join("\n").trim();
    return `${messages.cmdCompactSkipped}\n${reason ? `- ${reason}` : ""}${details ? `\n\n${details}` : ""}`.trim();
  }
  if (value.startsWith("压缩完成。") || lower.startsWith("compaction completed")) {
    const lines = value.split("\n");
    return lines.length > 1
      ? `${messages.cmdCompactDone}\n${lines.slice(1).join("\n")}`
      : messages.cmdCompactDone;
  }
  if (value.startsWith("压缩失败") || lower.startsWith("compaction failed")) {
    return `${messages.cmdCompactFailed}\n${value}`;
  }

  if (value.includes("正在重置会话（写入长期记忆") || lower.includes("resetting session and saving long-term memory")) {
    return messages.cmdResetProgressWithMemory;
  }
  if (value.includes("正在重置会话（不写入长期记忆") || lower.includes("resetting session (without writing long-term memory")) {
    return messages.cmdResetProgressNoMemory;
  }
  if (value.includes("会话已重置（本轮对话未写入长期记忆）") || lower.includes("session has been reset (this round was not written")) {
    return messages.cmdResetDoneNoMemory;
  }
  if (value.includes("会话已重置")) {
    const queued = value.includes("长期记忆将在后台保存") || lower.includes("saved in the background");
    return queued
      ? `${messages.cmdResetDoneWithMemory}\n${messages.cmdResetDoneQueued}`
      : messages.cmdResetDoneWithMemory;
  }
  return value;
}
