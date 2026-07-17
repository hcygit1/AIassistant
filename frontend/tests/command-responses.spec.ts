import { expect, test } from "@playwright/test";

import { formatCommandResponse } from "../src/lib/commandResponses";
import { getMessages } from "../src/lib/i18n/locales";

test("localizes the backend help list", () => {
  const messages = getMessages("en-US");
  const formatted = formatCommandResponse(
    "## 可用命令\n- `/new`\n- `/reset`\n- `/help`",
    messages,
  );

  expect(formatted).toContain(`## ${messages.helpListTitle}`);
  expect(formatted).toContain(`- \`/compact\` — ${messages.cmdCompactDesc}`);
  expect(formatted).toContain(`- \`/whoami\` — ${messages.cmdWhoamiDesc}`);
});

test("normalizes a compaction skip reason and preserves backend details", () => {
  const messages = getMessages("zh-CN");
  expect(formatCommandResponse(
    "压缩未执行：消息过少\n当前消息数: 2",
    messages,
  )).toBe(`${messages.cmdCompactSkipped}\n- ${messages.cmdCompactReasonTooFewMessages}\n\n当前消息数: 2`);
});

test("normalizes reset completion with queued memory persistence", () => {
  const messages = getMessages("en-US");
  expect(formatCommandResponse(
    "会话已重置，长期记忆将在后台保存",
    messages,
  )).toBe(`${messages.cmdResetDoneWithMemory}\n${messages.cmdResetDoneQueued}`);
});
