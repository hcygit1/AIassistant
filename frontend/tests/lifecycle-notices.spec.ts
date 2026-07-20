import { expect, test } from "@playwright/test";

import { resolveLifecycleNotice } from "../src/lib/hooks/useLifecycleNotices";

test("maps memory lifecycle events to user notices", () => {
  expect(resolveLifecycleNotice({
    type: "lifecycle",
    event: "session_memory_saved",
    timestamp: 1,
    data: { path: "memory/session.md" },
  })).toEqual({
    kind: "success",
    text: "长期记忆已后台保存（memory/session.md）",
  });

  expect(resolveLifecycleNotice({
    type: "lifecycle",
    event: "session_memory_failed",
    timestamp: 2,
    data: { reason: "disk full" },
  })).toEqual({
    kind: "error",
    text: "长期记忆后台保存失败：disk full",
  });

  expect(resolveLifecycleNotice({
    type: "lifecycle",
    event: "session_memory_saved",
    timestamp: 3,
  })).toEqual({
    kind: "success",
    text: "长期记忆已后台保存",
  });

  expect(resolveLifecycleNotice({
    type: "lifecycle",
    event: "session_memory_failed",
    timestamp: 4,
  })).toEqual({
    kind: "error",
    text: "长期记忆后台保存失败",
  });
});

test("ignores lifecycle events without a user notice", () => {
  expect(resolveLifecycleNotice({
    type: "lifecycle",
    event: "session_compacted",
    timestamp: 5,
  })).toBeNull();
});
