import { expect, test } from "@playwright/test";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

function listTsxFiles(root: string): string[] {
  return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const path = join(root, entry.name);
    if (entry.isDirectory()) return listTsxFiles(path);
    return entry.isFile() && path.endsWith(".tsx") ? [path] : [];
  });
}

test("keeps inspector persistence and file operations outside AppProvider", () => {
  const hookPath = "src/lib/hooks/useInspectorState.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const store = readFileSync("src/lib/store.tsx", "utf8");
  const inspector = readFileSync(hookPath, "utf8");

  expect(store).not.toContain("pipixia.inspector.");
  expect(store).not.toContain("api.readFile(");
  expect(store).not.toContain("api.saveFile(");
  expect(inspector).toContain("pipixia.inspector.mode");
  expect(inspector).toContain("api.readFile(");
  expect(inspector).toContain("api.saveFile(");
});

test("keeps locale, theme and modal state outside AppProvider", () => {
  const hookPath = "src/lib/hooks/useAppUiState.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const store = readFileSync("src/lib/store.tsx", "utf8");
  const uiState = readFileSync(hookPath, "utf8");

  expect(store).not.toContain("pipixia.locale");
  expect(store).not.toContain("useTheme(");
  expect(uiState).toContain("pipixia.locale");
  expect(uiState).toContain("useTheme(");
  expect(uiState).toContain("showConfigModal");
  expect(uiState).toContain("showMemoryModal");
});

test("exposes UI state through a dedicated context", () => {
  const contextPath = "src/lib/uiContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");

  expect(context).toContain("UiProvider");
  expect(context).toContain("useUi");
  expect(store).not.toContain("useAppUiState()");
});

test("keeps agent event subscription outside AppProvider", () => {
  const hookPath = "src/lib/hooks/useAgentEvents.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const store = readFileSync("src/lib/store.tsx", "utf8");
  const hook = readFileSync(hookPath, "utf8");

  expect(store).not.toContain("subscribeAgentEvents");
  expect(store).not.toContain("approval_required");
  expect(hook).toContain("subscribeAgentEvents");
  expect(hook).toContain("approval_required");
});

test("keeps approval state outside the workspace composition", () => {
  const contextPath = "src/lib/approvalContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const modal = readFileSync("src/components/layout/ApprovalModal.tsx", "utf8");

  expect(context).toContain("ApprovalProvider");
  expect(context).toContain("useApproval");
  expect(store).not.toContain("const [pendingApproval");
  expect(store).not.toContain("pendingApproval,");
  expect(modal).toContain("useApproval");
  expect(modal).not.toContain("pendingApproval } = useApp()");
});

test("keeps Agent workspace orchestration outside the provider composition", () => {
  const hookPath = "src/lib/hooks/useAgentWorkspace.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const hook = readFileSync(hookPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");

  expect(hook).toContain("useChat(");
  expect(hook).toContain("useSubagents(");
  expect(store).not.toContain("useChat(");
  expect(store).not.toContain("useSubagents(");
  expect(store).toContain("useAppWorkspace(");
});

test("keeps workspace runtime composition outside the provider composition", () => {
  const hookPath = "src/lib/hooks/useAppWorkspace.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const hook = readFileSync(hookPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");

  expect(hook).toContain("useAgentWorkspace(");
  expect(hook).toContain("useInspectorState(");
  expect(hook).toContain("useAgentEvents(");
  expect(store).toContain("useAppWorkspace(");
  expect(store).not.toContain("useInspectorState(");
  expect(store).not.toContain("useAgentEvents(");
});

test("keeps lifecycle notice side effects outside the provider composition", () => {
  const hookPath = "src/lib/hooks/useLifecycleNotices.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const hook = readFileSync(hookPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const chatContext = readFileSync("src/lib/chatContext.tsx", "utf8");

  expect(store).not.toContain("useLifecycleNotices(");
  expect(store).not.toContain("lastLifecycleNoticeKeyRef");
  expect(store).not.toContain('last.event === "session_memory_saved"');
  expect(store).not.toContain('last.event === "session_memory_failed"');
  expect(hook).toContain("session_memory_saved");
  expect(hook).toContain("session_memory_failed");
  expect(chatContext).toContain("useLifecycleNotices(");
});

test("isolates Chat state in a dedicated context broadcast domain", () => {
  const contextPath = "src/lib/chatContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const chatFields = [
    "messages",
    "isStreaming",
    "lifecycleEvents",
    "lastUsage",
    "contextUtilization",
    "sessionError",
    "sendMessage",
    "stopStreaming",
  ];
  const providerStart = store.indexOf("<ChatProvider chat={chat}>");

  expect(context).toContain("ChatProvider");
  expect(context).toContain("useChatState");
  expect(context).toContain("useMemo<ChatState>");
  expect(context).toContain("useLifecycleNotices(");
  expect(store).toContain("<ChatProvider chat={chat}>");
  expect(providerStart).toBeGreaterThanOrEqual(0);
  for (const field of chatFields) {
    expect(store, `${field} should not be in store composition`).not.toMatch(
      new RegExp(`\\b${field}\\b`),
    );
  }

  const chatConsumers = [
    "src/app/page.tsx",
    "src/components/chat/ChatInput.tsx",
    "src/components/chat/ChatPanel.tsx",
    "src/components/inspector/EventTimeline.tsx",
    "src/components/layout/Navbar.tsx",
  ];
  for (const file of chatConsumers) {
    expect(readFileSync(file, "utf8"), file).toContain("useChatState()");
  }
});

test("isolates Inspector state in a dedicated context broadcast domain", () => {
  const contextPath = "src/lib/inspectorContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const workspace = readFileSync("src/lib/hooks/useAppWorkspace.ts", "utf8");
  const inspectorFields = [
    "inspectorWidth",
    "setInspectorWidth",
    "isCompactLayout",
    "inspectorPanelMode",
    "setInspectorPanelMode",
    "inspectorTab",
    "setInspectorTab",
    "inspectorFile",
    "inspectorFileLoading",
    "openFile",
    "saveInspectorFile",
  ];

  expect(context).toContain("InspectorProvider");
  expect(context).toContain("useInspectorContext");
  expect(context).toContain("useMemo<InspectorContextState>");
  expect(store).toContain("<InspectorProvider inspector={inspector}>");
  expect(workspace).toContain("inspector,");
  for (const field of inspectorFields) {
    expect(store, `${field} should not be in store composition`).not.toMatch(
      new RegExp(`\\b${field}\\b`),
    );
  }

  for (const file of [
    "src/app/page.tsx",
    "src/components/layout/WorkspaceRail.tsx",
    "src/components/editor/InspectorPanel.tsx",
  ]) {
    expect(readFileSync(file, "utf8"), file).toContain(
      "useInspectorContext()",
    );
  }
});

test("isolates Subagent state in a dedicated context broadcast domain", () => {
  const contextPath = "src/lib/subagentContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) throw new Error(`${contextPath} is required`);

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const workspace = readFileSync("src/lib/hooks/useAgentWorkspace.ts", "utf8");
  const subagentFields = [
    "subagentTree",
    "subagents",
    "runningSubagents",
    "subagentTraceMap",
    "subagentsLoading",
    "refreshSubagents",
  ];

  expect(context).toContain("SubagentProvider");
  expect(context).toContain("useSubagentContext");
  expect(context).toContain("useMemo<SubagentContextState>");
  expect(store).toContain("<SubagentProvider subagents={subagents}>");
  expect(workspace).toContain("    subagents,");
  for (const field of subagentFields.filter((field) => field !== "subagents")) {
    expect(store, `${field} should not be in store composition`).not.toMatch(
      new RegExp(`\\b${field}\\b`),
    );
  }

  const consumers = [
    "src/components/chat/SubagentInlineCard.tsx",
    "src/components/chat/ChatPanel.tsx",
    "src/components/layout/Navbar.tsx",
    "src/components/inspector/SubagentPanel.tsx",
  ];
  for (const file of consumers) {
    expect(readFileSync(file, "utf8"), file).toContain("useSubagentContext()");
  }
});

test("isolates Agent state in a dedicated context broadcast domain", () => {
  const contextPath = "src/lib/agentContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) throw new Error(`${contextPath} is required`);

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const workspace = readFileSync("src/lib/hooks/useAgentWorkspace.ts", "utf8");
  const agentFields = [
    "agents",
    "currentAgentId",
    "currentSessionId",
    "currentModel",
    "loadAgents",
    "switchAgent",
    "loadMainSession",
  ];

  expect(context).toContain("AgentProvider");
  expect(context).toContain("useAgentContext");
  expect(context).toContain("useMemo<AgentContextState>");
  expect(store).toContain("<AgentProvider agent={agent}>");
  expect(workspace).toContain("    agent,");
  for (const field of agentFields) {
    expect(store, `${field} should not be in store composition`).not.toMatch(
      new RegExp(`\\b${field}\\b`),
    );
  }

  for (const file of [
    "src/app/page.tsx",
    "src/components/chat/ChatInput.tsx",
    "src/components/chat/ChatPanel.tsx",
    "src/components/editor/InspectorPanel.tsx",
    "src/components/inspector/HeartbeatPanel.tsx",
    "src/components/inspector/SubagentPanel.tsx",
    "src/components/layout/ConfigModal.tsx",
    "src/components/layout/Navbar.tsx",
    "src/components/memory/MemoryModal.tsx",
    "src/components/skills/SkillsPanel.tsx",
  ]) {
    expect(readFileSync(file, "utf8"), file).toContain("useAgentContext()");
  }
});

test("removes the generic AppContext and isolates skills refresh", () => {
  const contextPath = "src/lib/skillsRefreshContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const workspace = readFileSync("src/lib/hooks/useAppWorkspace.ts", "utf8");
  const skillsPanel = readFileSync(
    "src/components/skills/SkillsPanel.tsx",
    "utf8",
  );

  expect(context).toContain("SkillsRefreshProvider");
  expect(context).toContain("useSkillsRefresh");
  expect(store).toContain(
    "<SkillsRefreshProvider version={skillsRefreshTrigger}>",
  );
  expect(store).not.toContain("AppContext");
  expect(store).not.toContain("interface AppState");
  expect(store).not.toContain("function useApp");
  expect(workspace).not.toContain("ragMode");
  expect(workspace).not.toContain("updateRagMode");
  expect(skillsPanel).toContain("useSkillsRefresh()");
  expect(skillsPanel).not.toContain("useApp()");
});

test("removes generic useApp consumers", () => {
  const files = [
    ...listTsxFiles("src/app"),
    ...listTsxFiles("src/components"),
  ];

  for (const file of files) {
    const source = readFileSync(file, "utf8");
    expect(source, `${file} should use a dedicated context`).not.toContain(
      "useApp()",
    );
  }
});

test("keeps AppProvider limited to provider composition", () => {
  const store = readFileSync("src/lib/store.tsx", "utf8");

  expect(store).not.toContain("createContext");
  expect(store).not.toContain("useContext");
  expect(store).not.toContain("useMemo");
  expect(store).toContain("<UiProvider>");
  expect(store).toContain("<ApprovalProvider>");
  expect(store).toContain("<WorkspaceProviders>");
  expect(store).toContain("<SkillsRefreshProvider");
  expect(store).toContain("<InspectorProvider");
  expect(store).toContain("<ChatProvider");
  expect(store).toContain("<SubagentProvider");
  expect(store).toContain("<AgentProvider");
});

test("keeps toast notifications clear of the workspace rail", () => {
  const styles = readFileSync("src/app/globals.css", "utf8");

  expect(styles).toContain("left: calc(4rem + 24px);");
  expect(styles).toContain("width: min(380px, calc(100vw - 112px));");
  expect(styles).toContain("min-width: 0;");
});

test("uses the config drawer as the incomplete initialization notice", () => {
  const page = readFileSync("src/app/page.tsx", "utf8");

  expect(page).not.toContain("检测到尚未完成初始化");
  expect(page).toContain("setShowConfigModal(true)");
});
