import { expect, test } from "@playwright/test";

import {
  normalizeInspectorPanelMode,
  resolveInspectorOpenMode,
} from "../src/lib/inspectorLayout";

test("hides a docked inspector when the viewport becomes compact", () => {
  expect(normalizeInspectorPanelMode("docked", true)).toBe("hidden");
  expect(normalizeInspectorPanelMode("overlay", true)).toBe("overlay");
  expect(normalizeInspectorPanelMode("hidden", true)).toBe("hidden");
});

test("keeps the selected inspector mode on a wide viewport", () => {
  expect(normalizeInspectorPanelMode("docked", false)).toBe("docked");
  expect(normalizeInspectorPanelMode("overlay", false)).toBe("overlay");
});

test("opens the inspector as an overlay only on compact viewports", () => {
  expect(resolveInspectorOpenMode(true)).toBe("overlay");
  expect(resolveInspectorOpenMode(false)).toBe("docked");
});
