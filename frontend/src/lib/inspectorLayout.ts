export type InspectorPanelMode = "docked" | "overlay" | "hidden";

export const COMPACT_LAYOUT_QUERY = "(max-width: 767px)";

export function normalizeInspectorPanelMode(
  mode: InspectorPanelMode,
  isCompactLayout: boolean,
): InspectorPanelMode {
  return isCompactLayout && mode === "docked" ? "hidden" : mode;
}

export function resolveInspectorOpenMode(
  isCompactLayout: boolean,
): InspectorPanelMode {
  return isCompactLayout ? "overlay" : "docked";
}
