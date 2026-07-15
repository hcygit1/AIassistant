import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const eslintConfig = [
  ...nextVitals,
  ...nextTypescript,
  {
    rules: {
      // API and SSE payloads are intentionally open-ended at external boundaries.
      "@typescript-eslint/no-explicit-any": "off",
      // Existing components intentionally hydrate or fetch state from effects.
      "react-hooks/set-state-in-effect": "warn",
    },
  },
];

export default eslintConfig;
