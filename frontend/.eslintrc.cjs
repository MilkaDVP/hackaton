/* Минимальный конфиг с одной по-настоящему важной для нас проверкой:
   react-hooks/rules-of-hooks. Именно её нарушение (useMemo после раннего
   return) уронило страницу /model в рантайме с React error #310. */
module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parser: "@typescript-eslint/parser",
  parserOptions: { ecmaVersion: "latest", sourceType: "module", ecmaFeatures: { jsx: true } },
  plugins: ["@typescript-eslint", "react-hooks"],
  rules: {
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn",
  },
  ignorePatterns: ["dist", "node_modules", "*.config.js", "*.config.ts"],
};
