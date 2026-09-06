/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Палитра через CSS-переменные: тёмная тема меняет только переменные.
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        raised: "rgb(var(--raised) / <alpha-value>)",
        line: "rgb(var(--line) / <alpha-value>)",
        ink: "rgb(var(--ink) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        faint: "rgb(var(--faint) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-soft": "rgb(var(--accent-soft) / <alpha-value>)",
        // Шкала риска: не светофор, а градиент от нейтрального к тёплому
        "risk-low": "rgb(var(--risk-low) / <alpha-value>)",
        "risk-mid": "rgb(var(--risk-mid) / <alpha-value>)",
        "risk-high": "rgb(var(--risk-high) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system",
               "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: { xl: "0.75rem", "2xl": "1rem" },
      maxWidth: { content: "78rem" },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "slide-in": {
          from: { transform: "translateX(1.5rem)", opacity: "0" },
          to: { transform: "translateX(0)", opacity: "1" },
        },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        "fade-in": "fade-in .18s ease-out",
        "slide-in": "slide-in .2s cubic-bezier(.22,1,.36,1)",
      },
    },
  },
  plugins: [],
};
