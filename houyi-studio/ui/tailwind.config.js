import defaultTheme from 'tailwindcss/defaultTheme';
import colors from 'tailwindcss/colors';

// Build a clean color palette without deprecated aliases that trigger warnings.
// We pick only the canonical names; deprecated ones (lightBlue → sky, etc.)
// are excluded to avoid Tailwind v3 deprecation warnings.
const safeColors = /** @type {Record<string, any>} */ ({});
const deprecated = new Set(['lightBlue', 'warmGray', 'trueGray', 'coolGray', 'blueGray']);
for (const key of Object.keys(colors)) {
  if (!deprecated.has(key)) {
    safeColors[key] = /** @type {any} */ (colors)[key];
  }
}

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    // Top-level colors: spread ALL default colors, then REPLACE gray
    // with CSS custom properties. theme.extend.colors cannot override
    // built-in gray — Tailwind merges extend shallowly and the built-in
    // gray (hardcoded RGB) always wins. Defining at the top level is the
    // only way to make bg-gray-800 etc. resolve to var(--gray-800).
    colors: {
      ...safeColors,
      // Override gray with CSS custom properties for runtime theme switching.
      // Tailwind v3 needs color values it can decompose into RGB channels to
      // support opacity modifiers (bg-gray-900/50).  Plain `var(--gray-900)`
      // is opaque to the compiler so it silently falls back to built-in gray.
      // Fix: define CSS vars as space-separated RGB channels (e.g. "24 24 27")
      // and reference them here with rgb() + <alpha-value>.
      gray: {
        50:  'rgb(var(--gray-50)  / <alpha-value>)',
        100: 'rgb(var(--gray-100) / <alpha-value>)',
        200: 'rgb(var(--gray-200) / <alpha-value>)',
        300: 'rgb(var(--gray-300) / <alpha-value>)',
        400: 'rgb(var(--gray-400) / <alpha-value>)',
        500: 'rgb(var(--gray-500) / <alpha-value>)',
        600: 'rgb(var(--gray-600) / <alpha-value>)',
        700: 'rgb(var(--gray-700) / <alpha-value>)',
        800: 'rgb(var(--gray-800) / <alpha-value>)',
        900: 'rgb(var(--gray-900) / <alpha-value>)',
        950: 'rgb(var(--gray-950) / <alpha-value>)',
      },
    },
    extend: {
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
      },
    },
  },
  plugins: [],
}
