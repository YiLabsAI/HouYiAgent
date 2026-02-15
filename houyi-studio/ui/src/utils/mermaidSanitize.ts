/**
 * Mermaid code sanitization utilities.
 *
 * Extracted from MermaidBlock so they can be unit-tested independently.
 */

/**
 * Normalize full-width CJK punctuation to ASCII equivalents.
 *
 * Mermaid's parser only understands half-width (ASCII) structural characters.
 * LLMs generating diagrams in a CJK locale frequently emit full-width
 * parentheses, brackets, braces, angle brackets, quotes, and common
 * punctuation — all of which break the parser.
 */
export function normalizeFullWidthPunctuation(code: string): string {
  const map: Record<string, string> = {
    '\uff08': '(',   // （
    '\uff09': ')',   // ）
    '\uff3b': '[',   // ［
    '\uff3d': ']',   // ］
    '\uff5b': '{',   // ｛
    '\uff5d': '}',   // ｝
    '\uff1c': '<',   // ＜
    '\uff1e': '>',   // ＞
    '\uff02': '"',   // ＂
    '\uff07': "'",   // ＇
    '\uff0c': ',',   // ，
    '\uff1a': ':',   // ：
    '\uff1b': ';',   // ；
    '\uff5c': '|',   // ｜
    '\u3010': '[',   // 【
    '\u3011': ']',   // 】
  };
  const re = new RegExp(`[${Object.keys(map).join('')}]`, 'g');
  return code.replace(re, (ch) => map[ch] ?? ch);
}

/**
 * Multi-pass sanitization for Mermaid code that fails to parse.
 *
 * Handles:
 *   1. Full-width → half-width punctuation normalization
 *   2. Subgraph titles containing parentheses  →  quoted titles
 *   3. Node labels containing parentheses      →  quoted labels
 *   4. Edge labels containing dots/CJK/parens  →  quoted labels
 */
export function sanitizeMermaidCode(code: string): string {
  // Phase 0: full-width → half-width normalization
  let safe = normalizeFullWidthPunctuation(code);

  // Phase 1: Fix subgraph titles containing parentheses.
  //   `subgraph Data Plane (Workers)` → `subgraph s_0 ["Data Plane (Workers)"]`
  //
  //   Mermaid treats bare parentheses in a subgraph line as node-shape syntax,
  //   causing parse errors.  We generate a synthetic ID and wrap the original
  //   title in brackets + quotes.
  let subgraphCounter = 0;
  safe = safe.replace(
    /^(\s*)subgraph\s+(.+)$/gm,
    (_match, indent: string, title: string) => {
      const trimmed = title.trim();
      // Already has an explicit id + bracket title  e.g. `subgraph myId ["title"]`
      if (/^\S+\s+\[/.test(trimmed)) return _match;
      // Contains problematic characters (parentheses, brackets) → quote it
      if (/[()[\]{}]/.test(trimmed)) {
        const id = `sg_${subgraphCounter++}`;
        return `${indent}subgraph ${id} ["${trimmed}"]`;
      }
      return _match;
    },
  );

  // Phase 2: Quote node labels containing parentheses:
  //    [text (with parens)] → ["text (with parens)"]
  safe = safe.replace(
    /\[([^\]"]*\([^\]]*)](?!\])/g,
    (_match, inner) => `["${inner}"]`,
  );

  // Phase 3: Quote edge labels containing dots, Chinese chars, or parentheses:
  //    -- label with dots --> → -- "label with dots" -->
  safe = safe.replace(
    /--\s+([^"\-\n][^\-\n]*?)\s+-->/g,
    (_match, label) => {
      if (/[.()\u4e00-\u9fff\uff08\uff09]/.test(label)) {
        return `-- "${label.trim()}" -->`;
      }
      return _match;
    },
  );

  return safe;
}
