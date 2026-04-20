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

// First non-blank line already declares the diagram kind? These are the
// keywords Mermaid treats as a valid diagram type — ordered by frequency
// so the common cases short-circuit the regex engine.
const DIAGRAM_TYPE_KEYWORDS = [
  'sequenceDiagram',
  'flowchart',
  'graph',
  'classDiagram',
  'stateDiagram',
  'stateDiagram-v2',
  'erDiagram',
  'journey',
  'gantt',
  'pie',
  'gitGraph',
  'timeline',
  'mindmap',
  'quadrantChart',
  'requirementDiagram',
  'C4Context',
  'C4Container',
  'C4Component',
  'sankey',
  'xychart-beta',
  'block',
];

const DIAGRAM_TYPE_RE = new RegExp(
  `^\\s*(?:${DIAGRAM_TYPE_KEYWORDS.join('|')})\\b`,
);

/**
 * Prepend an inferred diagram-type header when the first non-blank line
 * of `code` is not a recognised Mermaid diagram kind.
 *
 * LLM writers occasionally emit only the body of a diagram (the opening
 * ```mermaid fence + type keyword were truncated or never written), so
 * mermaid.parse fails with "No diagram type detected". Probe the body
 * for syntax markers unique to one of the common diagram types and
 * prepend the matching keyword so the parser has a chance.
 *
 * Returns the input unchanged when a valid header is already present or
 * the body does not match any known diagram shape.
 */
export function ensureDiagramTypeHeader(code: string): string {
  const trimmed = code.replace(/^\s+/, '');
  if (DIAGRAM_TYPE_RE.test(trimmed)) return code;

  // Sequence diagrams — ``Alice->>Bob`` style arrows plus the common
  // grouping keywords (alt / else / loop / par / opt / note / activate)
  // appearing on their own line. The `end` keyword alone is too generic.
  if (
    /(^|\n)\s*[^\n]*?-[->x]+>[^\n]*/.test(code) &&
    /(^|\n)\s*(?:alt|else|loop|par|opt|note|activate|deactivate|participant)\b/.test(
      code,
    )
  ) {
    return `sequenceDiagram\n${code}`;
  }
  if (/(^|\n)\s*participant\b/.test(code)) {
    return `sequenceDiagram\n${code}`;
  }

  // State diagram — ``[*] -->`` start-state marker. Probed before
  // flowchart because its body also contains long arrows.
  if (/\[\*\]\s*-->/.test(code)) {
    return `stateDiagram-v2\n${code}`;
  }

  // Class diagram — ``ClassA <|-- ClassB`` inheritance syntax. Probed
  // before flowchart for the same reason.
  if (/<\|--|--\|>|\.\.\|>|\*--|o--/.test(code)) {
    return `classDiagram\n${code}`;
  }

  // Flowchart/graph — long arrows (`A --> B`) without the more specific
  // markers above.
  if (/(^|\n)[^\n]*?\s--+>\s/.test(code)) {
    return `flowchart TD\n${code}`;
  }

  return code;
}

/**
 * Multi-pass sanitization for Mermaid code that fails to parse.
 *
 * Handles:
 *   1. Full-width → half-width punctuation normalization
 *   2. Subgraph titles containing parentheses  →  quoted titles
 *   3. Node labels containing parentheses      →  quoted labels
 *   4. Edge labels containing dots/CJK/parens  →  quoted labels
 *   5. Missing diagram-type header            →  inferred keyword prepended
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

  // Phase 4: Prepend an inferred diagram-type header when the body is
  // missing one (e.g. the opening ```mermaid fence + keyword were cut).
  return ensureDiagramTypeHeader(safe);
}
