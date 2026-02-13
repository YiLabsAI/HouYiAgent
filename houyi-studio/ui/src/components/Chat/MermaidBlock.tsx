/**
 * MermaidBlock: lazy-loaded Mermaid diagram renderer.
 *
 * Dynamically imports mermaid library only when a mermaid code block is detected.
 * Uses debounced rendering to avoid flickering during streaming.
 * Shows a gentle loading state while code is still arriving.
 */
import React from 'react';
import { useThemeStore } from '@/stores/useThemeStore';

const mermaidMinHeightCache = new Map<string, number>();
const mermaidSvgCache = new Map<string, string>();

function heuristicMinHeightFromCode(code: string): number {
  const lines = Math.max(1, code.split('\n').filter((l) => l.trim()).length);
  const estimated = 160 + lines * 18;
  return Math.min(720, Math.max(240, estimated));
}

// Normalize SVG string using pure string ops (NO DOMParser — it drops
// foreignObject HTML content when serialising back via outerHTML).
function normalizeMermaidSvg(svgStr: string): string {
  let s = svgStr;

  // 1. Ensure viewBox exists — extract from width/height attrs if missing.
  if (!/viewBox\s*=/.test(s)) {
    const wm = s.match(/<svg[^>]*?\bwidth\s*=\s*"([^"]*)"/);
    const hm = s.match(/<svg[^>]*?\bheight\s*=\s*"([^"]*)"/);
    const w = parseFloat(wm?.[1] || '') || 800;
    const h = parseFloat(hm?.[1] || '') || 600;
    s = s.replace(/<svg\b/, `<svg viewBox="0 0 ${w} ${h}"`);
  }

  // 2. Read viewBox width for the max-width cap.
  let viewBoxW = 0;
  const vbMatch = s.match(/viewBox\s*=\s*"([^"]*)"/);
  if (vbMatch) {
    const parts = vbMatch[1].split(/[\s,]+/).map(Number);
    if (parts.length >= 3) viewBoxW = parts[2];
  }

  // 3. Remove width / height attributes from <svg> tag only.
  s = s.replace(/(<svg\b[^>]*?)\s+width\s*=\s*"[^"]*"/g, '$1');
  s = s.replace(/(<svg\b[^>]*?)\s+height\s*=\s*"[^"]*"/g, '$1');

  // 4. Set style on <svg>: width:100% + max-width capped at viewBox width.
  const maxW = viewBoxW > 0 ? `min(100%, ${viewBoxW}px)` : '100%';
  const newStyle = `display:block;width:100%;max-width:${maxW};height:auto;`;
  if (/(<svg\b[^>]*?)\bstyle\s*=\s*"/.test(s)) {
    s = s.replace(/(<svg\b[^>]*?\bstyle\s*=\s*")[^"]*"/, `$1${newStyle}"`);
  } else {
    s = s.replace(/<svg\b/, `<svg style="${newStyle}"`);
  }

  // 5. Shrink font-size > 14px → ×0.875 (Mermaid default 16→14, match body text).
  s = s.replace(/font-size:\s*(\d+(?:\.\d+)?)px/g, (_match, size) => {
    const orig = parseFloat(size);
    if (orig <= 14) return _match;
    return `font-size: ${Math.round(orig * 0.875)}px`;
  });

  return s;
}

// Multi-pass sanitization for Mermaid code that fails to parse.
function sanitizeMermaidCode(code: string): string {
  let safe = code;

  // 1. Quote node labels containing parentheses:
  //    [text (with parens)] → ["text (with parens)"]
  safe = safe.replace(
    /\[([^\]"]*\([^\]]*)](?!\])/g,
    (_match, inner) => `["${inner}"]`,
  );

  // 2. Quote edge labels containing dots, Chinese chars, or parentheses:
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

interface MermaidBlockProps {
  children: string;
}

export const MermaidBlock: React.FC<MermaidBlockProps> = ({ children }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const code = children.trim();
  const theme = useThemeStore((s) => s.theme);
  const cacheKey = React.useMemo(() => `${theme}:${code}`, [theme, code]);

  // Initialize from SVG cache — zero flash on conversation switch
  const [svg, setSvg] = React.useState<string | null>(() => mermaidSvgCache.get(cacheKey) ?? null);
  const [error, setError] = React.useState<string | null>(null);
  const [rendering, setRendering] = React.useState(false);
  const [expanded, setExpanded] = React.useState(false);
  const renderTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  // Use a ref for cancellation so StrictMode double-invocation doesn't
  // permanently cancel the render (closure var gets stuck as true).
  const cancelledRef = React.useRef(false);

  const cachedMinHeight = React.useMemo(() => {
    const v = mermaidMinHeightCache.get(cacheKey);
    return typeof v === 'number' ? v : heuristicMinHeightFromCode(code);
  }, [cacheKey, code]);

  // After a successful render, measure the real DOM height and cache it.
  React.useLayoutEffect(() => {
    if (!containerRef.current || !svg || rendering) return;
    const el = containerRef.current;
    const raf = requestAnimationFrame(() => {
      const h = el.getBoundingClientRect().height;
      if (Number.isFinite(h) && h > 0) {
        mermaidMinHeightCache.set(cacheKey, Math.min(1200, Math.max(180, h)));
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [cacheKey, svg, rendering]);

  React.useEffect(() => {
    if (renderTimeoutRef.current) {
      clearTimeout(renderTimeoutRef.current);
    }
    cancelledRef.current = false;

    // If SVG cache has an exact match, use it immediately — no debounce.
    const cached = mermaidSvgCache.get(cacheKey);
    if (cached) {
      setSvg(cached);
      setError(null);
      setRendering(false);
      return;
    }

    // No cache hit — show loading state if we have no previous SVG.
    if (!svg) {
      setRendering(true);
    }

    const isLight = theme === 'light';

    const doRender = async () => {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: isLight ? 'default' : 'dark',
          suppressErrorRendering: true,
          fontSize: 12,
          themeVariables: isLight
            ? {
                darkMode: false,
                background: '#f6f8fa',
                primaryColor: '#1e66f5',
                primaryTextColor: '#4c4f69',
                lineColor: '#8c8fa3',
              }
            : {
                darkMode: true,
                background: '#1a1a2e',
                primaryColor: '#3b82f6',
                primaryTextColor: '#e5e7eb',
                lineColor: '#6b7280',
              },
        });

        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

        // Multi-pass rendering: try original, then progressively sanitize
        let rendered = '';
        const parseOk = await mermaid.parse(code, { suppressErrors: true });

        if (parseOk) {
          ({ svg: rendered } = await mermaid.render(id, code));
        } else {
          // Try sanitized code
          const safeCode = sanitizeMermaidCode(code);
          const safeId = `${id}-safe`;
          try {
            ({ svg: rendered } = await mermaid.render(safeId, safeCode));
          } catch {
            // Last resort: try original code anyway (parse may be overly strict)
            ({ svg: rendered } = await mermaid.render(`${id}-orig`, code));
          }
        }
        if (!cancelledRef.current) {
          const normalized = normalizeMermaidSvg(rendered);
          mermaidSvgCache.set(cacheKey, normalized);
          setSvg(normalized);
          setError(null);
          setRendering(false);
        }
      } catch (e: any) {
        console.error('[MermaidBlock] render failed:', e);
        if (!cancelledRef.current) {
          setError(e.message || 'Failed to render Mermaid diagram');
          setSvg(null);
          setRendering(false);
        }
      }
    };

    // Debounce: wait 500ms after last code change before rendering
    renderTimeoutRef.current = setTimeout(doRender, 500);

    return () => {
      cancelledRef.current = true;
      if (renderTimeoutRef.current) {
        clearTimeout(renderTimeoutRef.current);
      }
    };
  }, [code, theme, cacheKey]);

  // Loading / streaming placeholder — show only when no previous SVG exists.
  // If we already rendered a diagram, keep showing the old one while re-rendering.
  if (rendering && !svg) {
    return (
      <div
        className="my-2 p-3 bg-gray-950 border border-gray-700/50 rounded-md overflow-hidden relative w-full"
        style={{ minHeight: cachedMinHeight }}
      >
        <div className="w-full flex items-center justify-center gap-2">
          <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-[11px] text-gray-500">Rendering diagram...</span>
        </div>
      </div>
    );
  }

  // If re-rendering with existing SVG, keep showing the old one at full
  // opacity to avoid a flash (e.g. during theme switch).  The new SVG will
  // atomically replace it once doRender completes.
  if (rendering && svg) {
    return (
      <div
        ref={containerRef}
        className="my-2 p-3 bg-gray-950 border border-gray-700/50 rounded-md overflow-hidden relative w-full"
        style={{ minHeight: cachedMinHeight }}
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    );
  }

  if (error && !rendering) {
    return (
      <div className="my-2 rounded-md overflow-hidden bg-gray-950 border border-amber-800/30">
        <div className="px-3 py-1.5 bg-amber-900/20 border-b border-amber-800/30 flex items-center gap-1.5">
          <span className="text-[10px] text-amber-400">⚠ Diagram rendering failed</span>
        </div>
        <pre className="overflow-x-auto p-3 text-[12px] leading-relaxed text-gray-400 max-h-[200px]">
          <code>{code}</code>
        </pre>
      </div>
    );
  }

  return (
    <>
      <div
        ref={containerRef}
        className="my-2 p-3 bg-gray-950 border border-gray-700/50 rounded-md overflow-hidden cursor-zoom-in relative w-full"
        title="Click to enlarge"
      >
        <div dangerouslySetInnerHTML={{ __html: svg || '' }} />
        {/* Transparent click overlay — SVG internals swallow clicks */}
        <div
          className="absolute inset-0"
          onClick={() => setExpanded(true)}
        />
      </div>

      {/* Lightbox overlay */}
      {expanded && svg && (
        <MermaidLightbox svg={svg} onClose={() => setExpanded(false)} />
      )}
    </>
  );
};

/**
 * MermaidLightbox: full-screen overlay with zoom (scroll wheel) and pan (drag).
 *
 * - Scroll wheel zooms in/out (0.25x – 5x), centered on cursor position
 * - Click-and-drag pans the diagram
 * - Double-click resets zoom to 1x
 * - Escape or click backdrop to close
 */
const MermaidLightbox: React.FC<{ svg: string; onClose: () => void }> = ({ svg, onClose }) => {
  const contentRef = React.useRef<HTMLDivElement>(null);
  const overlayRef = React.useRef<HTMLDivElement>(null);
  const [scale, setScale] = React.useState(1);
  const [translate, setTranslate] = React.useState({ x: 0, y: 0 });
  const isDragging = React.useRef(false);
  const dragStart = React.useRef({ x: 0, y: 0 });
  const translateStart = React.useRef({ x: 0, y: 0 });

  // Strip inline width/height from SVG and set width=100% so it scales
  React.useEffect(() => {
    if (!contentRef.current) return;
    const svgEl = contentRef.current.querySelector('svg');
    if (svgEl) {
      if (!svgEl.getAttribute('viewBox')) {
        const w = svgEl.getAttribute('width') || svgEl.style.width;
        const h = svgEl.getAttribute('height') || svgEl.style.height;
        const wNum = parseFloat(w || '800');
        const hNum = parseFloat(h || '600');
        svgEl.setAttribute('viewBox', `0 0 ${wNum} ${hNum}`);
      }
      svgEl.removeAttribute('width');
      svgEl.removeAttribute('height');
      svgEl.style.width = '100%';
      svgEl.style.height = 'auto';
    }
    overlayRef.current?.focus();
  }, [svg]);

  // Wheel zoom centered on cursor
  const handleWheel = React.useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const container = contentRef.current?.parentElement;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const cursorX = e.clientX - rect.left;
    const cursorY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(5, Math.max(0.25, scale * zoomFactor));
    const ratio = newScale / scale;

    setTranslate((prev) => ({
      x: cursorX - ratio * (cursorX - prev.x),
      y: cursorY - ratio * (cursorY - prev.y),
    }));
    setScale(newScale);
  }, [scale]);

  // Drag to pan
  const handlePointerDown = React.useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
    translateStart.current = { ...translate };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [translate]);

  const handlePointerMove = React.useCallback((e: React.PointerEvent) => {
    if (!isDragging.current) return;
    setTranslate({
      x: translateStart.current.x + (e.clientX - dragStart.current.x),
      y: translateStart.current.y + (e.clientY - dragStart.current.y),
    });
  }, []);

  const handlePointerUp = React.useCallback(() => {
    isDragging.current = false;
  }, []);

  // Double-click to reset
  const handleDoubleClick = React.useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  const zoomPercent = Math.round(scale * 100);

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center"
      onClick={onClose}
      onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}
      tabIndex={0}
      role="dialog"
      aria-label="Enlarged diagram"
    >
      <div
        className="w-[90vw] h-[85vh] overflow-hidden bg-gray-900 rounded-xl border border-gray-700 shadow-2xl relative"
        style={{ cursor: isDragging.current ? 'grabbing' : 'grab' }}
        onClick={(e) => e.stopPropagation()}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onDoubleClick={handleDoubleClick}
      >
        <div
          ref={contentRef}
          className="flex justify-center items-center w-full h-full"
          style={{
            transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
            transformOrigin: '0 0',
            transition: isDragging.current ? 'none' : 'transform 0.1s ease-out',
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>

      {/* Zoom indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 px-3 py-1 bg-gray-800/90 rounded-full text-[11px] text-gray-400 select-none pointer-events-none">
        {zoomPercent}% · scroll to zoom · drag to pan · double-click to reset
      </div>

      <button
        onClick={onClose}
        className="absolute top-4 right-4 p-2 bg-gray-800 hover:bg-gray-700 rounded-full text-gray-300 hover:text-white transition-colors"
        type="button"
        aria-label="Close"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
  );
};
