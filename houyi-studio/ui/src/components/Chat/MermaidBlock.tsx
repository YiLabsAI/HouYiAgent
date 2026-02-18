/**
 * MermaidBlock: lazy-loaded Mermaid diagram renderer.
 *
 * Dynamically imports mermaid library only when a mermaid code block is detected.
 * Uses debounced rendering to avoid flickering during streaming.
 * Shows a gentle loading state while code is still arriving.
 *
 * scroll-safe, flicker-free rendering:
 *   - UNIFIED container (single DOM element) for loading/SVG states.
 *     Avoids DOM unmount/mount during transitions which disrupts scroll
 *     anchoring in the parent ChatTimeline.
 *   - FIXED height on the container (from cache or heuristic) prevents
 *     layout shifts during the loading → SVG swap.  Height is measured
 *     synchronously in useLayoutEffect (before paint) and cached for
 *     subsequent renders.
 *   - Mermaid background matches container bg-gray-950 (#030712) so the
 *     SVG blends seamlessly — no "nested rectangles" even with max-width cap.
 *   - SVG max-width capped at viewBox width (no upscaling), unused space
 *     invisible thanks to matching backgrounds.
 */
import React from 'react';
import { useThemeStore } from '@/stores/useThemeStore';

// Module-level caches — survive re-renders and re-mounts.
const mermaidHeightCache = new Map<string, number>();
const mermaidSvgCache = new Map<string, string>();

function heuristicHeightFromCode(code: string): number {
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
  // This prevents the SVG from upscaling beyond its natural size.
  // The matching Mermaid background (#030712 = bg-gray-950) makes any
  // unused space on the right invisible — no "nested rectangles".
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

// Sanitization logic extracted to utils/mermaidSanitize.ts for testability.
import { normalizeFullWidthPunctuation, sanitizeMermaidCode } from '@/utils/mermaidSanitize';

interface MermaidBlockProps {
  children: string;
}

export const MermaidBlock: React.FC<MermaidBlockProps> = ({ children }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const code = children.trim();
  const theme = useThemeStore((s) => s.theme);
  const cacheKey = React.useMemo(() => `${theme}:${code}`, [theme, code]);

  // Initialize from caches — zero flash on conversation switch.
  const [svg, setSvg] = React.useState<string | null>(() => mermaidSvgCache.get(cacheKey) ?? null);
  const [error, setError] = React.useState<string | null>(null);
  const [rendering, setRendering] = React.useState(false);
  const [expanded, setExpanded] = React.useState(false);
  // Fixed container height: from cache (exact) or heuristic (estimated).
  // Using `height` (not `minHeight`) prevents layout shifts.
  const [containerHeight, setContainerHeight] = React.useState<number>(() =>
    mermaidHeightCache.get(cacheKey) ?? heuristicHeightFromCode(code),
  );
  const renderTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  // Use a ref for cancellation so StrictMode double-invocation doesn't
  // permanently cancel the render (closure var gets stuck as true).
  const cancelledRef = React.useRef(false);

  // When cacheKey changes, update container height from cache/heuristic
  // BEFORE paint — so the container renders at the correct height on the
  // very first frame after a conversation switch.
  React.useLayoutEffect(() => {
    setContainerHeight(mermaidHeightCache.get(cacheKey) ?? heuristicHeightFromCode(code));
  }, [cacheKey, code]);

  // After SVG renders, measure its natural height and update the container
  // height synchronously (before paint).  The user never sees a clipped or
  // overflowing intermediate state.
  React.useLayoutEffect(() => {
    if (!containerRef.current || !svg || rendering) return;
    const svgEl = containerRef.current.querySelector('svg');
    if (!svgEl) return;
    const svgH = svgEl.getBoundingClientRect().height;
    if (Number.isFinite(svgH) && svgH > 0) {
      // Container has p-3 (12px * 2 = 24px padding) + border (1px * 2 = 2px)
      const totalH = Math.min(1200, Math.max(180, Math.ceil(svgH + 26)));
      mermaidHeightCache.set(cacheKey, totalH);
      setContainerHeight(totalH);
    }
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
                // Must match the container bg-gray-950 (#030712) so the
                // SVG background blends seamlessly — no "nested rectangles".
                background: '#030712',
                primaryColor: '#3b82f6',
                primaryTextColor: '#e5e7eb',
                lineColor: '#6b7280',
              },
        });

        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

        // Hidden off-screen container for Mermaid rendering.
        // Mermaid.render() creates temporary DOM elements; without a hidden
        // container they flash visibly in the viewport (first-render flicker).
        const offscreen = document.createElement('div');
        offscreen.style.cssText = 'position:fixed;left:-9999px;top:-9999px;width:900px;visibility:hidden;pointer-events:none;';
        document.body.appendChild(offscreen);

        // Always normalize full-width punctuation before any parsing.
        const normalizedCode = normalizeFullWidthPunctuation(code);

        // Multi-pass rendering: try normalized, then progressively sanitize
        let rendered = '';
        try {
          const parseOk = await mermaid.parse(normalizedCode, { suppressErrors: true });

          if (parseOk) {
            ({ svg: rendered } = await mermaid.render(id, normalizedCode, offscreen));
          } else {
            // Try deeper sanitization (quoting labels, etc.)
            const safeCode = sanitizeMermaidCode(code);
            const safeId = `${id}-safe`;
            try {
              ({ svg: rendered } = await mermaid.render(safeId, safeCode, offscreen));
            } catch {
              // Last resort: try normalized code anyway (parse may be overly strict)
              ({ svg: rendered } = await mermaid.render(`${id}-orig`, normalizedCode, offscreen));
            }
          }
        } finally {
          // Always clean up the off-screen container
          offscreen.remove();
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

  // Error state: separate container (rare transition, different visual style)
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

  // ── Unified container for loading → SVG transitions ──
  // A SINGLE DOM element hosts both the loading spinner and the rendered SVG.
  // Fixed `height` (not minHeight) prevents layout shifts that disrupt the
  // ChatTimeline's scroll-position anchoring.
  return (
    <>
      <div
        ref={containerRef}
        className="my-2 p-3 bg-gray-950 border border-gray-700/50 rounded-md overflow-hidden relative w-full"
        style={{ height: containerHeight }}
      >
        {svg ? (
          <>
            <div dangerouslySetInnerHTML={{ __html: svg }} />
            {/* Transparent click overlay — SVG internals swallow clicks */}
            <div
              className="absolute inset-0 cursor-zoom-in"
              onClick={() => setExpanded(true)}
              title="Click to enlarge"
            />
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center gap-2">
            <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-[11px] text-gray-500">Rendering diagram...</span>
          </div>
        )}
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
