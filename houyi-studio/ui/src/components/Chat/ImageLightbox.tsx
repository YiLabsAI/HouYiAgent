/**
 * ImageLightbox: full-screen overlay with zoom (scroll wheel) and pan (drag).
 *
 * Reuses the same interaction pattern as MermaidLightbox:
 * - Scroll wheel zooms in/out (0.25x – 5x), centered on cursor position
 * - Click-and-drag pans the image
 * - Double-click resets zoom to fit
 * - Escape or click backdrop to close
 */
import React from 'react';

interface ImageLightboxProps {
  src: string;
  alt: string;
  onClose: () => void;
}

export const ImageLightbox: React.FC<ImageLightboxProps> = ({ src, alt, onClose }) => {
  const overlayRef = React.useRef<HTMLDivElement>(null);
  const [scale, setScale] = React.useState(1);
  const [translate, setTranslate] = React.useState({ x: 0, y: 0 });
  const isDragging = React.useRef(false);
  const dragStart = React.useRef({ x: 0, y: 0 });
  const translateStart = React.useRef({ x: 0, y: 0 });

  React.useEffect(() => {
    overlayRef.current?.focus();
  }, []);

  // Wheel zoom centered on cursor
  const handleWheel = React.useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const container = e.currentTarget as HTMLElement;
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
      aria-label="Enlarged image"
    >
      <div
        className="w-[90vw] h-[85vh] overflow-hidden bg-gray-900 rounded-xl border border-gray-700 shadow-2xl relative flex items-center justify-center"
        style={{ cursor: isDragging.current ? 'grabbing' : 'grab' }}
        onClick={(e) => e.stopPropagation()}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onDoubleClick={handleDoubleClick}
      >
        <img
          src={src}
          alt={alt}
          className="max-w-full max-h-full object-contain select-none"
          draggable={false}
          style={{
            transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
            transformOrigin: '0 0',
            transition: isDragging.current ? 'none' : 'transform 0.1s ease-out',
          }}
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
