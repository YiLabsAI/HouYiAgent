import React from 'react';
import { createPortal } from 'react-dom';

interface InfoHintProps {
  content: string;
  align?: 'left' | 'right';
  testId?: string;
}

export const InfoHint: React.FC<InfoHintProps> = ({
  content,
  align = 'left',
  testId,
}) => {
  const [open, setOpen] = React.useState(false);
  const buttonRef = React.useRef<HTMLButtonElement | null>(null);
  const [position, setPosition] = React.useState<{ top: number; left: number } | null>(null);

  const updatePosition = React.useCallback(() => {
    if (!buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    const tooltipWidth = 256;
    const gap = 8;
    const left = align === 'right'
      ? Math.max(gap, rect.right - tooltipWidth)
      : Math.min(rect.left, window.innerWidth - tooltipWidth - gap);
    const top = Math.min(rect.bottom + gap, window.innerHeight - gap);
    setPosition({ top, left: Math.max(gap, left) });
  }, [align]);

  React.useEffect(() => {
    if (!open) return;
    updatePosition();
    const handleWindowChange = () => updatePosition();
    window.addEventListener('resize', handleWindowChange);
    window.addEventListener('scroll', handleWindowChange, true);
    return () => {
      window.removeEventListener('resize', handleWindowChange);
      window.removeEventListener('scroll', handleWindowChange, true);
    };
  }, [open, updatePosition]);

  return (
    <span className="relative inline-flex items-center">
      <button
        ref={buttonRef}
        type="button"
        aria-label="More information"
        aria-expanded={open}
        data-testid={testId}
        onMouseEnter={() => {
          updatePosition();
          setOpen(true);
        }}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => {
          updatePosition();
          setOpen(true);
        }}
        onBlur={() => setOpen(false)}
        onClick={() => {
          updatePosition();
          setOpen((current) => !current);
        }}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-gray-600 text-[10px] font-medium leading-none text-gray-400 transition-colors hover:border-gray-500 hover:text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
      >
        i
      </button>
      {open && position && typeof document !== 'undefined'
        ? createPortal(
          <span
            role="tooltip"
            className="fixed z-[80] w-64 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-left text-[11px] leading-4 text-gray-200 shadow-lg"
            style={{ top: `${position.top}px`, left: `${position.left}px` }}
          >
            {content}
          </span>,
          document.body,
        )
        : null}
    </span>
  );
};
