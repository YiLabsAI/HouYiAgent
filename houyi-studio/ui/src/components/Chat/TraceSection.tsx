import React from 'react';

interface TraceSectionProps {
  title: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  testId?: string;
}

export const TraceSection: React.FC<TraceSectionProps> = ({ title, children, actions, className = '', testId }) => (
  <section data-testid={testId} className={`rounded border border-gray-800 bg-gray-900/60 p-3 ${className}`.trim()}>
    <div className="mb-2 flex items-center justify-between gap-2">
      <div className="text-[11px] font-medium uppercase tracking-wide text-gray-300">{title}</div>
      {actions ? <div className="shrink-0">{actions}</div> : null}
    </div>
    {children}
  </section>
);
