import React, { useCallback, useMemo, useRef, useEffect } from 'react';
import type { ResearchReport, ResearchPlan } from '@/stores/useResearchStore';
import { MarkdownRenderer } from '@/components/Chat/MarkdownRenderer';
import { Download, FileText, ChevronDown, ChevronRight, ListChecks } from 'lucide-react';

/**
 * Post-process rendered markdown to style citation markers `[N]` as superscript.
 * Avoids needing rehype-raw or HTML in markdown source.
 */
function styleCitations(container: HTMLElement) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const nodesToReplace: { node: Text; fragments: Node[] }[] = [];

  while (walker.nextNode()) {
    const textNode = walker.currentNode as Text;
    const text = textNode.textContent || '';
    if (!/\[\d+\]/.test(text)) continue;

    const fragments: Node[] = [];
    let lastIdx = 0;
    const re = /\[(\d+)\]/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > lastIdx) {
        fragments.push(document.createTextNode(text.slice(lastIdx, m.index)));
      }
      const sup = document.createElement('sup');
      sup.className = 'text-[10px] text-purple-400 cursor-default';
      sup.textContent = `[${m[1]}]`;
      fragments.push(sup);
      lastIdx = m.index + m[0].length;
    }
    if (lastIdx < text.length) {
      fragments.push(document.createTextNode(text.slice(lastIdx)));
    }
    if (fragments.length > 0) {
      nodesToReplace.push({ node: textNode, fragments });
    }
  }

  for (const { node, fragments } of nodesToReplace) {
    const parent = node.parentNode;
    if (!parent) continue;
    for (const frag of fragments) parent.insertBefore(frag, node);
    parent.removeChild(node);
  }
}

interface Props {
  report: ResearchReport;
  plan?: ResearchPlan | null;
}

/**
 * Map raw `[ref_xxx]` citation tokens to sequential superscript `<sup>[1]</sup>`
 * and build a reference ordering that matches the numbering.
 */
function buildNumberedMarkdown(
  sections: ResearchReport['sections'],
  references: ResearchReport['references'],
): { markdown: string; orderedRefs: ResearchReport['references'] } {
  const refIdToIndex = new Map<string, number>();
  const ordered: ResearchReport['references'] = [];

  const refLookup = new Map<string, ResearchReport['references'][0]>();
  for (const ref of references) {
    if (ref.reference_id) refLookup.set(ref.reference_id, ref);
  }

  const assignIndex = (refId: string): number => {
    const existing = refIdToIndex.get(refId);
    if (existing !== undefined) return existing;
    const idx = ordered.length + 1;
    refIdToIndex.set(refId, idx);
    const ref = refLookup.get(refId);
    ordered.push(ref ?? { url: '', title: refId, snippet: '', reliability: 0 });
    return idx;
  };

  const replaceCitations = (text: string): string => {
    return text.replace(/\[ref_([a-zA-Z0-9]+)\]/g, (_match, id) => {
      const num = assignIndex(`ref_${id}`);
      return `[${num}]`;
    });
  };

  const parts: string[] = [];
  for (const section of sections) {
    parts.push(`## ${section.title}\n`);
    parts.push(replaceCitations(section.content));
    parts.push('');
  }

  const usedRefs = ordered.length > 0 ? ordered : references;

  return { markdown: parts.join('\n'), orderedRefs: usedRefs };
}

const ReportBody: React.FC<{ markdown: string }> = ({ markdown }) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) styleCitations(ref.current);
  }, [markdown]);

  return (
    <div ref={ref} className="rounded-xl border border-gray-700/50 bg-gray-800/30 p-6">
      <div className="prose prose-invert prose-sm max-w-none">
        <MarkdownRenderer content={markdown} />
      </div>
    </div>
  );
};

export const ReportViewer: React.FC<Props> = ({ report, plan }) => {
  const [showExportMenu, setShowExportMenu] = React.useState(false);
  const [showPlan, setShowPlan] = React.useState(false);

  const { markdown: bodyMarkdown, orderedRefs } = useMemo(
    () => buildNumberedMarkdown(report.sections, report.references),
    [report.sections, report.references],
  );

  const fullMarkdown = useMemo(() => {
    return `# ${report.title}\n\n${bodyMarkdown}`;
  }, [report.title, bodyMarkdown]);

  const exportMarkdown = useMemo(() => {
    const parts = [fullMarkdown];
    if (orderedRefs.length > 0) {
      parts.push('\n## References\n');
      orderedRefs.forEach((ref, i) => {
        parts.push(`${i + 1}. [${ref.title || ref.url}](${ref.url})${ref.snippet ? ` — ${ref.snippet}` : ''}`);
      });
    }
    return parts.join('\n');
  }, [fullMarkdown, orderedRefs]);

  const handleExportMd = useCallback(() => {
    const blob = new Blob([exportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${report.title.replace(/[^a-zA-Z0-9\u4e00-\u9fff]/g, '_').slice(0, 60)}.md`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
  }, [exportMarkdown, report.title]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-gray-300">Research Report</h3>
          {report.quality_score && (
            <div className="flex gap-3 mt-1">
              <span className="text-xs text-gray-500">
                RACE: <span className="text-purple-400">{report.quality_score.race_overall.toFixed(1)}</span>
              </span>
              <span className="text-xs text-gray-500">
                FACT: <span className="text-purple-400">{report.quality_score.fact_overall.toFixed(1)}</span>
              </span>
            </div>
          )}
        </div>

        {/* Export dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setShowExportMenu(!showExportMenu)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 border border-gray-700 hover:border-gray-600 rounded-lg transition-colors"
          >
            <Download size={14} />
            Export
            <ChevronDown size={12} />
          </button>
          {showExportMenu && (
            <div className="absolute right-0 mt-1 w-48 rounded-lg bg-gray-800 border border-gray-700 shadow-xl z-10 py-1">
              <button
                type="button"
                onClick={handleExportMd}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-gray-700 transition-colors text-left"
              >
                <FileText size={14} />
                Markdown (.md)
              </button>
              <div className="px-3 py-2 text-xs text-gray-600 cursor-default">
                PDF — Coming in Phase 4
              </div>
              <div className="px-3 py-2 text-xs text-gray-600 cursor-default">
                PPTX — Coming in Phase 4
              </div>
              <div className="px-3 py-2 text-xs text-gray-600 cursor-default">
                DOCX — Coming in Phase 4
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sub-question decomposition review */}
      {plan && plan.sub_questions.length > 0 && (
        <div className="rounded-xl border border-gray-700/50 bg-gray-800/20 overflow-hidden">
          <button
            type="button"
            onClick={() => setShowPlan(!showPlan)}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-left hover:bg-gray-800/40 transition-colors"
          >
            <ListChecks size={14} className="text-purple-400 shrink-0" />
            <span className="text-xs font-medium text-gray-400 flex-1">
              Research Plan ({plan.sub_questions.length} sub-questions)
            </span>
            {showPlan
              ? <ChevronDown size={14} className="text-gray-500" />
              : <ChevronRight size={14} className="text-gray-500" />
            }
          </button>
          {showPlan && (
            <div className="px-4 pb-3 space-y-1.5 border-t border-gray-700/30">
              {plan.sub_questions
                .sort((a, b) => b.priority - a.priority)
                .map((sq, i) => (
                  <div key={sq.question_id} className="flex items-start gap-2 py-1.5">
                    <span className="text-xs text-gray-600 pt-0.5 shrink-0 w-5 text-right">{i + 1}.</span>
                    <span className="text-xs text-gray-300">{sq.question}</span>
                    <span className="ml-auto text-[10px] text-gray-600 shrink-0">P{sq.priority}</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}

      {/* Report body */}
      <ReportBody markdown={fullMarkdown} />

      {/* Sources panel — single source of truth for references */}
      {orderedRefs.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
            References ({orderedRefs.length})
          </h4>
          <div className="space-y-1.5">
            {orderedRefs.map((ref, i) => (
              <a
                key={`ref-${i}`}
                href={ref.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-2 px-3 py-2 rounded-lg bg-gray-800/30 border border-gray-700/30 hover:border-gray-600 transition-colors group"
              >
                <span className="text-xs text-gray-600 pt-0.5 shrink-0 w-5 text-right">[{i + 1}]</span>
                <div className="min-w-0">
                  <div className="text-xs text-gray-300 group-hover:text-purple-300 truncate">
                    {ref.title || ref.url}
                  </div>
                  {ref.snippet && (
                    <div className="text-xs text-gray-600 mt-0.5 line-clamp-2">{ref.snippet}</div>
                  )}
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
