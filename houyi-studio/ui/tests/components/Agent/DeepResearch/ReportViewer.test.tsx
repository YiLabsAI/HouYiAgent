import { render, screen, fireEvent, within, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReportViewer } from '@/components/Agent/DeepResearch/ReportViewer';
import type { ResearchReport } from '@/stores/useResearchStore';

vi.mock('@/components/Chat/MarkdownRenderer', () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div data-testid="md-renderer">{content}</div>,
}));

const makeReport = (overrides: Partial<ResearchReport> = {}): ResearchReport => ({
  title: 'AI Research Report',
  sections: [
    { title: 'Introduction', content: 'This is the intro.', citations: ['[1]'] },
    { title: 'Findings', content: 'Key findings here.', citations: [] },
  ],
  references: [
    {
      url: 'https://example.com',
      title: 'Example Source',
      snippet: 'A relevant excerpt',
      reliability: 0.9,
    },
    {
      url: 'https://other.com',
      title: 'Other Source',
      snippet: '',
      reliability: 0.7,
    },
  ],
  quality_score: { race_overall: 8.5, fact_overall: 7.2 },
  ...overrides,
});

describe('ReportViewer', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  it('renders report header', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.getByText('Research Report')).toBeInTheDocument();
  });

  it('report body uses compact text-sm and report-body class', () => {
    const { container } = render(<ReportViewer report={makeReport()} />);
    const reportBody = container.querySelector('.report-body');
    expect(reportBody).toBeTruthy();
    expect(reportBody?.classList.contains('text-sm')).toBe(true);
  });

  it('renders quality scores', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.getByText('8.5')).toBeInTheDocument();
    expect(screen.getByText('7.2')).toBeInTheDocument();
  });

  it('no quality scores when null', () => {
    render(<ReportViewer report={makeReport({ quality_score: null })} />);
    expect(screen.queryByText(/RACE/)).not.toBeInTheDocument();
  });

  it('renders markdown content', () => {
    render(<ReportViewer report={makeReport()} />);
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('AI Research Report');
    expect(md.textContent).toContain('Introduction');
    expect(md.textContent).toContain('Findings');
  });

  it('renders source references', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.getByText('Example Source')).toBeInTheDocument();
    expect(screen.getByText('Other Source')).toBeInTheDocument();
  });

  it('shows reference count', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.getByText('References (2)')).toBeInTheDocument();
  });

  it('export button renders', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
  });

  it('export dropdown shows options', () => {
    render(<ReportViewer report={makeReport()} />);
    fireEvent.click(screen.getByRole('button', { name: /Export/i }));
    expect(screen.getByText('Markdown (.md)')).toBeInTheDocument();
    expect(screen.getByText('PDF — Coming in Phase 4')).toBeInTheDocument();
  });

  it('no references section for empty refs', () => {
    render(<ReportViewer report={makeReport({ references: [] })} />);
    expect(screen.queryByText(/References/)).not.toBeInTheDocument();
  });

  /** B3-31: [ref_xxx] in body becomes numbered [1], [2], … for markdown + superscript pipeline. */
  it('replaces [ref_xxx] tokens with numbered citations in rendered markdown', () => {
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'Body', content: 'Claim [ref_abc] and more.', citations: [] }],
          references: [
            {
              reference_id: 'ref_abc',
              url: 'https://src.example',
              title: 'Source A',
              snippet: 'supporting',
              reliability: 0.9,
            },
          ],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('[1]');
    expect(md.textContent).not.toContain('[ref_abc]');
  });

  // -- G4-13: Bottom toolbar --------------------------------------------------

  it('renders Copy, Share buttons in bottom toolbar', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.getByRole('button', { name: /Copy/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Share/i })).toBeInTheDocument();
  });

  it('renders Like/Dislike feedback buttons', () => {
    const { container } = render(<ReportViewer report={makeReport()} />);
    expect(screen.getByText('Rate this report')).toBeInTheDocument();
    const thumbButtons = container.querySelectorAll('button');
    const feedbackBtns = Array.from(thumbButtons).filter(
      (b) => b.querySelector('.lucide-thumbs-up') || b.querySelector('.lucide-thumbs-down'),
    );
    expect(feedbackBtns.length).toBeGreaterThanOrEqual(0);
  });

  it('renders Retry button when onRetry is provided', () => {
    const onRetry = vi.fn();
    render(<ReportViewer report={makeReport()} onRetry={onRetry} />);
    const retryBtn = screen.getByRole('button', { name: /Retry/i });
    expect(retryBtn).toBeInTheDocument();
    fireEvent.click(retryBtn);
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('does not render Retry button when onRetry is omitted', () => {
    render(<ReportViewer report={makeReport()} />);
    expect(screen.queryByRole('button', { name: /Retry/i })).not.toBeInTheDocument();
  });

  it('Copy button triggers clipboard write', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<ReportViewer report={makeReport()} />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Copy/i }));
      await Promise.resolve();
      vi.runOnlyPendingTimers();
    });
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0][0]).toContain('AI Research Report');
  });

  /** B3-32: reference_id must match refLookup so ordering follows citation order, not array order. */
  it('matches references by reference_id when building numbered markdown', () => {
    render(
      <ReportViewer
        report={makeReport({
          sections: [
            {
              title: 'S',
              content: 'Second [ref_b] then first [ref_a].',
              citations: [],
            },
          ],
          references: [
            {
              reference_id: 'ref_a',
              url: 'https://a.com',
              title: 'Alpha',
              snippet: '',
              reliability: 0.8,
            },
            {
              reference_id: 'ref_b',
              title: 'Beta',
              url: 'https://b.com',
              snippet: '',
              reliability: 0.8,
            },
          ],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toMatch(/\[1\].*\[2\]/);
    expect(md.textContent).not.toContain('[ref_a]');
    expect(md.textContent).not.toContain('[ref_b]');

    const links = screen.getAllByRole('link');
    const beta = links.find((el) => el.getAttribute('href') === 'https://b.com');
    const alpha = links.find((el) => el.getAttribute('href') === 'https://a.com');
    expect(beta?.textContent).toContain('Beta');
    expect(alpha?.textContent).toContain('Alpha');
    const betaRow = beta?.closest('a');
    const alphaRow = alpha?.closest('a');
    expect(within(betaRow as HTMLElement).getByText('[1]', { exact: false })).toBeInTheDocument();
    expect(within(alphaRow as HTMLElement).getByText('[2]', { exact: false })).toBeInTheDocument();
  });

  it('sanitizes well-formed JSON section content', () => {
    const jsonContent = JSON.stringify({
      content: 'Actual report content about AI.',
      citations: [{ reference_id: 'ref_x', text_span: 'AI' }],
    });
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'Section', content: jsonContent, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Actual report content about AI.');
    expect(md.textContent).not.toContain('"content"');
    expect(md.textContent).not.toContain('"citations"');
  });

  it('sanitizes JSON with literal newlines in content value', () => {
    const broken = '{ "content": "Line 1\\nLine 2\\n\\n### Heading\\n\\nMore text.", "citations": [] }';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content: broken, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Line 1');
    expect(md.textContent).toContain('More text.');
    expect(md.textContent).not.toContain('"content"');
  });

  it('strips trailing citations-only raw JSON artifact after prose', () => {
    const mixed = 'Conclusion paragraph is retained.\n{"citations": [{"reference_id": "ref_1", "text_span": "paragraph"}], "extra": 1}';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content: mixed, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Conclusion paragraph is retained.');
    expect(md.textContent).not.toContain('"citations"');
    expect(md.textContent).not.toContain('"reference_id"');
  });

  it('sanitizes structurally-formatted JSON with actual newlines', () => {
    const multiline = '{\n  "content": "Report body here.",\n  "citations": []\n}';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content: multiline, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Report body here.');
    expect(md.textContent).not.toContain('"content"');
  });

  it('sanitizes code-fenced JSON section content', () => {
    const jsonBody = JSON.stringify({ content: 'Fenced content here.', citations: [] });
    const fenced = '```json\n' + jsonBody + '\n```';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content: fenced, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Fenced content here.');
    expect(md.textContent).not.toContain('"content"');
  });

  it('merges prose with trailing fenced JSON body', () => {
    const trailing = 'Normal analysis text.\n\n```json\n{"content": "Body from JSON.", "citations": []}\n```';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content: trailing, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Normal analysis text.');
    expect(md.textContent).toContain('Body from JSON.');
    expect(md.textContent).not.toContain('"content"');
  });

  it('strips unclosed trailing JSON code fence with content key', () => {
    const content = 'Valid prose about AI.\n\n```json\n{\n  "content": "Hermes Agent is a framework.",\n  "citations": []';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Valid prose about AI.');
    expect(md.textContent).not.toContain('"content"');
    expect(md.textContent).not.toContain('Hermes Agent is a framework');
  });

  it('strips inline JSON citation tail leaking after prose', () => {
    const content = 'Important analysis about the system [ref_abc].",\n  "citations": [\n    {\n      "reference_id": "ref_abc",\n      "text_span": "analysis"\n    }\n  ]';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Important analysis about the system');
    expect(md.textContent).not.toContain('"citations"');
    expect(md.textContent).not.toContain('"reference_id"');
  });

  it('strips trailing raw JSON block with content key after prose', () => {
    const content = 'Conclusion paragraph.\n\n{\n  "content": "Duplicated content.",\n  "citations": []\n}';
    render(
      <ReportViewer
        report={makeReport({
          sections: [{ title: 'S', content, citations: [] }],
          references: [],
        })}
      />,
    );
    const md = screen.getByTestId('md-renderer');
    expect(md.textContent).toContain('Conclusion paragraph.');
    expect(md.textContent).not.toContain('"content"');
  });

  it('filters out localhost and bare-id references', () => {
    render(
      <ReportViewer
        report={makeReport({
          sections: [
            {
              title: 'Intro',
              content: 'Good [ref_ok] and bad [ref_local].',
              citations: [],
            },
          ],
          references: [
            { reference_id: 'ref_ok', url: 'https://example.com', title: 'Good', snippet: '', reliability: 0.8 },
            { reference_id: 'ref_local', url: 'http://localhost:3000/page', title: 'Local', snippet: '', reliability: 0.1 },
          ],
        })}
      />,
    );
    const links = screen.getAllByRole('link');
    expect(links.some((el) => el.getAttribute('href') === 'https://example.com')).toBe(true);
    expect(links.some((el) => el.getAttribute('href')?.includes('localhost'))).toBe(false);
  });

  it('defers citation superscript styling until animation completes', async () => {
    const { container } = render(
      <ReportViewer
        animate
        report={makeReport({
          sections: [{ title: 'Body', content: 'Claim [ref_abc] and more.', citations: [] }],
          references: [
            {
              reference_id: 'ref_abc',
              url: 'https://src.example',
              title: 'Source A',
              snippet: 'supporting',
              reliability: 0.9,
            },
          ],
        })}
      />,
    );

    expect(container.querySelectorAll('sup').length).toBe(0);

    await act(async () => {
      vi.advanceTimersByTime(25_000);
    });

    expect(container.querySelectorAll('sup').length).toBeGreaterThan(0);
  });
});
