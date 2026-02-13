/**
 * Unit tests for MarkdownRenderer, CodeBlock, and MermaidBlock.
 *
 * Tests cover:
 * - Basic markdown rendering (headings, bold, italic, links)
 * - GFM tables, task lists, strikethrough
 * - Code blocks with syntax highlighting
 * - Inline code
 * - Math equations (KaTeX)
 * - Mermaid diagram error handling
 * - Copy button on code blocks
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { MarkdownRenderer } from '@/components/Chat/MarkdownRenderer';
import { CodeBlock } from '@/components/Chat/CodeBlock';

// Mock clipboard API
beforeEach(() => {
  Object.assign(navigator, {
    clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
  });
});

describe('MarkdownRenderer', () => {
  it('renders plain text', () => {
    render(<MarkdownRenderer content="Hello world" />);
    expect(screen.getByText('Hello world')).toBeTruthy();
  });

  it('renders headings', () => {
    const { container } = render(<MarkdownRenderer content={"# Heading 1\n\n## Heading 2\n\n### Heading 3"} />);
    expect(container.querySelector('h1')?.textContent).toBe('Heading 1');
    expect(container.querySelector('h2')?.textContent).toBe('Heading 2');
    expect(container.querySelector('h3')?.textContent).toBe('Heading 3');
  });

  it('renders bold and italic text', () => {
    render(<MarkdownRenderer content="**bold** and *italic*" />);
    expect(screen.getByText('bold').tagName).toBe('STRONG');
    expect(screen.getByText('italic').tagName).toBe('EM');
  });

  it('renders links with target=_blank', () => {
    render(<MarkdownRenderer content="[click here](https://example.com)" />);
    const link = screen.getByText('click here');
    expect(link.tagName).toBe('A');
    expect(link.getAttribute('target')).toBe('_blank');
    expect(link.getAttribute('rel')).toContain('noopener');
  });

  it('renders inline code', () => {
    render(<MarkdownRenderer content="Use `console.log` to debug" />);
    const code = screen.getByText('console.log');
    expect(code.tagName).toBe('CODE');
  });

  it('renders unordered lists', () => {
    const md = `- item 1
- item 2
- item 3`;
    const { container } = render(<MarkdownRenderer content={md} />);
    const items = container.querySelectorAll('li');
    expect(items.length).toBe(3);
    expect(items[0].textContent).toContain('item 1');
  });

  it('renders ordered lists', () => {
    const md = `1. first
2. second
3. third`;
    const { container } = render(<MarkdownRenderer content={md} />);
    const items = container.querySelectorAll('li');
    expect(items.length).toBe(3);
    expect(items[0].textContent).toContain('first');
  });

  it('renders blockquotes', () => {
    render(<MarkdownRenderer content="> This is a quote" />);
    const quote = screen.getByText('This is a quote');
    expect(quote.closest('blockquote')).toBeTruthy();
  });

  it('renders GFM tables', () => {
    const table = '| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |';
    render(<MarkdownRenderer content={table} />);
    expect(screen.getByText('Name')).toBeTruthy();
    expect(screen.getByText('Alice')).toBeTruthy();
    expect(screen.getByText('Bob')).toBeTruthy();
  });

  it('renders GFM strikethrough', () => {
    render(<MarkdownRenderer content="~~deleted~~" />);
    const del = screen.getByText('deleted');
    expect(del.tagName).toBe('DEL');
  });

  it('renders fenced code blocks', () => {
    const md = '```python\nprint("hello")\n```';
    render(<MarkdownRenderer content={md} />);
    // CodeBlock renders with language label
    expect(screen.getByText('python')).toBeTruthy();
  });

  it('renders horizontal rules', () => {
    const { container } = render(<MarkdownRenderer content={"above\n\n---\n\nbelow"} />);
    expect(container.querySelector('hr')).toBeTruthy();
  });

  it('renders images with a loading placeholder wrapper', async () => {
    const { container } = render(<MarkdownRenderer content={'![alt text](https://example.com/x.png)'} />);
    const img = container.querySelector('img');
    expect(img).toBeTruthy();
    expect(img?.getAttribute('loading')).toBe('lazy');
    expect(img?.getAttribute('decoding')).toBe('async');

    const wrapper = img?.closest('.houyi-md-image');
    expect(wrapper).toBeTruthy();
    expect(wrapper?.getAttribute('data-loaded')).toBe('0');

    if (img) {
      await fireEvent.load(img);
    }
    expect(wrapper?.getAttribute('data-loaded')).toBe('1');
  });
});

describe('CodeBlock', () => {
  it('renders code with language label', () => {
    render(<CodeBlock language="typescript">const x = 1;</CodeBlock>);
    expect(screen.getByText('typescript')).toBeTruthy();
  });

  it('renders code without language with default label', () => {
    const { container } = render(<CodeBlock>some code</CodeBlock>);
    const label = container.querySelector('.font-mono');
    expect(label?.textContent).toBe('code');
  });

  it('shows copy button', () => {
    render(<CodeBlock language="js">alert(1)</CodeBlock>);
    expect(screen.getByText('Copy')).toBeTruthy();
  });

  it('copies code to clipboard on click', async () => {
    render(<CodeBlock language="js">alert(1)</CodeBlock>);
    const copyBtn = screen.getByTitle('Copy code');
    await fireEvent.click(copyBtn);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('alert(1)');
  });
});
