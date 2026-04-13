/**
 * MarkdownRenderer: renders markdown content with full formatting support.
 *
 * Features:
 * - GFM tables, strikethrough, task lists, autolinks
 * - Syntax-highlighted code blocks (highlight.js)
 * - Math equations (KaTeX via remark-math + rehype-katex)
 * - Mermaid diagrams (lazy-loaded)
 * - Copy button on code blocks
 *
 */
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import katex from 'katex';
import type { Components } from 'react-markdown';
import { CodeBlock } from './CodeBlock';
import { MermaidBlock } from './MermaidBlock';
import 'katex/dist/katex.min.css';

/**
 * LatexBlock: renders a latex/math/tex code block as KaTeX display math.
 * Falls back to showing raw code if KaTeX fails to parse.
 */
const LatexBlock: React.FC<{ children: string }> = ({ children }) => {
  const html = React.useMemo(() => {
    try {
      return katex.renderToString(children, {
        displayMode: true,
        throwOnError: false,
        trust: true,
      });
    } catch {
      return null;
    }
  }, [children]);

  if (!html) {
    return (
      <pre className="my-2 p-3 bg-gray-950 border border-gray-700/50 rounded-md overflow-x-auto text-[12px] text-gray-400">
        <code>{children}</code>
      </pre>
    );
  }

  return (
    <div
      className="my-2 max-w-full overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

interface MarkdownRendererProps {
  content: string;
}

const imageAspectRatioCache = new Map<string, number>();

const ImageBlock: React.FC<{ src?: string; alt?: string }> = ({ src, alt }) => {
  const [loaded, setLoaded] = React.useState(false);
  const initialRatio = src ? imageAspectRatioCache.get(src) : undefined;
  const [ratio, setRatio] = React.useState<number>(initialRatio ?? 16 / 9);

  if (!src) return null;

  return (
    <span className="houyi-md-image" data-loaded={loaded ? '1' : '0'} style={{ aspectRatio: `${ratio}` }}>
      <img
        src={src}
        alt={alt ?? ''}
        loading="lazy"
        decoding="async"
        onLoad={(e) => {
          const img = e.currentTarget;
          const w = img.naturalWidth;
          const h = img.naturalHeight;
          if (w > 0 && h > 0) {
            const r = w / h;
            imageAspectRatioCache.set(src, r);
            setRatio(r);
          }
          setLoaded(true);
        }}
        className={loaded ? 'houyi-md-image__img' : 'houyi-md-image__img houyi-md-image__img--hidden'}
        draggable={false}
      />
    </span>
  );
};

const components: Components = {
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '');
    const lang = match?.[1];
    const codeStr = String(children).replace(/\n$/, '');

    // Inline code (no language class, short content)
    const isInline = !className && !codeStr.includes('\n');
    if (isInline) {
      return (
        <code className="px-1 py-0.5 bg-gray-800 rounded text-[12px] text-pink-300 font-mono" {...props}>
          {children}
        </code>
      );
    }

    // Mermaid diagram
    if (lang === 'mermaid') {
      return <MermaidBlock>{codeStr}</MermaidBlock>;
    }

    // LaTeX/math code blocks — render as KaTeX display math
    if (lang === 'latex' || lang === 'math' || lang === 'tex') {
      return <LatexBlock>{codeStr}</LatexBlock>;
    }

    // Syntax-highlighted code block
    return <CodeBlock language={lang}>{codeStr}</CodeBlock>;
  },

  // Override pre to avoid double-wrapping (CodeBlock already has <pre>)
  pre({ children }) {
    return <>{children}</>;
  },

  // Table styling
  table({ children }) {
    return (
      <div className="my-2 max-w-full overflow-x-auto rounded-md border border-gray-700/50">
        <table className="min-w-full table-auto text-[12px]">{children}</table>
      </div>
    );
  },
  thead({ children }) {
    return <thead className="bg-gray-800/80">{children}</thead>;
  },
  th({ children }) {
    return (
      <th className="px-3 py-1.5 text-left text-[11px] font-semibold text-gray-300 border-b border-gray-700/50 min-w-[120px]">
        {children}
      </th>
    );
  },
  td({ children }) {
    return (
      <td className="px-3 py-1.5 text-gray-300 border-b border-gray-800/50 min-w-[120px]">
        {children}
      </td>
    );
  },

  // Block elements
  blockquote({ children }) {
    return (
      <blockquote className="my-2 pl-3 border-l-2 border-gray-600 text-gray-400 italic">
        {children}
      </blockquote>
    );
  },
  hr() {
    return <hr className="my-3 border-gray-700/50" />;
  },

  // Headings
  h1({ children }) {
    return <h1 className="text-lg font-bold text-gray-100 mt-3 mb-1">{children}</h1>;
  },
  h2({ children }) {
    return <h2 className="text-base font-bold text-gray-100 mt-3 mb-1">{children}</h2>;
  },
  h3({ children }) {
    return <h3 className="text-sm font-semibold text-gray-200 mt-2 mb-1">{children}</h3>;
  },

  // Lists
  ul({ children }) {
    return <ul className="my-1 ml-4 list-disc text-gray-200">{children}</ul>;
  },
  ol({ children }) {
    return <ol className="my-1 ml-4 list-decimal text-gray-200">{children}</ol>;
  },
  li({ children }) {
    return <li className="my-0.5">{children}</li>;
  },

  // Links
  a({ href, children }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-400 hover:text-blue-300 underline underline-offset-2"
      >
        {children}
      </a>
    );
  },

  // Paragraphs
  p({ children }) {
    return <p className="my-1">{children}</p>;
  },

  img({ src, alt }) {
    return <ImageBlock src={src} alt={alt} />;
  },

  // Strong / emphasis
  strong({ children }) {
    return <strong className="font-semibold text-gray-100">{children}</strong>;
  },
  em({ children }) {
    return <em className="italic text-gray-300">{children}</em>;
  },
};

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = React.memo(({ content }) => {
  return (
    <div className="min-w-0 max-w-full overflow-x-hidden">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

MarkdownRenderer.displayName = 'MarkdownRenderer';
