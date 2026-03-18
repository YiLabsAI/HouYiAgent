import { describe, it, expect } from 'vitest';
import {
  normalizeFullWidthPunctuation,
  sanitizeMermaidCode,
} from '@/utils/mermaidSanitize';

describe('normalizeFullWidthPunctuation', () => {
  it('converts parentheses', () => {
    expect(normalizeFullWidthPunctuation('API（Server）')).toBe('API(Server)');
  });

  it('converts full-width brackets', () => {
    expect(normalizeFullWidthPunctuation('text【A】')).toBe('text[A]');
  });

  it('converts mixed punctuation', () => {
    expect(normalizeFullWidthPunctuation('a：b，c；d')).toBe('a:b,c;d');
  });

  it('keeps CJK text', () => {
    const input = 'API Server 控制平面 (Control)';
    expect(normalizeFullWidthPunctuation(input)).toBe(input);
  });
});

describe('sanitizeMermaidCode', () => {
  // ─── The K8s diagram that user reported as failing ─────────────
  const k8sCode = `%% 简化的K8s架构图（Mermaid语法）
graph TD
    subgraph Control Plane
        API[API Server]
        Scheduler
        CM[Controller Manager]
        ETCD[etcd]
    end

    subgraph Data Plane (Workers)
        Kubelet
        KubeProxy
        ContainerRuntime
        Pod1[Pod]
        Pod2[Pod]
    end

    API -->|通信| Scheduler
    API -->|通信| CM
    API -->|存储状态| ETCD
    API -->|指令下发| Kubelet
    Kubelet -->|管理| ContainerRuntime
    KubeProxy -->|网络规则| Pod1
    KubeProxy -->|网络规则| Pod2`;

  it('fixes subgraph titles', () => {
    const result = sanitizeMermaidCode(k8sCode);
    // The problematic line should now have a synthetic ID + quoted title
    expect(result).toContain('subgraph sg_0 ["Data Plane (Workers)"]');
    // The non-problematic subgraph should remain unchanged
    expect(result).toContain('subgraph Control Plane');
  });

  it('normalizes comment parens', () => {
    const result = sanitizeMermaidCode(k8sCode);
    // Comment's full-width （ and ） should be normalized
    expect(result).toContain('K8s架构图(Mermaid语法)');
  });

  it('keeps edge labels', () => {
    const result = sanitizeMermaidCode(k8sCode);
    expect(result).toContain('-->|通信|');
    expect(result).toContain('-->|存储状态|');
    expect(result).toContain('-->|指令下发|');
  });

  it('preserves node definitions', () => {
    const result = sanitizeMermaidCode(k8sCode);
    expect(result).toContain('API[API Server]');
    expect(result).toContain('CM[Controller Manager]');
    expect(result).toContain('Pod1[Pod]');
  });

  // ─── Subgraph edge cases ───────────────────────────────────────

  it('keeps plain subgraphs', () => {
    const input = 'subgraph My Section\n  A\nend';
    expect(sanitizeMermaidCode(input)).toContain('subgraph My Section');
  });

  it('keeps explicit subgraph ids', () => {
    const input = 'subgraph myId ["Title (with parens)"]';
    expect(sanitizeMermaidCode(input)).toBe(input);
  });

  it('handles multiple subgraphs', () => {
    const input = `graph TD
    subgraph A (first)
      X
    end
    subgraph B (second)
      Y
    end`;
    const result = sanitizeMermaidCode(input);
    expect(result).toContain('subgraph sg_0 ["A (first)"]');
    expect(result).toContain('subgraph sg_1 ["B (second)"]');
  });

  // ─── Node label quoting ────────────────────────────────────────

  it('quotes node labels', () => {
    const input = 'graph TD\n  A[text (extra)]';
    const result = sanitizeMermaidCode(input);
    expect(result).toContain('["text (extra)"]');
  });

  it('avoids double quotes', () => {
    const input = 'graph TD\n  A["already quoted (ok)"]';
    const result = sanitizeMermaidCode(input);
    // Should not add another layer of quotes
    expect(result).toContain('["already quoted (ok)"]');
  });

  // ─── Full-width in structural positions ─────────────────────────

  it('fixes full-width node shape', () => {
    // LLM might output A（text） instead of A(text)
    const input = 'graph TD\n  A（圆形节点）';
    const result = sanitizeMermaidCode(input);
    expect(result).toContain('A(圆形节点)');
  });

  it('is idempotent', () => {
    const once = sanitizeMermaidCode(k8sCode);
    const twice = sanitizeMermaidCode(once);
    expect(twice).toBe(once);
  });
});
