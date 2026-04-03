# HouYi <Feature> Design

**Version**: v1.0
**Status**: Draft | Design Review | Approved
**Created**: YYYY-MM-DD
**Last Updated**: YYYY-MM-DD
**Companion Doc**: [<feature>-acceptance.md](./<feature>-acceptance.md)
**Design Reference**: <!-- Other design docs referenced, e.g. [chatbox-design.md](./chatbox-design.md) -->
**Related**: <!-- Related documents, e.g. [xxx-acceptance.md](./xxx-acceptance.md) -->

---

## 0. Design Review Log

> Append per review round: date, conclusion, absorbed changes.

### Review #1 (YYYY-MM-DD)

**Conclusion**: <Approved / Conditionally approved / Revision required>

- Issue 1: <description> → Absorbed, see §X
- Issue 2: <description> → Deferred to Phase N

### Change History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | YYYY-MM-DD | Initial version |

---

## Table of Contents

<!-- Update per actual sections -->

---

## Executive Summary

<One paragraph: what this feature is, the core problem it solves, and its positioning within HouYi.>

### Target Overview

> **End-state first**: Set challenging but justifiable terminal goals, benchmarked against industry SOTA.
> Decompose into phased milestones, each focused on deliverable incremental value.
> Detailed research in §2. Benchmark verification framework in §2.4.

| Dimension | Industry SOTA (Who, Value) | HouYi Phase Target | HouYi Terminal Target | Differentiation |
|-----------|---------------------------|--------------------|-----------------------|-----------------|
| ... | ... | ... | Aligned with SOTA | ... |

### Key Decision Summary

| # | Decision | Choice | Status | Notes |
|---|----------|--------|--------|-------|
| D1 | ... | ... | ✅ Decided | ... |

### Core Design Principles

- **SDK as Foundation**: Core capabilities are designed and implemented at the SDK layer; applications build on the SDK and feed requirements back to drive its evolution
- **Scenario Validation**: Real product scenarios validate SDK design soundness, expose gaps, and drive iteration
- **Industrial-Grade**: Quantifiable metrics are the ultimate quality measure
- **Incremental Delivery**: Each phase delivers a complete, usable product capability

---

## 1. Design Goals & Scope

### 1.1 In Scope

### 1.2 Out of Scope

### 1.3 Conditional

### 1.4 Design Constraints

---

## 2. Industry Research & Landscape

> Survey existing products and frameworks to map the competitive landscape and position HouYi's technical approach.

### 2.1 Product Overview

| Product / Framework | Positioning | Core Capability | Limitations |
|---------------------|------------|----------------|-------------|
| <Product A> | ... | ... | ... |
| <Product B> | ... | ... | ... |

### 2.2 Capability Comparison

| Dimension | Industry State (Representative) | HouYi Approach | Technical Advantage |
|-----------|--------------------------------|---------------|---------------------|
| ... | ... | ... | ... |

### 2.3 Gap Analysis & Technical Roadmap

> Based on the comparison above, identify current gaps, technical entry points, and phased catch-up paths.
> **Reminder**: Stay current with recent industry practices and paradigms (e.g., event-driven architectures, multimodal fusion).
> Ensure the design direction keeps pace with the field. For promising frontier ideas, describe HouYi's integration and landing approach.

### 2.4 Benchmark Verification Framework

> **Verifiable closed-loop**: Each design target should have a corresponding benchmark (industry-standard or custom)
> as its delivery acceptance criterion. Targets without measurable benchmarks cannot be objectively verified.

| Benchmark | Source | Dimensions | HouYi Phase Target | HouYi Terminal Target |
|-----------|--------|-----------|--------------------|-----------------------|
| <Industry Benchmark A> | ... | ... | ... | ... |
| <Custom Benchmark (if needed)> | This project | ... | ... | ... |

**Bench scaffold requirements**:
- Evaluation scripts committed to repo (`benchmarks/<feature>/`), runnable with a single command
- Each Phase Gate delivery must pass the corresponding benchmark threshold
- Results recorded as Gate verification evidence in the acceptance document

---

## 3. Architecture Overview

> Diagram conventions: Mermaid preferred, ASCII also accepted.
> This section must include at least these four diagram types:

### 3.1 System Architecture Diagram

> Module layering and dependency relationships.

```
┌─────────────────────────────────────┐
│              UI Layer               │
├─────────────────────────────────────┤
│           Server Layer              │
├─────────────────────────────────────┤
│            SDK Layer                │
└─────────────────────────────────────┘
```

### 3.2 Core Object Interaction Sequence Diagram

> Call order and data flow between modules in the main flow.

### 3.3 Key Module Flowchart

> Internal decision and branching logic for the most complex algorithm or pipeline.

### 3.4 Core Class Diagram

> UML class diagram or equivalent showing inheritance, composition, and dependency relationships.

### 3.5 Integration Boundaries

---

## 4. Data Model & Type System

### 4.1 Core Type Definitions

### 4.2 Type Relationships

---

## 5. Core Algorithms / Pipelines

### 5.1 Main Flow

### 5.2 Key Algorithms

### 5.3 Performance Constraints

---

## 6. API Contract

> This is the canonical definition for all interfaces. Acceptance docs and code must align with this.

### 6.1 SDK Layer API

### 6.2 Server Layer API (REST / SSE)

### 6.3 UI Layer Interface

> **Flexible organisation**: For large features with multiple subsystems (e.g. Deep Research spans Memory Engine + Agent Runtime + Research Engine), §4-6 may be organised by subsystem instead, with each subsystem containing its own data model, algorithms, and API sections.

---

## 7. Configuration & Tuning

| Config | Default | Description |
|--------|---------|-------------|
| ... | ... | ... |

---

## 8. Error Handling & Observability

### 8.1 Error Classification & Recovery

### 8.2 Logging & Tracing

---

## 9. UI Design (if applicable)

> For features with UI deliverables: page design, interaction flows, state models.

### 9.1 Page Structure & Layout

### 9.2 Interaction Design & State Flow

### 9.3 SSE / Real-time Data Convergence

---

## 10. Integration with Other Design Docs

> Define boundaries and sync points with other existing design documents.

| Document | Integration Point | Items to Sync |
|----------|-------------------|---------------|
| ... | ... | ... |

---

## 11. Delivery Plan

> Overview only; detailed Sprint task breakdown lives in the acceptance doc. Uses complexity scale, not time estimates.

| Phase | Scope | Overall Complexity | Core Tasks |
|-------|-------|--------------------|------------|
| Phase 1 | ... | L | ~N |
| Phase 2 | ... | XL | ~N |

**Complexity scale**: S (< 80 LOC) / M (80–200) / L (200–400) / XL (400+)

---

## 12. Risk & Mitigation

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| R1 | ... | High/Med/Low | High/Med/Low | ... |

---

## 13. Code Directory Structure


```
houyi/
├── adapters/<feature>/       # SDK layer implementation
├── application/<feature>/    # Application layer orchestration
├── domain/<feature>/         # Domain models
tests/
├── adapters/<feature>/       # Unit tests (mirror source tree)
├── application/<feature>/    # Application layer tests
houyi-studio/
├── server/.../               # Server layer
├── ui/src/components/...     # UI components
```

---

## Appendix

### A. Industry Comparison Matrix (Detailed)

### B. Glossary

### C. Decision Log

> Record key technical debates and the rationale behind final choices.
