# HouYi <Feature> Acceptance & Delivery Tracking

**Version**: v1.0
**Status**: Tracking
**Created**: YYYY-MM-DD
**Last Updated**: YYYY-MM-DD
**Design Reference**: [<feature>-design.md](./<feature>-design.md) vX.Y
**Related**: <!-- Related documents, e.g. [chatbox-acceptance.md](./chatbox-acceptance.md) -->

---

> **Document Scope**: Delivery tracking & acceptance, independent of the design doc.
> Behaviour specs and API definitions defer to the design doc as the canonical reference.
> This document governs: task breakdown, Phase Gates, test matrices, delivery logs.

## 0. Status Snapshot

| Dimension | Status |
|-----------|--------|
| Current Phase | Phase X |
| Current Sprint | Sprint N |
| Gate Status | Gate X in progress |
| Blockers | None / <describe> |

---

## 1. Implementation Task Breakdown

### Sprint Planning Principles

| Principle | Meaning | Anti-pattern |
|-----------|---------|-------------|
| **Vertical Slice** | Each Sprint delivers a full SDK → Server → UI chain | All SDK first, then all UI |
| **Incremental Value** | Sprint N adds one complete, usable capability | Half-done, needs next Sprint |
| **Dependency-Driven** | Sprint boundaries at natural architecture break points | Arbitrary task assignment |
| **Risk-First** | High-complexity / core architecture modules first | Hardest tasks last |

### Definition of Done Chain

```
Task ✅ ──→ Sprint ✅ ──→ Gate ✅ ──→ Phase ✅
 (built)     (all tasks)    (verified)    (deliverable)
```

| Level | Condition | Anti-pattern |
|-------|-----------|-------------|
| **Task** ✅ | Code + tests pass + lint clean | Code done but tests not run |
| **Sprint** ✅ | ALL tasks in Sprint ✅ | Sprint marked done with tasks still ⬜ |
| **Gate** ✅ | All MUST conditions verified with evidence | Assuming Gate pass from task completion |
| **Phase** ✅ | All MUST Gate conditions ✅ | "Close enough" declaration |

Propagation: Task → Sprint → Gate → Phase. No level may be skipped.
**No false checkmarks**: every ✅ must have runnable verification evidence.

### Verification Workflow (Sprint Delivery Checklist)

```
Step 1: Per-Task Confirmation
  ├─ Code implemented?  Tests pass?  Lint clean?  Coverage met?
  └─ Any task failing → STOP, fix, re-check

Step 2: Sprint Status
  ├─ All tasks ✅ → Mark Sprint ✅
  └─ Any task ⬜ → Sprint CANNOT be ✅

Step 3: Gate Verification
  ├─ For each MUST condition:
  │   ├─ Related tasks all ✅?
  │   ├─ Run verification (tests / inspect / review)
  │   ├─ Fill Result column (✅ or ❌)
  │   └─ Fill Evidence column (test file + count / screenshot / command)
  └─ Any MUST ❌ → STOP, fix, restart from Step 1

Step 4: Phase Delivery Declaration
  ├─ All MUST gates ✅ → Phase complete
  └─ Record in Delivery Log
```

### Engineering Conventions

| Dimension | Rule | Anti-pattern |
|-----------|------|-------------|
| File naming | Source `<module>.py`; test `test_<module>.py` mirror | `test_phase3_stuff.py` |
| Function naming | `test_<subject>_<behaviour>[_<condition>]` ≤ 35 chars | 50-char verbose names |
| Test directory | Mirror `houyi/` source tree exactly | Flat `tests/` dump |
| Test coverage | ≥ 4 cases per module: happy + boundary + error + interaction | All happy-path only |
| Public API | Docstring + type hints required | Bare `dict` / `Any` |
| Comments | English only; explain non-obvious intent | Narration or non-English |

### Complexity Scale

| Code | Meaning | Est. LOC | Agent Rounds |
|------|---------|----------|--------------|
| S | Single file, clear logic | < 80 | 1–2 |
| M | 1–2 files, moderate design | 80–200 | 2–4 |
| L | Multi-file, integration test needed | 200–400 | 4–8 |
| XL | Core architecture, perf validation | 400+ | 8+ |

Status markers: ⬜ Not started · 🔵 In progress · ✅ Complete · ⏸ Blocked

### Two-Tier Test Standard

> Applies to every Phase Gate. Defined once, referenced everywhere.

#### Tier 1: Common Quality Gate (all Phases)

| # | Condition | Verification | Notes |
|---|-----------|-------------|-------|
| Q-1 | `make check` all pass | Exit code 0 | Includes ruff lint + format + complexity + mypy + tsc + pytest + vitest |
| Q-2 | New code line coverage ≥ 90% | `pytest-cov` report | Core modules ≥ 95% |
| Q-3 | Integration + E2E tests pass | `pytest -m integration` | Tests involving real LLM calls may be marked `slow`; can skip in CI but must verify locally |

#### Tier 2: Phase Business Gate (per Gate)

Each Gate defines its own business acceptance conditions. See Gate template below.

**Delivery Log Categorisation**:

| Category | Section | ID Pattern | Example |
|----------|---------|------------|---------|
| **Sprint Delivery** | Sprint N task table | S*-T* | S7-T1 ~ T6 |
| **Bug Fix** | "Round N Fix" paragraph | B*-xx ~ B*-yy | Round 10 Fix (B3-58 ~ B3-61) |
| **Architecture Correction** | "Phase X.Y Architecture" table | F-* | F-4 ~ F-7 |

These three categories must never be mixed in the same section.

---

### Task Overview

| Sprint | Tasks | Overall Complexity | Status |
|--------|-------|--------------------|--------|
| Sprint 1 | N | L | ⬜ |
| Sprint 2 | N | XL | ⬜ |

### Sprint-Phase Mapping & Gate Triggers

| Sprint | Phase | Gate |
|--------|-------|------|
| Sprint 1 | Phase 1 | Gate 1 |
| Sprint 2 | Phase 1 | Gate 1 |

### Task Dependencies (Critical Path)

> Mark cross-Sprint, cross-module critical dependencies and identify parallelisation opportunities.

```
S1-1 → S1-3 → S2-1 → S3-1 (critical path)
S1-2 ──────────→ S2-5 (parallelisable)
```

---

### Sprint N: <Title>

#### N.1 <Functional Area>

| # | Task | Target File | Key Interface | Complexity | Deps | Status |
|---|------|-------------|---------------|------------|------|--------|
| SN-1 | ... | `houyi/...` | `Class.method()` | S | None | ⬜ |

---

## 2. Phase Gate Definitions

> Each Phase must pass its corresponding Gate before advancing.

### Gate N: <Title>

> **Sprint Mapping**: SN-1 ~ SN-X

#### Common Quality Gate: ⬜ Pending

> <After passing, write a one-line evidence summary referencing Q-1 ~ Q-3 above.>

#### Business Gate

| # | Condition | Level | Related Tasks | Result | Evidence |
|---|-----------|-------|---------------|--------|----------|
| GN.1 | ... | MUST | SN-x | ⬜ | ... |
| GN.2 | ... | MUST | SN-y | ⬜ | ... |
| GN.3 | ... | SHOULD | SN-z | ⬜ | ... |

**Gate N Summary**: X/Y PASS. <Brief note on deferred items.>

---

## 3. Phase N Verification

### N.1 Test Matrix

| Test Type | Test File | Cases | Pass | Coverage |
|-----------|-----------|-------|------|----------|
| Unit | `test_xxx.py` | ... | ... | ...% |
| Integration | `test_xxx_integration.py` | ... | ... | — |

### N.2 Smoke Test

| # | User Story | Expected | Actual | Status |
|---|-----------|----------|--------|--------|
| SM-1 | ... | ... | ... | ⬜ |

---

## Performance Benchmarks

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| ... | < Xms | ... | ⬜ |

---

## Regression Checklist

| Module | Regression Item | Verification | Status |
|--------|----------------|-------------|--------|
| ... | ... | `pytest tests/...` | ⬜ |

---

## Delivery Log

> Updated incrementally. Each delivery records: date, completed tasks, Gate verification, issues found.

### Phase N Delivery

#### Sprint N — <Title> ✅

**Date**: YYYY-MM-DD
**Completed Tasks**: SN-1 ~ SN-X (X/X)
**New/Modified Files**:

| Category | File | Description |
|----------|------|-------------|
| ... | `houyi/...` | ... |

**Test Results**:

| Suite | Cases | Status |
|-------|-------|--------|
| `test_xxx.py` | N | ✅ All pass |

**Coverage**: module XX%
**Lint**: zero errors
**Issues Found**:

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| PN-1 | ... | Bug / Design flaw / Naming | ✅ Fixed ... |

---

---

### Delivery Retrospective

> Post-Phase review. For systematic reflection after multiple delivery rounds.

#### Root Cause Analysis

| # | Category | Occurrences | Root Cause |
|---|----------|-------------|------------|
| 1 | ... | N times | ... |

#### Improvement Actions

| # | Action | Scope | Status |
|---|--------|-------|--------|
| 1 | ... | All future Phases | ⬜ |

---

#### Round N Fix (BX-aa ~ BX-bb): <Theme>

| # | Issue | Severity | Solution | Test Coverage |
|---|-------|----------|----------|---------------|
| BX-aa | ... | P0/P1/P2 | ... | `test_x.py` N cases |
