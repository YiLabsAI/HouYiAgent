# SimpleSkill Specification

> **Version**: 0.1.0
>
> **Status**: Public Draft
>
> **Purpose**: Define a standard for extending agent capabilities so that capability packages from different hosts and providers can be declared, installed, activated, executed, governed, observed, and evaluated in a consistent way, enabling ecosystem interoperability across skills.
>
> **Scope**: This specification defines the normative model, required clauses, optional capability-gated clauses, and informative implementation mapping for the current SimpleSkill release scope.
>
> **Compatibility strategy**: SimpleSkill is designed for compatibility with the Claude `SKILL.md` format. A conforming host SHOULD accept valid Claude-style `SKILL.md` packages where the host supports the referenced capabilities. This compatibility requirement is one reason this version permits host-native aliases and partial declaration forms for some fields.
>
> **Integration modes**: This specification supports five categories of skill integration modes: hooks-heavy, metadata-only, tools-declarative, builtin `@tool`, and builtin RAG. Together they cover both builtin skills and external Claude-ecosystem skills.

## 1. Normative Terms

- **Host**: The runtime host that executes a SimpleSkill extension, such as an agent framework, IDE agent, or server-side agent runtime.
- **Provider**: The distribution and trust boundary of an extension, such as official, private, community, or third-party.
- **Plugin/Extension**: A distributable, installable, and upgradable capability package.
- **Skill**: A workflow-oriented capability that may orchestrate tools and resources and includes invocation and governance semantics.
- **Tool**: A structured, verifiable, atomic executable interface following a schema-first model.
- **Resource**: A static or dynamic resource that can be read or retrieved, such as files, indexes, datasets, or templates.
- **Capability**: A declared set of optional features supported by a host or extension and used during negotiation.
- **Conformance**: The minimum required tests and supporting evidence showing compliance with the specification.

This specification uses the following normative keywords:

- **MUST / SHALL**
- **SHOULD**
- **MAY**

## 2. Layered Model

### 2.1 Layer A: Package / Manifest

Layer A defines identity, compatibility, activation, contribution points, permission declarations, trust source, and resource layout.

### 2.2 Layer B: Host Runtime API

Layer B defines lifecycle management, capability negotiation, cancellation / timeout, progress, observability, consent, and evaluation interfaces.

### 2.3 Execution Forms

The same specification allows three execution forms to coexist:

- **in-process**: The extension runs in the same process as the host. This provides high performance but requires high trust.
- **subprocess (protocol)**: The extension runs in a subprocess and communicates through a protocol boundary. This isolates crashes and resource abuse and supports cross-language execution.
- **MCP**: The extension exposes tools, resources, or prompts through an MCP server, either local or remote.

The host MUST declare which execution forms it supports through capability negotiation, and MUST provide explicit degradation or rejection semantics when a requested form is unsupported.

## 3. Layer A: Manifest

### 3.1 Manifest File and Location

- The package root MUST contain one manifest file. Recommended filenames are `simpleskill.json` or `simpleskill.yaml`, though implementations MAY extend this.
- The manifest MUST serve as a declarative index. The host MUST NOT be required to load all full-text content in order to support discovery and list presentation. This enables progressive disclosure.

### 3.2 Identity

The manifest MUST include:

- `id`: a unique kebab-case identifier, ideally matching the package directory name
- `version`: a semantic version following SemVer
- `name`: a human-readable name
- `description`: a short description used before loading the full body

The manifest SHOULD include:

- `publisher` / `author`
- `license`
- `repository` (URL)

### 3.3 Compatibility

The manifest MUST declare compatibility requirements:

- `engines.host`: the supported host identifier and minimum version, such as `houyi>=x.y`
- `platform` (optional): OS / architecture constraints

### 3.4 Activation

The manifest MAY declare an array of activation events in `activationEvents[]`. Recommended event types include:

- `onCommand:<id>`: explicit command invocation
- `onTaskMatch:<pattern>`: task or intent matching, with host-defined pattern semantics
- `onFileType:<ext>`: file-type detection
- `onWorkspaceContains:<glob>`: workspace contains a given file or directory pattern
- `onStartupFinished`: delayed activation after startup completes
- `onSchedule:<cron>` (optional)

The host MUST define which activation events it supports. Unsupported events MUST either be ignored or reported as errors according to host policy, and this behavior MUST be declared through capability negotiation.

### 3.5 Contribution Points

The manifest MUST declare a `contributions` index and MUST support at least:

- `tools[]`: structured tool interface declarations, pointing to tool definitions or inline schemas
- `skills[]`: skill declarations, pointing to skill definition files
- `resources[]`: resource declarations, including static files, indexes, and retrievable directories

The manifest MAY support:

- `prompts[]`, `workflows[]`
- `hooks[]`, when the host declares hook capability support

### 3.6 Permissions

The manifest MUST declare the permissions required by the extension in `permissions`:

- `filesystem`: read / write / delete capabilities and path allowlists, including host-defined variables such as workspace and home
- `network`: whether network access is allowed and the domain allowlist
- `exec`: whether external commands may be executed, plus recommended command / argument constraints
- `secrets`: whether access to secrets is required; the host MUST supply this through secure storage and MUST NOT expose secrets in plaintext
- `resources`: CPU / memory / time / concurrency limits, with at least a minimal declaration model

The host MUST use permissions as an input to consent decisions. A provider MAY supply a default policy, but final authority remains with the host and the user’s consent.

### 3.7 Trust

The manifest SHOULD declare:

- `source`: the source category, such as `builtin`, `community`, `private`, or `third_party`
- `signature` (optional): signature and certificate chain
- `verified` (optional): review and certification status

> **Note on Core Protection**: `is_core` is a host-runtime protection property and is not declared in the manifest or `SKILL.md`. A tool with `is_core=True` must come from `trust.source=builtin`, but a builtin source does not imply `is_core=True`. If an external extension attempts to declare `is_core`, the host MUST ignore it.

## 4. Minimum Definition of Skills and Tools

### 4.1 Tool Definition

Each tool MUST define:

- `name`: a globally unique name; namespaced forms such as `pdf.extract_text` or `rag.search` are recommended
- `description`
- `inputSchema`: JSON Schema or an equivalent schema format

Each tool SHOULD define:

- `outputSchema`

Each tool MAY define:

- `execution`: the execution-form declaration (`in-process`, `subprocess`, or `MCP`) and the corresponding entry point
- `permissions`: an optional narrowing of manifest-level permissions

If tool-level permissions are present, they MUST NOT expand the manifest-level permissions.

### 4.2 Skill Definition

Each skill MUST define:

- `id`, or a host-native equivalent identifier field such as `name`
- `description`
- `invocationPolicy` (see §5.2)
- `toolRefs[]`, or a host-native equivalent allowlist field such as `allowed_tools`; this MAY be empty for reference-only skills
- `resources[]` (optional)

A skill SHOULD support progressive disclosure. The host SHOULD be able to discover and select a skill using only metadata such as `id`, `description`, and `invocationPolicy`, without loading the full skill body.

## 5. Layer B: Host Runtime API (Required Capabilities)

### 5.1 Capability Negotiation

The host MUST provide capability negotiation results, whether through an API, environment contract, or initialization handshake. At minimum, negotiation MUST cover:

- supported manifest formats and versions
- supported execution forms: `in-process`, `subprocess(protocol)`, and `MCP`
- supported hook-capability matrix (see §6)
- consent model: interactive, non-interactive, or policy-driven
- observability support: the minimum supported set of traces, events, and metrics
- evaluation support: whether the host can run benchmarks, record scores, and export evidence

### 5.2 Invocation Policy

Each skill MUST define an `invocationPolicy` with at least:

- `modelAutoInvoke`: `allow | deny | allow_with_consent`
- `userInvocable`: boolean
- `sideEffect`: `none | filesystem | network | exec | mixed`

**Relationship among Permissions, SideEffect, and InvocationPolicy**:

- **Permissions** are the foundation. They declare which resources the skill may access, such as path or domain allowlists, and the host enforces them at runtime.
- **SideEffect** is a derived or display-oriented field. It MAY be declared explicitly in the manifest. If it is omitted, the host SHOULD derive it automatically from permissions; for example, filesystem permission implies `filesystem`, and network permission implies `network`. This field is primarily used for quick display in the console. If declared explicitly, the host SHOULD validate consistency with the declared permissions.
- **InvocationPolicy** is the decision layer. It governs whether and how a skill may be invoked based on side effects and permissions.

Normative requirements:

- For a skill whose `sideEffect != none`, `modelAutoInvoke` SHOULD default to `deny` or `allow_with_consent`, according to the host’s baseline policy. The host MUST define a clear default.
- The host MUST use `Permissions + SideEffect + InvocationPolicy` together as input to consent decisions.

### 5.3 Consent

The host MUST implement a unified consent interface, independent of any specific UI, for:

- first-time enablement of extension permissions
- one-off high-risk operations, such as command execution, writing to sensitive paths, or external network requests

Consent MUST be able to represent decisions semantically equivalent to `allow`, `deny`, and `ask_later` (or deferred handling), and MUST record audit information covering who approved or denied what, when, why, and for which permissions or operations.

### 5.4 Observability

The host MUST be able to record at least the following events, whether represented as trace spans, events, or both:

- `ExtensionActivated` / `ExtensionDeactivated`
- `ToolUsageStarted` / `ToolUsageFinished` / `ToolUsageError`

Event attributes SHOULD include:

- tool / skill / plugin ID and version
- redacted input summary, output summary, and error type
- latency, retry count, and cache hit information, if applicable
- permission / consent decision ID, if a consent flow was triggered

## 6. Hooks (Optional, Capability-Gated)

### 6.1 Event Semantics

If the host declares hook support, it SHOULD support at least:

- `PreToolUse` / `PostToolUse`
- `SessionStart` / `Stop`
- `PreExecution` / `PostExecution`, when the host has a plan or execution concept

**Global hook constraint**: The host MAY register builtin global hook handlers for safety governance. Global hooks MUST execute with higher priority than any extension-provided hook handlers.

### 6.2 Handler Types and Degradation

Hook handler types include a portable baseline and optional extensions.

**Portable baseline handler types**:

- `command`: execute a shell command or script; typically used for deterministic preprocessing whose output may be injected into model context
- `handler`: call a host-native function, such as `houyi.skills.planning.hooks.pre_tool_use`

**Optional handler types**:

- `agent`: delegate to a sub-agent; this requires explicit host support and additional isolation or governance controls
- `prompt`: inject a textual prompt into model context
- `tool`: invoke a registered tool as a hook

The host MUST declare which handler types it supports through capability negotiation.

If a handler type is unsupported:

- the host MUST explicitly reject it or degrade it; it MUST NOT silently pretend to support it
- degradation order:
  - `handler -> command -> ignore`
  - `agent -> ignore`

Hosts MAY define additional degradation paths for optional handler types such as `prompt` or `tool`, but such paths MUST be declared through capability negotiation or host documentation.

## 7. Evaluation / Selection

### 7.1 Metrics Schema

Extension-level or host-level evaluation results SHOULD produce standardized metrics in JSON form, including:

- `quality`: such as accuracy, F1, or groundedness, with domain-specific extensions allowed
- `latency_ms`: at least average and p95
- `cost`: optional, such as tokens, dollars, or compute usage
- `reliability`: such as error rate and timeout rate
- `privacy`: whether execution is local or remote, and whether data egress occurs
- `conformance`: pass/fail plus details

### 7.2 Selection Policy

The host SHOULD support a selection policy interface:

- default flow: apply a `conformance` gate first, then rank by user-preference weighting
- users MUST be able to override the selection result, for example by pinning a provider or implementation

## 8. Certification & Expertise (Optional Extensions)

### 8.1 Certification Levels

- Bronze / Silver / Gold / Expert: level definitions SHOULD be tied to conformance results, benchmark performance, and audit evidence

### 8.2 Expertise Bundles

An Expertise bundle is a declaration of a capability cluster and includes:

- `requiredSkills[]` / `requiredTools[]`
- a target level
- evidence references, such as benchmark reports, audit summaries, and version constraints

## 9. Skill Organization Forms

This section defines standard directory layouts, registration entry points, and organization patterns for skills, supporting structures from simple to complex.

### 9.1 Directory Layout

#### 9.1.1 Simple Skill

Suitable for lightweight skills without complex dependencies:

```text
skills/
└── weather/
    ├── simpleskill.yaml
    ├── skill.md
    └── README.md
```

#### 9.1.2 Standard Skill

Suitable for skills with multiple tools, resources, or configuration files:

```text
skills/
└── pdf_processor/
    ├── simpleskill.yaml
    ├── tools/
    │   ├── extract_text.py
    │   ├── extract_images.py
    │   └── schemas/
    │       └── tool_schemas.json
    ├── resources/
    │   └── templates/
    ├── tests/
    │   └── test_extract.py
    └── README.md
```

#### 9.1.3 Complex Skill

Suitable for large capability modules with an independent service layer and multi-provider support, such as RAG or Web Search:

```text
skills/
└── web_search/
    ├── simpleskill.yaml
    ├── __init__.py
    ├── service/
    │   ├── __init__.py
    │   ├── search_service.py
    │   └── result_processor.py
    ├── providers/
    │   ├── __init__.py
    │   ├── google.py
    │   ├── bing.py
    │   └── duckduckgo.py
    ├── tools/
    │   ├── search.py
    │   └── schemas.py
    ├── config/
    │   └── default.yaml
    └── tests/
```

### 9.2 Registration Entry

#### 9.2.1 Manifest-Based Discovery

The host MUST support automatic discovery based on the manifest:

```python
# Host scans for simpleskill.yaml in configured paths
skill_paths = ["houyi/skills/", "~/.houyi/skills/"]
for path in skill_paths:
    for manifest in glob(f"{path}/**/simpleskill.yaml"):
        registry.register_from_manifest(manifest)
```

#### 9.2.2 Programmatic Registration

The host SHOULD support programmatic registration for complex skills or dynamic loading:

```python
from houyi.core.skill import SkillRegistry, SkillSpec

registry.register(SkillSpec(
    id="weather.forecast",
    description="Get weather forecast",
))

registry.register_from_manifest("path/to/simpleskill.yaml")

def register_builtin_skills(registry: SkillRegistry):
    registry.register_from_manifest("houyi/skills/planning/simpleskill.yaml")
    registry.register_from_manifest("houyi/web_search/simpleskill.yaml")
    registry.register_from_manifest("houyi/rag/skills/simpleskill.yaml")
```

**Conflict resolution**:

The host MUST protect its builtin core tools from tampering or replacement by external extensions.

1. **Namespace renaming**: When an external tool conflicts with an already registered core tool and `overwrite=False`, the host SHOULD automatically add an `ext__` prefix to the external tool and emit a warning instead of throwing or replacing the core tool.
2. **Overwrite rejection**: Even when `overwrite=True` is explicitly requested, the host MUST reject overwriting any existing core tool and SHOULD raise a clear security error.

#### 9.2.3 Startup Hooks Integration

For complex skills, the host MAY use startup hooks as a unified registration entry point:

```python
def initialize_skills(context):
    registry = context.skill_registry
    for skill_dir in Path("houyi/skills").iterdir():
        if (skill_dir / "simpleskill.yaml").exists():
            registry.register_from_manifest(skill_dir / "simpleskill.yaml")

    from houyi.web_search import register_skills
    from houyi.rag.skills import register_skills as register_rag_skills

    register_skills(registry)
    register_rag_skills(registry)
```

### 9.3 Organization Patterns

#### 9.3.1 Pattern A: Flat Structure

All skills are placed under one unified `skills/` directory:

```text
houyi/skills/
├── planning/
├── weather/
├── location/
└── calculator/
```

Advantages: simple and uniform loading behavior.  
Suitable for: simple skills and generic capabilities.

#### 9.3.2 Pattern B: Domain-Scoped

Complex modules such as RAG live as top-level directories, and skills are exposed as submodules:

```text
houyi/
├── skills/
│   ├── planning/
│   └── calculator/
├── web_search/
│   ├── service/
│   ├── providers/
│   └── skills/
└── rag/
    ├── indexed/
    ├── agentic/
    └── skills/
```

Advantages: complex modules remain cohesive, while service logic and skill surfaces stay separated.  
Suitable for: capability modules with independent business logic.

#### 9.3.3 Recommended Approach

The host SHOULD adopt **Pattern B** together with startup hooks, so that:

- physical layout remains flexible
- logical discovery stays unified through one API
- progressive migration is possible without breaking existing import paths

### 9.4 Naming Conventions

| Element | Convention | Example |
|------|------|------|
| Skill ID | `domain.action` or `domain.sub.action` | `weather.forecast`, `rag.kb.search` |
| Directory | kebab-case | `web-search/`, `pdf-processor/` |
| Manifest | fixed filename | `simpleskill.yaml` or `simpleskill.json` |
| Tool name | `namespace.verb_noun` | `rag.search_documents`, `pdf.extract_text` |

## 10. Conformance Tests

An extension claiming conformance with this specification MUST pass the following verifiable checks:

- manifest validation: required fields, version correctness, and contribution-index parsing
- tool-schema validation: completeness of input and output schema
- permission-declaration validation: static enforcement of the least-privilege principle
- observability validation: activation and tool-usage events must be emitted
- capability-negotiation consistency: declared support must match runtime behavior

## 11. HouYi Mapping (Informative)

This section is informative. It explains how specification clauses map to the HouYi reference implementation.

### 11.1 Mapping from Spec Clauses to Code

| Specification clause | HouYi code location | Notes |
|---------|---------------|------|
| §3 Manifest | `houyi/domain/skill/manifest.py` | parsing for `simpleskill.json` / `simpleskill.yaml` |
| §3.5 Contribution Points | `houyi/domain/skill/manifest.py` and `houyi/domain/skill/spec.py` | manifest indexes for tools / skills / resources |
| §3.6 Permissions | `houyi/domain/skill/policy.py` and `houyi/domain/skill/manifest.py` | path / domain allowlists and permission parsing |
| §3.7 Trust | `houyi/domain/skill/spec.py` | `is_core` protection and host-side trust handling |
| §4.1 Tool Definition | `houyi/domain/skill/spec.py` and `houyi/domain/skill/manifest.py` | current host rendering exposes `name`, `description`, and `inputSchema` equivalents; `outputSchema` is modeled and recommended by the spec |
| §4.2 Skill Definition | `houyi/domain/skill/spec.py` | current host uses `name` and `allowed_tools` as host-native equivalents of `id` and `toolRefs[]` |
| §5.1 Capability Negotiation | `houyi/domain/skill/capability.py` | host-extension capability negotiation |
| §5.2 InvocationPolicy | `houyi/domain/skill/policy.py` | `allow` / `allow_with_consent` / `deny` + side effects |
| §5.3 Consent | `houyi/domain/skill/consent.py` | host-native consent results are mapped to semantic equivalents of allow / deny / defer, with audit logging |
| §5.4 Observability | `houyi/domain/skill/metrics.py` and host observability integrations | metrics store exists; concrete event plumbing is host-defined |
| §6 Hooks | `houyi/domain/skill/hooks.py` | `PreToolUse`, `PostToolUse`, `SessionStart`, `Stop` |
| §6.2 Handler Types | `houyi/domain/skill/hooks.py` and `houyi/domain/skill/capability.py` | stable baseline is `command` + `handler`; `agent` is capability-negotiated; other types are optional |
| Core Protection | `houyi/domain/skill/spec.py` | `is_core` is treated as a host-runtime protection property |
| §7 Metrics | `houyi/domain/skill/metrics.py` | quality / latency / cost / reliability / privacy |
| §8 Certification | `houyi/domain/skill/spec.py` | field model exists; full workflow is implementation-defined |
| §9 Skill Organization | `houyi/skills/` + `houyi/rag/skills/` | builtin and packaged skill organization patterns |
| §10 Conformance Tests | `tests/domain/skill/` | validation of manifest, schema, policy, consent, hooks, and metrics |
| `SKILL.md` parsing | `houyi/domain/skill/spec.py` | compatibility with Claude-style frontmatter parsing |
| Console integration | `houyi-studio/server/houyi_studio/server/skill_service.py` | list, detail, metrics, load, unload, dry-run, and consent surfaces |
| Console UI | `houyi-studio/ui/src/components/` | list, detail panel, and client-side skill interaction logic |

### 11.2 Mapping of Integration Modes

The five supported integration modes map into HouYi as follows:

| Integration mode | Representative skill | Code entry | Validation |
|---------|---------|---------|---------|
| hooks-heavy | planning-with-files | `houyi/skills/planning/SKILL.md` + hooks | raw external `SKILL.md` loading + full hook chain |
| metadata-only | superpowers, frontend-design | `SkillSpec.from_file()` + `extra_frontmatter` | parsing + preservation of unknown fields |
| tools-declarative | skill-creator | `SkillSpec.from_file()` + `toolRefs` allowlist | `allowed-tools` enforcement |
| builtin `@tool` | weather, location, web_search | startup-hook registration | visible in skill list + available in dry run |
| builtin RAG | kb-search / ingest / graph / analyze | `houyi/rag/skills/SKILL.md` | permission rendering + policy evaluation |
