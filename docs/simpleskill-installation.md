# SimpleSkill Installation Guide (Community Skills)

This guide explains how to install community SimpleSkills into HouYi in a reproducible way.

## Scope

Use this guide when you want to install skills such as:

- `planning-with-files`
- `using-superpowers`
- `frontend-design`
- `notebooklm`
- `skill-creator`

## Prerequisites

- HouYi repository cloned locally
- Python/uv environment ready
- Access to the source skill folders

## 1) Install by copying full skill folders

Do **not** copy only `SKILL.md`.
Copy the full skill folder with all bundled resources.

Required content (if present):

- `SKILL.md`
- `examples.md` / `reference.md`
- `scripts/`
- `templates/`
- `references/`
- `assets/`

Example:

```bash
# run in HouYi repo root
cp -R /path/to/source-skill-folder skills/
ls -la skills/<skill-name>/
```

## 2) Keep directory naming canonical

Prefer directory names aligned with `name` in frontmatter.

Good:

- `skills/notebooklm/` (name: `notebooklm`)

Avoid duplicate-name folders loaded at the same time, e.g.:

- `skills/notebooklm/` and `skills/notebooklm-skill/`

This can cause duplicate registration skip warnings and source confusion.

## 3) Load and verify registration

Use startup hooks and inspect serialized summary/detail fields.

```python
from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi_studio.server.skill.startup_hooks import register_console_skills
from houyi_studio.server.skill.serializer import SkillSerializer

register_console_skills()
ser = SkillSerializer()

skill = DEFAULT_SKILL_REGISTRY.get("notebooklm")
summary = ser.to_summary(skill)
print(summary["name"], summary["source"], summary["runtime_binding"], summary["is_core"])
```

Expected:

- `source == "community"` (or `third_party` for protected alias cases)
- `is_core == False`
- `runtime_binding` is populated (`prompt_instructions` or `python_executor`)

## 4) Dry-run verification checklist

For each installed skill, verify at least:

1. Static alignment: instructions/hook evidence exists
2. Routing evidence: expected vs requested_input vs observed tool call
3. Execution evidence: distinguish routing-only from executor-backed runs

## 5) Live verify model/provider selection (UI)

Dry Run Live mode supports provider/model configuration in UI.

For `notebooklm`, recommended default live settings:

- provider: `vertex`
- model: `gemini-2.5-pro`

If live evidence is missing (`observed` empty), first check environment:

- model adapters installed
- provider credentials configured

## 6) Troubleshooting quick table

- **Skill not found after copy**
  - Check folder contains `SKILL.md` and frontmatter `name`
  - Check startup loading path is `skills/`

- **Duplicate skill warning**
  - Remove duplicate folders that map to the same frontmatter name

- **Live verify has no real evidence**
  - Ensure API key/credentials are configured
  - Ensure model adapters are installed

## 7) Recommended acceptance flow

1. Install full folders
2. Start/load skills
3. Verify summary/detail metadata
4. Run dry-run static checks
5. Run live verify with explicit provider/model
6. Record observed evidence and blockers
