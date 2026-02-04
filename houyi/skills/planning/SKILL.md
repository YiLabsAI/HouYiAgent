---
name: planning-with-files
version: "1.0.0"
description: |
  Task planning skill that creates and manages task plans in markdown files.
  Tracks progress through a structured plan file, ensuring tasks are completed
  systematically before marking as done.

user-invocable: true

allowed-tools:
  - Write
  - Read
  - Edit
  - StrReplace
  - Glob
  - Grep

hooks:
  PreToolUse:
    - matcher: "Write|Edit|StrReplace"
      type: handler
      handler: houyi.skills.planning.hooks:pre_write_hook
  PostToolUse:
    - matcher: ".*"
      type: handler
      handler: houyi.skills.planning.hooks:post_tool_hook
  Stop:
    - type: handler
      handler: houyi.skills.planning.hooks:stop_hook

invocationPolicy:
  modelAutoInvoke: allow
  userInvocable: true
  sideEffect: filesystem

permissions:
  filesystem:
    read: true
    write: true
    paths:
      - "${WORKSPACE}/**/*.md"
      - "${WORKSPACE}/.plan/**"
---

# Planning with Files

A task planning skill that helps you manage complex tasks through structured plan files.

## Overview

This skill creates and maintains a `PLAN.md` file that tracks:
- Task breakdown into subtasks
- Progress status for each subtask
- Completion criteria

## Usage

When starting a complex task, create a plan:

```
/plan create "Implement user authentication"
```

The skill will:
1. Create a `PLAN.md` file with task breakdown
2. Track progress as you work through subtasks
3. Prevent premature completion until all subtasks are done

## Plan File Format

```markdown
# Task: [Task Description]

## Status: in_progress | completed | blocked

## Subtasks

- [ ] Subtask 1 description
- [x] Subtask 2 description (completed)
- [ ] Subtask 3 description

## Notes

Any relevant notes or context.
```

## Hooks

### PreToolUse (Write/Edit)

Before any file write, checks if the operation aligns with the current plan.

### PostToolUse

After each tool use, updates plan progress if relevant.

### Stop

Before stopping, verifies all planned subtasks are complete.
If incomplete tasks remain, prompts to continue or explicitly acknowledge incomplete work.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["create", "update", "complete", "status"],
      "description": "Action to perform on the plan"
    },
    "task": {
      "type": "string",
      "description": "Task description (for create action)"
    },
    "subtask_index": {
      "type": "integer",
      "description": "Subtask index to update (for update action)"
    },
    "status": {
      "type": "string",
      "enum": ["pending", "in_progress", "completed", "blocked"],
      "description": "New status (for update action)"
    }
  },
  "required": ["action"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean"
    },
    "plan_path": {
      "type": "string",
      "description": "Path to the plan file"
    },
    "status": {
      "type": "string",
      "description": "Current plan status"
    },
    "progress": {
      "type": "object",
      "properties": {
        "total": { "type": "integer" },
        "completed": { "type": "integer" },
        "percentage": { "type": "number" }
      }
    },
    "message": {
      "type": "string"
    }
  }
}
```
