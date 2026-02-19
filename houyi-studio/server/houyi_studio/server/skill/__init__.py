"""Skill management subsystem.

Public API:
    - SkillService: Facade for skill lifecycle management (list, detail, configure,
      metrics, GitHub import)
    - SkillCommandHandler: WebSocket command handler for skill operations
      (list, detail, configure, load, unload, dry-run, consent)

Internal:
    - DryRunValidator: Skill validation and dry-run execution
    - SkillLoader: Load/unload skills from filesystem or URLs
    - SkillSerializer: Serialize skill metadata for UI consumption
    - startup_hooks: Register built-in skills on server startup
"""
