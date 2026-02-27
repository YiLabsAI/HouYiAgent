"""Startup hooks for console server.

Registers built-in skills for the console runtime.
User/custom skills will be loaded through a unified mechanism (SimpleSkill project).
"""

from __future__ import annotations

import logging
import os

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY

from .service import SkillService, set_skill_service

logger = logging.getLogger(__name__)
_EXTERNAL_PREFIX = "ext__"


def _as_core(skill):
    """Mark a host-built-in skill as core for registry protection."""
    return skill.model_copy(update={"is_core": True})


def _register_builtin_core(skill, registered_skills: list[str]) -> None:
    """Register a built-in skill as core, idempotently.

    If the same core skill is already present (e.g., repeated startup init),
    skip re-registration.
    """
    core_skill = _as_core(skill)
    existing = DEFAULT_SKILL_REGISTRY.get(core_skill.name)
    if existing is not None and getattr(existing, "is_core", False):
        logger.debug("Core skill already registered, skip: %s", core_skill.name)
        if core_skill.name not in registered_skills:
            registered_skills.append(core_skill.name)
        return

    DEFAULT_SKILL_REGISTRY.register(core_skill, overwrite=True)
    registered_skills.append(core_skill.name)


def _group_registered_names(
    registered_skills: list[str],
    registry=None,
) -> tuple[list[str], list[str]]:
    """Group registered names into core and external buckets.

    Unknown names are treated as external for visibility in logs.
    """
    skill_registry = registry or DEFAULT_SKILL_REGISTRY
    seen: set[str] = set()
    core_names: list[str] = []
    external_names: list[str] = []

    for name in registered_skills:
        if not name or name in seen:
            continue
        seen.add(name)
        skill = skill_registry.get(name)
        if skill is not None and getattr(skill, "is_core", False):
            core_names.append(name)
        else:
            external_names.append(name)

    return core_names, external_names


def _schema_is_empty(schema: object | None) -> bool:
    """Return True when schema is missing or has no declared fields."""
    if schema is None or not hasattr(schema, "model_json_schema"):
        return True
    try:
        payload = schema.model_json_schema()
    except Exception:
        return True
    if not isinstance(payload, dict):
        return True
    props = payload.get("properties")
    required = payload.get("required")
    return not props and not required


def _hydrate_external_runtime(
    registered_skills: list[str],
    registry=None,
) -> list[str]:
    """Hydrate external alias skills with core runtime when missing.

    Two-phase resolution:
    1. RuntimeResolver: if a skill declares a ``runtime`` contract with an
       ``adapter``, dynamically import and bind it as executor.
    2. Core fallback: if the skill still lacks an executor, inherit from the
       matching core skill (ext__X -> X).

    This keeps ecosystem skill integration lightweight while preserving
    core-tool protection and naming isolation.
    """
    skill_registry = registry or DEFAULT_SKILL_REGISTRY

    # Phase 1: resolve runtime contracts via adapter import
    try:
        from houyi.core.skill.runtime_resolver import RuntimeResolver

        resolver = RuntimeResolver()
        for name in registered_skills:
            if not name:
                continue
            skill = skill_registry.get(name)
            if skill is None:
                continue
            rc = getattr(skill, "runtime_contract", None)
            if rc is None:
                continue
            resolved = resolver.resolve(skill)
            if resolved is not skill:
                skill_registry.register(resolved, overwrite=True)
    except ImportError:
        pass

    # Phase 2: core fallback hydration
    hydrated: list[str] = []

    for name in registered_skills:
        if not name or not name.startswith(_EXTERNAL_PREFIX):
            continue
        external = skill_registry.get(name)
        if external is None:
            continue

        from houyi.core.skill.runtime_contract import CapabilityTier

        if getattr(external, "capability_tier", None) == CapabilityTier.EXECUTABLE and callable(
            getattr(external, "executor", None)
        ):
            continue
        if callable(getattr(external, "executor", None)) and not _schema_is_empty(
            getattr(external, "input_schema", None)
        ):
            continue

        core_name = name[len(_EXTERNAL_PREFIX) :]
        core = skill_registry.get(core_name)
        if core is None:
            continue
        core_executor = getattr(core, "executor", None)
        if not callable(core_executor):
            continue

        updates: dict[str, object] = {"executor": core_executor}
        if _schema_is_empty(getattr(external, "input_schema", None)):
            updates["input_schema"] = core.input_schema
        if _schema_is_empty(getattr(external, "output_schema", None)):
            updates["output_schema"] = core.output_schema

        hydrated_skill = external.model_copy(update=updates)
        skill_registry.register(hydrated_skill, overwrite=True)
        hydrated.append(name)
        logger.info(
            "Hydrated external skill runtime '%s' from core '%s'",
            name,
            core_name,
        )

    return hydrated


def register_console_skills() -> None:
    """Register built-in skills for console runtime.

    Registers the following skills:
    - Web search: web_search
    - Weather: get_date, get_weather (real Open-Meteo API, with hooks)
    - Location: get_location
    - RAG: kb-search (if available)

    Note: User/custom skills should be loaded through the unified skill
    loading mechanism (to be implemented in SimpleSkill project optimization).
    """
    registered_skills: list[str] = []

    # Initialize SkillService with governance components
    _init_skill_service()

    # 1. Web search skill
    try:
        from houyi.web_search.skill import build_web_search_skill

        _register_builtin_core(build_web_search_skill(), registered_skills)
    except ImportError as e:
        logger.warning("Web search skill not available: %s", e)

    # 2. Weather tools (each @tool is a SkillSpec)
    try:
        from houyi.skills.weather import get_date, get_weather

        _register_builtin_core(get_date, registered_skills)
        _register_builtin_core(get_weather, registered_skills)
    except ImportError as e:
        logger.warning("Weather skills not available: %s", e)

    # 3. Location tool
    try:
        from houyi.skills.location import get_location

        _register_builtin_core(get_location, registered_skills)
    except ImportError as e:
        logger.warning("Location skill not available: %s", e)

    # 4. RAG skills (kb-search, kb-ingest, kb-graph, kb-analyze)
    try:
        from houyi.rag.skills.kb_search import kb_search_skill

        _register_builtin_core(kb_search_skill, registered_skills)
    except ImportError as e:
        logger.debug("RAG kb-search skill not available: %s", e)

    try:
        from houyi.rag.skills.kb_ingest import kb_ingest_skill

        _register_builtin_core(kb_ingest_skill, registered_skills)
    except ImportError as e:
        logger.debug("RAG kb-ingest skill not available: %s", e)

    try:
        from houyi.rag.skills.kb_graph import kb_graph_skill

        _register_builtin_core(kb_graph_skill, registered_skills)
    except ImportError as e:
        logger.debug("RAG kb-graph skill not available: %s", e)

    try:
        from houyi.rag.skills.kb_analyze import kb_analyze_skill

        _register_builtin_core(kb_analyze_skill, registered_skills)
    except ImportError as e:
        logger.debug("RAG kb-analyze skill not available: %s", e)

    # 5. Planning skill
    try:
        from houyi.skills.planning import PlanningSkill

        planning_skill = PlanningSkill()
        _register_builtin_core(planning_skill.to_spec(), registered_skills)
    except ImportError as e:
        logger.debug("Planning skill not available: %s", e)

    # 6. External / community SKILL.md files from skills/ directory
    _load_external_skills(registered_skills)
    _hydrate_external_runtime(registered_skills)

    # Log summary
    total_skills = len(DEFAULT_SKILL_REGISTRY.list())
    core_names, external_names = _group_registered_names(registered_skills)
    logger.info(
        "Registered %d skills (core=%d, external=%d): core=[%s]; external=[%s]",
        total_skills,
        len(core_names),
        len(external_names),
        ", ".join(core_names) if core_names else "-",
        ", ".join(external_names) if external_names else "-",
    )


def _load_external_skills(registered_skills: list[str]) -> None:
    """Load external/community SKILL.md files from the skills/ directory.

    Scans the project-root ``skills/`` directory for sub-directories that
    contain a ``SKILL.md`` file and registers each one.  Skills that are
    already registered (e.g. identical name to a built-in) are skipped
    unless they have a different name to avoid overwrites.
    """
    import pathlib

    # Resolve skills/ relative to project root
    this_file = pathlib.Path(__file__).resolve()
    # skill/startup_hooks.py -> server -> houyi_studio -> server -> houyi-studio -> root
    project_root = this_file.parents[5]
    skills_dir = project_root / "skills"

    if not skills_dir.is_dir():
        logger.debug("External skills directory not found: %s", skills_dir)
        return

    discovered_names = _discover_external_skill_names(skills_dir)

    loaded = DEFAULT_SKILL_REGISTRY.register_from_directory(
        skills_dir,
        pattern="SKILL.md",
        recursive=True,
        overwrite=False,
    )
    registered_skills.extend(loaded)
    pruned = _prune_stale_external_skills(skills_dir, discovered_names)
    if pruned:
        logger.info("Pruned %d stale external skills: %s", len(pruned), ", ".join(pruned))
    if loaded:
        logger.info("Loaded %d external skills from %s", len(loaded), skills_dir)
        logger.debug("External skills: %s", ", ".join(loaded))


def _discover_external_skill_names(skills_dir) -> set[str]:
    """Discover declared skill names from SKILL.md files under *skills_dir*."""
    from houyi.core.skill.spec import SkillSpec

    discovered: set[str] = set()
    for skill_path in skills_dir.glob("**/SKILL.md"):
        try:
            spec = SkillSpec.from_file(str(skill_path))
        except Exception:
            continue
        name = getattr(spec, "name", "")
        if isinstance(name, str) and name.strip():
            discovered.add(name.strip())
    return discovered


def _prune_stale_external_skills(
    skills_dir, discovered_names: set[str], registry=None
) -> list[str]:
    """Remove stale external skills whose local SKILL.md no longer exists."""
    skill_registry = registry or DEFAULT_SKILL_REGISTRY

    skills_root = skills_dir.resolve()
    effective_discovered: set[str] = set(discovered_names)
    for name in list(discovered_names):
        if name.startswith(_EXTERNAL_PREFIX):
            effective_discovered.add(name[len(_EXTERNAL_PREFIX) :])
        else:
            effective_discovered.add(f"{_EXTERNAL_PREFIX}{name}")

    stale_names: list[str] = []

    for skill in list(skill_registry.list()):
        if getattr(skill, "is_core", False):
            continue
        skill_dir = getattr(skill, "skill_dir", None)
        if skill_dir is None:
            continue
        try:
            skill_dir_path = skill_dir.resolve()
        except Exception:
            continue
        if skills_root not in skill_dir_path.parents and skill_dir_path != skills_root:
            continue

        skill_name = getattr(skill, "name", None)
        if not isinstance(skill_name, str) or not skill_name:
            continue
        if skill_name in effective_discovered:
            continue

        if skill_registry.unregister(skill_name):
            stale_names.append(skill_name)

    return stale_names


def _init_skill_service() -> None:
    """Initialize SkillService with optional governance components.

    This function sets up the global SkillService instance with:
    - MetricsStore: For collecting skill execution metrics
    - PolicyEnforcer: For invocation policy checks (if configured)
    - ConsentManager: For user consent flow (if UI consent is enabled)
    """
    try:
        from houyi.core.skill.metrics import InMemoryMetricsStore

        metrics_store = InMemoryMetricsStore()
        logger.debug("Initialized MetricsStore for skill metrics collection")
    except ImportError:
        metrics_store = None
        logger.debug("MetricsStore not available")

    policy_enforcer = None
    consent_manager = None

    # Initialize PolicyEnforcer if governance is enabled
    enable_governance = (os.getenv("HOUYI_SKILL_GOVERNANCE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if enable_governance:
        try:
            from houyi.core.skill.policy import PolicyEnforcer

            policy_enforcer = PolicyEnforcer()
            logger.debug("Initialized PolicyEnforcer for skill governance")
        except ImportError:
            logger.debug("PolicyEnforcer not available")

        try:
            from houyi.core.skill.consent import (
                ConsentManager,
                InMemoryConsentStore,
                PolicyBasedConsentHandler,
            )

            consent_store = InMemoryConsentStore()
            consent_handler = PolicyBasedConsentHandler(default_grant=False)
            consent_manager = ConsentManager(
                store=consent_store,
                handler=consent_handler,
                interactive=True,
            )
            logger.debug("Initialized ConsentManager for UI consent flow")
        except ImportError:
            logger.debug("ConsentManager components not available")

    # Create and set the global SkillService
    skill_service = SkillService(
        registry=DEFAULT_SKILL_REGISTRY,
        metrics_store=metrics_store,
        policy_enforcer=policy_enforcer,
        consent_manager=consent_manager,
    )
    set_skill_service(skill_service)
    logger.debug("SkillService initialized with Console integration")
