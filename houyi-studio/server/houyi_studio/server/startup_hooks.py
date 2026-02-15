"""Startup hooks for console server.

Registers built-in skills for the console runtime.
User/custom skills will be loaded through a unified mechanism (SimpleSkill project).
"""

from __future__ import annotations

import logging
import os

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY

from .skill_service import SkillService, set_skill_service

logger = logging.getLogger(__name__)


def register_console_skills() -> None:
    """Register built-in skills for console runtime.

    Registers the following skills:
    - Web search: web_search
    - Weather: get_date, get_weather, get_weather_live
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

        skill = build_web_search_skill()
        DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)
        registered_skills.append(skill.name)
    except ImportError as e:
        logger.warning("Web search skill not available: %s", e)

    # 2. Weather tools (each @tool is a SkillSpec)
    try:
        from houyi.skills.weather import get_date, get_weather, get_weather_live

        DEFAULT_SKILL_REGISTRY.register(get_date, overwrite=True)
        DEFAULT_SKILL_REGISTRY.register(get_weather, overwrite=True)
        DEFAULT_SKILL_REGISTRY.register(get_weather_live, overwrite=True)
        registered_skills.extend(["get_date", "get_weather", "get_weather_live"])
    except ImportError as e:
        logger.warning("Weather skills not available: %s", e)

    # 3. Location tool
    try:
        from houyi.skills.location import get_location

        DEFAULT_SKILL_REGISTRY.register(get_location, overwrite=True)
        registered_skills.append("get_location")
    except ImportError as e:
        logger.warning("Location skill not available: %s", e)

    # 4. RAG skills (kb-search, kb-ingest, kb-graph, kb-analyze)
    try:
        from houyi.rag.skills.kb_search import kb_search_skill

        DEFAULT_SKILL_REGISTRY.register(kb_search_skill, overwrite=True)
        registered_skills.append(kb_search_skill.name)
    except ImportError as e:
        logger.debug("RAG kb-search skill not available: %s", e)

    try:
        from houyi.rag.skills.kb_ingest import kb_ingest_skill

        DEFAULT_SKILL_REGISTRY.register(kb_ingest_skill, overwrite=True)
        registered_skills.append(kb_ingest_skill.name)
    except ImportError as e:
        logger.debug("RAG kb-ingest skill not available: %s", e)

    try:
        from houyi.rag.skills.kb_graph import kb_graph_skill

        DEFAULT_SKILL_REGISTRY.register(kb_graph_skill, overwrite=True)
        registered_skills.append(kb_graph_skill.name)
    except ImportError as e:
        logger.debug("RAG kb-graph skill not available: %s", e)

    try:
        from houyi.rag.skills.kb_analyze import kb_analyze_skill

        DEFAULT_SKILL_REGISTRY.register(kb_analyze_skill, overwrite=True)
        registered_skills.append(kb_analyze_skill.name)
    except ImportError as e:
        logger.debug("RAG kb-analyze skill not available: %s", e)

    # 5. Planning skill
    try:
        from houyi.skills.planning import PlanningSkill

        planning_skill = PlanningSkill()
        planning_spec = planning_skill.to_spec()
        DEFAULT_SKILL_REGISTRY.register(planning_spec, overwrite=True)
        registered_skills.append(planning_spec.name)
    except ImportError as e:
        logger.debug("Planning skill not available: %s", e)

    # 6. External / community SKILL.md files from skills/ directory
    _load_external_skills(registered_skills)

    # Log summary
    total_skills = len(DEFAULT_SKILL_REGISTRY.list())
    logger.info(
        "Registered %d skills: %s",
        total_skills,
        ", ".join(registered_skills),
    )


def _load_external_skills(registered_skills: list[str]) -> None:
    """Load external/community SKILL.md files from the skills/ directory.

    Scans the project-root ``skills/`` directory for sub-directories that
    contain a ``SKILL.md`` file and registers each one.  Skills that are
    already registered (e.g. identical name to a built-in) are skipped
    unless they have a different name to avoid overwrites.
    """
    import pathlib

    # Resolve skills/ relative to project root (two levels up from this file)
    this_file = pathlib.Path(__file__).resolve()
    # startup_hooks.py is at houyi-studio/server/houyi_studio/server/startup_hooks.py
    # project root is 4 levels up
    project_root = this_file.parents[4]
    skills_dir = project_root / "skills"

    if not skills_dir.is_dir():
        logger.debug("External skills directory not found: %s", skills_dir)
        return

    loaded = DEFAULT_SKILL_REGISTRY.register_from_directory(
        skills_dir,
        pattern="SKILL.md",
        recursive=True,
        overwrite=False,
    )
    registered_skills.extend(loaded)
    if loaded:
        logger.info(
            "Loaded %d external skills from %s: %s", len(loaded), skills_dir, ", ".join(loaded)
        )


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
        logger.info("Initialized MetricsStore for skill metrics collection")
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
            logger.info("Initialized PolicyEnforcer for skill governance")
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
            logger.info("Initialized ConsentManager for UI consent flow")
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
    logger.info("SkillService initialized with Console integration")
