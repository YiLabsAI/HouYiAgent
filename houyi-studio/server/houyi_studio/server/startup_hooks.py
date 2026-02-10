"""Startup hooks for console server.

Registers built-in skills for the console runtime.
User/custom skills will be loaded through a unified mechanism (SimpleSkill project).
"""

from __future__ import annotations

import logging

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY

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

    # 4. RAG kb-search skill
    try:
        from houyi.rag.skills.kb_search import kb_search_skill

        DEFAULT_SKILL_REGISTRY.register(kb_search_skill, overwrite=True)
        registered_skills.append(kb_search_skill.name)
    except ImportError as e:
        logger.debug("RAG kb-search skill not available: %s", e)

    # Log summary
    total_skills = len(DEFAULT_SKILL_REGISTRY.list())
    logger.info(
        "Registered %d skills: %s",
        total_skills,
        ", ".join(registered_skills),
    )
