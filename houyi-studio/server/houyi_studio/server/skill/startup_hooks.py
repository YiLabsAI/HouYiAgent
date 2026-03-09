"""Startup hooks for console server.

Registers built-in skills for the console runtime.
User/custom skills will be loaded through a unified mechanism (SimpleSkill project).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY, CoreToolProtectionError
from houyi.infrastructure.config import env

from .paths import resolve_managed_skills_dir
from .service import SkillService, set_skill_service

logger = logging.getLogger(__name__)
_EXTERNAL_PREFIX = "ext__"
_SKILL_FILE_PATTERNS = ("SKILL.md", "skill.md")


def _entry_file_priority(path: Path) -> int:
    name = path.name
    if name == "SKILL.md":
        return 2
    if name == "skill.md":
        return 1
    return 0


def _iter_external_skill_files(skills_dir: Path) -> list[Path]:
    """Return unique SKILL.md files, including those under symlinked package dirs."""
    scan_roots: list[Path] = [skills_dir]
    for child in skills_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            scan_roots.append(child.resolve())
        except Exception:
            scan_roots.append(child)

    files: list[Path] = []
    seen: set[Path] = set()
    for root in scan_roots:
        for pattern in _SKILL_FILE_PATTERNS:
            for path in root.rglob(pattern):
                if not path.is_file():
                    continue
                try:
                    resolved = path.resolve()
                except Exception:
                    resolved = path
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(resolved)

    # Keep one canonical entry file per package dir. Prefer SKILL.md over skill.md.
    selected_by_dir: dict[Path, Path] = {}
    for skill_file in files:
        package_dir = skill_file.parent
        existing = selected_by_dir.get(package_dir)
        if existing is None or _entry_file_priority(skill_file) > _entry_file_priority(existing):
            selected_by_dir[package_dir] = skill_file

    return sorted(selected_by_dir.values(), key=lambda p: str(p))


def _read_declared_skill_name(skill_path: Path) -> str | None:
    """Extract declared skill name from a skill file."""
    from houyi.domain.skill.spec import SkillSpec

    try:
        spec = SkillSpec.from_file(str(skill_path))
    except Exception:
        return None

    name = getattr(spec, "name", "")
    if not isinstance(name, str):
        return None
    normalized = name.strip()
    return normalized or None


def _default_startup_skills_dir() -> Path:
    return resolve_managed_skills_dir()


def _resolve_external_skill_scan_dirs() -> list[Path]:
    configured = (env.startup_skills_dir or "").strip()
    primary = Path(configured).expanduser() if configured else _default_startup_skills_dir()
    dirs: list[Path] = [primary]

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


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


def _should_inherit_core_runtime(external: object, core: object) -> bool:
    external_hooks = getattr(external, "hooks", None) or []
    core_hooks = getattr(core, "hooks", None) or []
    if external_hooks and not core_hooks:
        return False
    return bool(core_hooks) or getattr(core, "runtime_contract", None) is not None


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
        from houyi.domain.skill.runtime_resolver import RuntimeResolver

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

        from houyi.domain.skill.runtime_contract import CapabilityTier

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
        if _should_inherit_core_runtime(external, core):
            updates["hooks"] = list(getattr(core, "hooks", None) or [])
            updates["skill_dir"] = getattr(core, "skill_dir", None)
            updates["skill_md_path"] = getattr(core, "skill_md_path", None)
            updates["runtime_contract"] = getattr(core, "runtime_contract", None)

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
        from houyi.skills.web_search.skill import build_web_search_skill

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

    # 6. Built-in local tools
    try:
        from houyi.skills.builtin.local_tools import register_builtin_local_tools

        for tool_name in register_builtin_local_tools(DEFAULT_SKILL_REGISTRY):
            if tool_name not in registered_skills:
                registered_skills.append(tool_name)
    except ImportError as e:
        logger.warning("Built-in local tools not available: %s", e)

    # 7. External / community SKILL.md files from skills/ directory
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
    """Load external/community SKILL.md files from managed startup scan dirs.

    Default scan path resolves to the managed ``.houyi/skills`` directory.
    ``HOUYI_STARTUP_SKILLS_DIR`` can override it explicitly.
    """
    for skills_dir in _resolve_external_skill_scan_dirs():
        if not skills_dir.is_dir():
            logger.debug("External skills directory not found: %s", skills_dir)
            continue

        discovered_names = _discover_external_skill_names(skills_dir)
        loaded: list[str] = []
        loaded_name_sources: dict[str, Path] = {}
        for skill_file in _iter_external_skill_files(skills_dir):
            declared_name = _read_declared_skill_name(skill_file)
            if declared_name and declared_name in loaded_name_sources:
                logger.info(
                    "Skip duplicate external skill '%s' from %s (already loaded from %s)",
                    declared_name,
                    skill_file,
                    loaded_name_sources[declared_name],
                )
                continue
            try:
                loaded_name = DEFAULT_SKILL_REGISTRY.register_from_skill_file(
                    str(skill_file),
                    overwrite=False,
                )
                if loaded_name:
                    loaded.append(loaded_name)
                    canonical_name = declared_name or loaded_name.removeprefix(_EXTERNAL_PREFIX)
                    loaded_name_sources.setdefault(canonical_name, skill_file)
            except CoreToolProtectionError:
                continue
            except Exception as exc:
                logger.warning("Failed loading external skill from %s: %s", skill_file, exc)

        registered_skills.extend(loaded)
        pruned = _prune_stale_external_skills(skills_dir, discovered_names)
        if pruned:
            logger.info("Pruned %d stale external skills: %s", len(pruned), ", ".join(pruned))
        if loaded:
            logger.info("Loaded %d external skills from %s", len(loaded), skills_dir)
            logger.debug("External skills: %s", ", ".join(loaded))


def _discover_external_skill_names(skills_dir) -> set[str]:
    """Discover declared skill names from SKILL.md files under *skills_dir*."""
    from houyi.domain.skill.spec import SkillSpec

    discovered: set[str] = set()
    for skill_path in _iter_external_skill_files(skills_dir):
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
        from houyi.domain.skill.metrics import InMemoryMetricsStore

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
            from houyi.domain.skill.policy import PolicyEnforcer

            policy_enforcer = PolicyEnforcer()
            logger.debug("Initialized PolicyEnforcer for skill governance")
        except ImportError:
            logger.debug("PolicyEnforcer not available")

        try:
            from houyi.domain.skill.consent import (
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
