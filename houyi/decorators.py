"""Decorators for HouYi framework."""

import inspect
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, create_model

from houyi.core.skill import SkillSpec


def tool(func: Callable) -> SkillSpec:
    """Decorator to convert a function into a SkillSpec.
    
    Automatically infers input/output schemas from type hints.
    
    Usage:
        @tool
        def search(query: str) -> list[str]:
            '''Search the web for information.'''
            return ["result1", "result2"]
    
    Args:
        func: Function to convert to a skill
        
    Returns:
        SkillSpec instance
    """
    # Get function metadata
    name = func.__name__
    description = func.__doc__ or f"Execute {name}"
    description = description.strip()
    
    # Get type hints
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    
    # Build input schema from parameters
    input_fields = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        
        param_type = hints.get(param_name, str)
        default = ... if param.default == inspect.Parameter.empty else param.default
        input_fields[param_name] = (param_type, default)
    
    input_schema = create_model(f"{name.title()}Input", **input_fields) if input_fields else type("EmptyInput", (BaseModel,), {})
    
    # Build output schema from return type
    return_type = hints.get("return", str)
    output_schema = create_model(f"{name.title()}Output", result=(return_type, ...))
    
    # Create SkillSpec
    skill = SkillSpec(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        executor=func,
    )
    
    # Attach original function for direct calling
    skill._original_func = func
    
    return skill
