"""Memory type classifier.

Classifies a MemoryCandidate into one of the MemoryType categories
using rule-based pattern matching.

LLM-based classification can be added as a fallback in a future
iteration — the interface is async to allow that path.
"""

from __future__ import annotations

import re

from houyi.adapters.memory.types import MemoryCandidate, MemoryType

_PROFILE_PATTERNS = re.compile(
    r"(?i)\b(my name is|i am|i'm|my age|my role|i work|my job|my title|"
    r"i live|my location|my email|my phone|user name)\b"
)
_PREFERENCE_PATTERNS = re.compile(
    r"(?i)\b(i prefer|i like|i enjoy|i hate|i dislike|i want|"
    r"i love|my favorite|my preferred|i'd rather|i always use)\b"
)
_PROCEDURE_PATTERNS = re.compile(
    r"(?i)\b(step \d|first .+ then|workflow|process|how to|procedure|"
    r"recipe|instructions|guide|tutorial)\b"
)
_CONSTRAINT_PATTERNS = re.compile(
    r"(?i)\b(don't|do not|never|avoid|stop|no more|exclude|"
    r"not allowed|forbidden|banned|must not|should not)\b"
)
_PROJECT_PATTERNS = re.compile(
    r"(?i)\b(project|repo|codebase|workspace|sprint|milestone|"
    r"deployment|architecture|tech stack|framework)\b"
)


class MemoryClassifier:
    """Rule-based memory type classifier.

    Classification priority (first match wins):
    1. CONSTRAINT — explicit user restrictions
    2. PROFILE — identity information
    3. PREFERENCE — likes / dislikes
    4. PROCEDURE — step-by-step instructions
    5. PROJECT — workspace / project context
    6. FACT — default fallback
    """

    async def classify(self, candidate: MemoryCandidate) -> MemoryType:
        """Classify a candidate into a memory type. Target p95 < 20ms."""
        text = candidate.content

        if _CONSTRAINT_PATTERNS.search(text):
            return MemoryType.CONSTRAINT
        if _PROFILE_PATTERNS.search(text):
            return MemoryType.PROFILE
        if _PREFERENCE_PATTERNS.search(text):
            return MemoryType.PREFERENCE
        if _PROCEDURE_PATTERNS.search(text):
            return MemoryType.PROCEDURE
        if _PROJECT_PATTERNS.search(text):
            return MemoryType.PROJECT
        return MemoryType.FACT
