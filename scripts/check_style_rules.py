#!/usr/bin/env python3
"""Style rules gate: CJK-free source, short test names, CJK escape readability.

Runs four checks against the Python / shell files passed on the command line.
Designed to be called from ``scripts/check_code.sh`` and
``scripts/quick_check.sh`` with the changed-files set so the gate stays cheap
in normal developer workflows (it only sees what ``git diff`` emits).

Usage
-----

    scripts/check_style_rules.py [--exclude GLOB] [PATH ...]

Pass ``--exclude`` (repeatable) to skip paths that should legitimately contain
CJK content — for example a future ``corpora/zh/`` fixture directory. Globs are
matched against the ``fnmatch`` pattern of the full path string, so both
``corpora/zh/*`` and ``**/zh_fixtures/**`` work.

Checks (in order)
-----------------

1. **ERROR** — source code must stay ASCII/latin-only. Any raw CJK character in
   a ``.py`` file fails; use ``\\uXXXX`` escapes for CJK literals instead.
   Rationale: per ``agent.md`` "no Chinese in code" (dai-ma-wu-zhong-wen)
   and B3-106 in ``docs/design/deep-research-acceptance.md``.
2. **ERROR** — test function names must stay short. Any ``def test_<name>``
   under ``tests/`` fails when ``<name>`` (the portion AFTER the leading
   ``test_``) contains more than 3 literal underscore characters. The check
   counts underscores in the captured tail only; the ``test_`` prefix itself
   is never included.

   Mapping onto ``agent.md`` §Test Function Naming (examples use
   ``<name>`` = ``alpha_beta_...``):

   - ``test_alpha_beta`` — 2 tail segments, 1 underscore: **preferred**.
   - ``test_alpha_beta_gamma`` — 3 tail segments, 2 underscores: preferred.
   - ``test_alpha_beta_gamma_delta`` — 4 tail segments, 3 underscores: at the
     hard cap. Passes, but reviewers should try to trim further.
   - ``test_alpha_beta_gamma_delta_epsilon`` — 5 tail segments,
     4 underscores: **FAILS**. Shorten the name or push scenario detail
     into a test class or docstring.

   A **WARN** also fires when the captured tail length exceeds 35 characters
   (soft preference) or 45 characters (hard limit per ``agent.md``). The
   45-char ceiling is also a hard expectation; it is kept as WARN rather
   than ERROR so borderline names surface without blocking otherwise clean
   commits. Treat any WARN as a rename opportunity in the same patch.
3. **WARN**  — files that embed 5+ ``\\uXXXX`` CJK escapes but carry no ASCII
   ``#`` comment anywhere in the file emit a warning so reviewers remember to
   add a short pinyin or English gloss.
4. **ERROR** — new ``scripts/*.{sh,py}`` entrypoints must ship ``-h`` /
   ``--help`` support. Enforces ``scripts/README.md`` L13 so adding a script
   does not silently drift from the self-describing-entrypoint convention.

Exit codes: 0 when there are no errors, 1 otherwise. Warnings print but never
fail the gate.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

# CJK ranges covering Han, extended Han, Hangul, Kana, and full-width punctuation.
_CJK_CHAR_RE = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
# \uXXXX escape sequences whose codepoint lands in the CJK-ish ranges above.
_CJK_ESCAPE_RE = re.compile(
    r"\\u(2e[89a-f][0-9a-f]|[3-9][0-9a-f]{3}|ac[0-9a-f]{2}|ad[0-9a-f]{2}|"
    r"a[ef][0-9a-f]{2}|b[0-9a-f]{3}|c[0-9a-f]{3}|d[0-7][0-9a-f]{2}|"
    r"f9[0-9a-f]{2}|fa[0-9a-f]{2}|ff[0-9a-f]{2})",
    re.IGNORECASE,
)
_TEST_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_([A-Za-z0-9_]+)\s*\(")
_ASCII_COMMENT_RE = re.compile(r"^\s*#\s")
# Any of these tokens anywhere in the file indicates the script supports
# ``-h`` / ``--help`` either directly (shell ``"-h"`` branches) or via argparse.
_HELP_SUPPORT_RE = re.compile(r'--help|"-h"|"\-\-help"|argparse|ArgumentParser|add_help')

_MAX_TEST_UNDERSCORES = 3
# Character-length limits for the captured tail of a test function name
# (everything after the leading ``test_``).  Matches ``agent.md`` §Test
# Function Naming: names should normally fit ~35 chars, and 45 is the hard
# cap.  The WARN threshold lets reviewers nudge borderline names before the
# next developer piles on a longer neighbour.
_TEST_NAME_WARN_CHARS = 35
_TEST_NAME_MAX_CHARS = 45
_CJK_ESCAPE_WARN_THRESHOLD = 5


def _check_file(path: Path, text: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    is_python = path.suffix == ".py"
    is_test_file = is_python and "tests" in path.parts
    # Only the repo-level ``scripts/`` directory holds true entrypoints.
    # A test fixture path such as ``tests/scripts/__init__.py`` must not
    # trip this gate just because its parent directory happens to be
    # named ``scripts``.  Empty ``__init__.py`` files are also never
    # entrypoints by convention.
    is_script_entrypoint = (
        path.parent.name == "scripts"
        and path.suffix in {".sh", ".py"}
        and path.name != "__init__.py"
        and "tests" not in path.parts
    )
    cjk_escape_count = 0
    has_ascii_comment = False

    if is_python:
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _CJK_CHAR_RE.search(line):
                errors.append(
                    f"{path}:{lineno}: raw CJK character in source; "
                    f"use \\uXXXX escape instead (agent.md 'no Chinese in code')."
                )

            if is_test_file:
                match = _TEST_FUNC_RE.match(line)
                if match:
                    tail = match.group(1)
                    underscores = tail.count("_")
                    if underscores > _MAX_TEST_UNDERSCORES:
                        errors.append(
                            f"{path}:{lineno}: test_{tail} has "
                            f"{underscores} underscore segments after 'test_' "
                            f"(limit {_MAX_TEST_UNDERSCORES}). Shorten the name "
                            f"or push scenario details into a test class/docstring."
                        )
                    tail_chars = len(tail)
                    if tail_chars > _TEST_NAME_MAX_CHARS:
                        warnings.append(
                            f"{path}:{lineno}: test_{tail} tail is "
                            f"{tail_chars} characters (exceeds hard limit "
                            f"{_TEST_NAME_MAX_CHARS}). Shorten the name or "
                            f"move scenario detail into a docstring."
                        )
                    elif tail_chars > _TEST_NAME_WARN_CHARS:
                        warnings.append(
                            f"{path}:{lineno}: test_{tail} tail is "
                            f"{tail_chars} characters (prefer "
                            f"<= {_TEST_NAME_WARN_CHARS}, hard limit "
                            f"{_TEST_NAME_MAX_CHARS})."
                        )

            cjk_escape_count += len(_CJK_ESCAPE_RE.findall(line))
            if _ASCII_COMMENT_RE.match(line):
                has_ascii_comment = True

        if cjk_escape_count >= _CJK_ESCAPE_WARN_THRESHOLD and not has_ascii_comment:
            warnings.append(
                f"{path}:0: {cjk_escape_count} \\uXXXX CJK escapes but no ASCII "
                f"comment in the file — add a short pinyin or English gloss so "
                f"future readers know what the literals mean."
            )

    if is_script_entrypoint and not _HELP_SUPPORT_RE.search(text):
        errors.append(
            f"{path}:0: scripts/ entrypoint has no -h/--help support "
            f"(see scripts/README.md L13). Add an argparse parser or a "
            f"``-h|--help`` branch."
        )

    return errors, warnings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_style_rules.py",
        description=(
            "Style gate for HouYi: rejects raw CJK in source, over-long test "
            "names, and scripts/ entrypoints missing -h/--help; warns on "
            "dense \\uXXXX CJK escapes without an ASCII comment."
        ),
        epilog=(
            "Typically invoked with the git-changed file set. Pass --exclude "
            "to skip directories that legitimately contain CJK content "
            "(e.g. Chinese test fixtures)."
        ),
    )
    parser.add_argument("paths", nargs="*", help="Files to check (.py or .sh).")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Glob (fnmatch) pattern against the full path string; may be "
            "given multiple times. Matched paths are skipped."
        ),
    )
    return parser


def _should_check(path: Path, excludes: list[str]) -> bool:
    if path.suffix not in {".py", ".sh"} or not path.is_file():
        return False
    path_str = str(path)
    return not any(fnmatch.fnmatch(path_str, pattern) for pattern in excludes)


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    files = [Path(p) for p in args.paths]
    files = [p for p in files if _should_check(p, args.exclude)]
    if not files:
        return 0

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            all_errors.append(f"{path}:0: cannot read file ({exc})")
            continue
        errs, warns = _check_file(path, text)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    for warning in all_warnings:
        print(f"warning: {warning}")
    for error in all_errors:
        print(f"error: {error}", file=sys.stderr)

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
