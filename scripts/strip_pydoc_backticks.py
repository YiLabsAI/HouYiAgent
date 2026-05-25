"""Strip RST and Markdown backtick markup from .py docstrings and comments.

Enforces agent.md L420-465: Python docstrings and inline # comments must
be plain prose. Three forms of inline markup are stripped:
  - RST cross-reference roles such as :class:, :func:, :meth:, :mod:,
    :attr:, :data:, :exc:, :obj:, :ref:
  - RST double-backtick inline code spans
  - Markdown single-backtick inline code spans

The script is conservative:
- Only modifies module / class / function / method docstrings AND
  inline comments. All other string literals (SQL, shell commands,
  regexes, user-facing markdown printed back to the terminal) are left
  intact.
- Locates docstrings via AST; locates comments via tokenize.
- Applies edits by character offset in reverse order so offsets stay
  stable.
- Skips files it cannot parse (syntax errors).
- Dry-run by default: pass --apply to actually write.

Usage:
    python scripts/strip_pydoc_backticks.py [--apply] [paths...]
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

# RST cross-reference roles followed by single-backticked target.
# Foo, bar.baz, ..., x, ...,
# ..., ..., ..., ...
_ROLE_RE = re.compile(r":(?:class|func|meth|mod|attr|data|exc|obj|ref):`([^`]+?)`", re.DOTALL)
# RST double-backtick inline code: foo -> foo
_DOUBLE_BT_RE = re.compile(r"``([^`]+?)``", re.DOTALL)
# Markdown single-backtick inline code: foo -> foo. Run AFTER double-bt
# so we never strip the inner pair of an unmatched triple. Anchored on
# both sides to avoid eating an unbalanced backtick.
_SINGLE_BT_RE = re.compile(r"`([^`]+?)`", re.DOTALL)


def clean_text(text: str) -> str:
    out = _ROLE_RE.sub(r"\1", text)
    out = _DOUBLE_BT_RE.sub(r"\1", out)
    out = _SINGLE_BT_RE.sub(r"\1", out)
    return out


def _line_col_to_offset(source: str, line: int, col: int) -> int:
    # ast lineno is 1-based; col_offset is 0-based byte offset on that line.
    pos = 0
    for _ in range(line - 1):
        pos = source.index("\n", pos) + 1
    return pos + col


def _collect_docstring_spans(source: str, tree: ast.AST) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        # ast lineno/col_offset point at the start of the docstring expr;
        # end_lineno/end_col_offset point one past its end.
        start = _line_col_to_offset(source, value.lineno, value.col_offset)
        end = _line_col_to_offset(source, value.end_lineno, value.end_col_offset)
        spans.append((start, end))
    return spans


def _collect_comment_spans(source: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        start = _line_col_to_offset(source, tok.start[0], tok.start[1])
        end = _line_col_to_offset(source, tok.end[0], tok.end[1])
        spans.append((start, end))
    return spans


def process_file(path: Path, *, apply: bool) -> tuple[int, int]:
    """Return (file_changed, total_replacements)."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0, 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"skip (syntax error): {path}", file=sys.stderr)
        return 0, 0

    spans = _collect_docstring_spans(source, tree) + _collect_comment_spans(source)
    if not spans:
        return 0, 0

    # Apply edits in reverse order so offsets remain valid.
    spans.sort(key=lambda s: s[0], reverse=True)
    new_source = source
    replacements = 0
    for start, end in spans:
        original = new_source[start:end]
        cleaned = clean_text(original)
        if cleaned == original:
            continue
        replacements += 1
        new_source = new_source[:start] + cleaned + new_source[end:]

    if not replacements:
        return 0, 0

    if apply:
        path.write_text(new_source, encoding="utf-8")
    return 1, replacements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=["houyi", "houyi-studio", "scripts", "tests"],
    )
    parser.add_argument("--apply", action="store_true", help="actually write changes")
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(
                f
                for f in p.rglob("*.py")
                if "/.venv/" not in str(f) and "/__pycache__/" not in str(f)
            )

    files_changed = 0
    total_replacements = 0
    for f in files:
        c, r = process_file(f, apply=args.apply)
        if c:
            files_changed += c
            total_replacements += r
            print(f"{'patched' if args.apply else 'would patch'}: {f} ({r} regions)")

    suffix = "(applied)" if args.apply else "(dry-run; pass --apply to write)"
    print(f"\n{files_changed} files, {total_replacements} replacement regions {suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
