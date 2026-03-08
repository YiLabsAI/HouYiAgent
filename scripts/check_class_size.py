#!/usr/bin/env python3
"""Check Python class definition size by line count."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassSizeViolation:
    file_path: Path
    class_name: str
    start_line: int
    end_line: int
    line_count: int


def _iter_class_violations(file_path: Path, min_lines: int) -> list[ClassSizeViolation]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[ClassSizeViolation] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._stack.append(node.name)
            end_line = getattr(node, "end_lineno", None)
            if end_line is not None:
                line_count = end_line - node.lineno + 1
                if line_count >= min_lines:
                    violations.append(
                        ClassSizeViolation(
                            file_path=file_path,
                            class_name=".".join(self._stack),
                            start_line=node.lineno,
                            end_line=end_line,
                            line_count=line_count,
                        )
                    )
            self.generic_visit(node)
            self._stack.pop()

    _Visitor().visit(tree)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check class line-count limits")
    parser.add_argument("files", nargs="*", help="Python files to inspect")
    parser.add_argument("--warn-lines", type=int, default=500)
    parser.add_argument("--error-lines", type=int, default=800)
    args = parser.parse_args()

    if args.warn_lines <= 0 or args.error_lines <= 0:
        print("[ClassSizeGate] --warn-lines and --error-lines must be > 0")
        return 2
    if args.warn_lines >= args.error_lines:
        print("[ClassSizeGate] --warn-lines must be < --error-lines")
        return 2

    files = [Path(p) for p in args.files if p.endswith(".py")]
    if not files:
        return 0

    violations: list[ClassSizeViolation] = []
    for file_path in files:
        if not file_path.exists() or not file_path.is_file():
            continue
        violations.extend(_iter_class_violations(file_path, args.warn_lines))

    if not violations:
        return 0

    warnings = [
        item for item in violations if args.warn_lines <= item.line_count < args.error_lines
    ]
    errors = [item for item in violations if item.line_count >= args.error_lines]

    if warnings:
        print(
            f"[ClassSizeGate] WARNING: {len(warnings)} class(es) in "
            f"[{args.warn_lines}, {args.error_lines}) lines:"
        )
        for item in sorted(warnings, key=lambda v: (str(v.file_path), v.start_line)):
            print(
                f"  - {item.file_path}:{item.start_line}-{item.end_line} "
                f"{item.class_name} ({item.line_count} lines)"
            )

    if not errors:
        return 0

    print(f"[ClassSizeGate] ERROR: {len(errors)} class(es) >= {args.error_lines} lines:")
    for item in sorted(errors, key=lambda v: (str(v.file_path), v.start_line)):
        print(
            f"  - {item.file_path}:{item.start_line}-{item.end_line} "
            f"{item.class_name} ({item.line_count} lines)"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
