"""Unit tests for scripts/check_style_rules.py.

Focus on the gates that are easy to regress silently: underscore-segment
counting, the 35/45 character test-name warn thresholds, CJK-source
rejection, and the scripts/ -h/--help requirement.
"""

# The test fixtures below intentionally embed short \uXXXX CJK escapes to
# exercise the raw-CJK and CJK-escape gates; do not treat them as prose.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_style_rules.py"


@pytest.fixture(scope="module")
def style_rules():
    spec = importlib.util.spec_from_file_location("_style_rules_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(style_rules, path: Path, text: str):
    return style_rules._check_file(path, text)


class TestUnderscoreSegmentGate:
    def test_four_tail_underscores_passes(self, style_rules):
        src = "def test_alpha_beta_gamma_delta_epsilon():\n    pass\n"
        errs, warns = _run(style_rules, Path("tests/x/test_mod.py"), src)
        assert errs == []
        assert warns == []

    def test_five_tail_underscores_fails(self, style_rules):
        src = "def test_alpha_beta_gamma_delta_epsilon_zeta():\n    pass\n"
        errs, _ = _run(style_rules, Path("tests/x/test_mod.py"), src)
        assert len(errs) == 1
        assert "underscore segments" in errs[0]
        assert "limit 4" in errs[0]

    def test_async_def_is_covered(self, style_rules):
        src = "async def test_a_b_c_d_e_f():\n    pass\n"
        errs, _ = _run(style_rules, Path("tests/x/test_mod.py"), src)
        assert len(errs) == 1

    def test_non_tests_skips_segment_gate(self, style_rules):
        src = "def test_a_b_c_d_e_f():\n    pass\n"
        errs, _ = _run(style_rules, Path("houyi/foo.py"), src)
        assert errs == []


class TestCharacterLengthWarn:
    def test_tail_exactly_35_ok(self, style_rules):
        tail = "a" * 35
        src = f"def test_{tail}():\n    pass\n"
        _, warns = _run(style_rules, Path("tests/x/test_mod.py"), src)
        assert warns == []

    def test_tail_36_soft_warn(self, style_rules):
        tail = "a" * 36
        src = f"def test_{tail}():\n    pass\n"
        errs, warns = _run(style_rules, Path("tests/x/test_mod.py"), src)
        assert errs == []
        assert len(warns) == 1
        assert "prefer" in warns[0]
        assert "hard limit 45" in warns[0]

    def test_tail_46_hard_warn(self, style_rules):
        tail = "a" * 46
        src = f"def test_{tail}():\n    pass\n"
        errs, warns = _run(style_rules, Path("tests/x/test_mod.py"), src)
        assert errs == []
        assert len(warns) == 1
        assert "exceeds hard limit 45" in warns[0]

    def test_length_warn_skips_non_tests(self, style_rules):
        tail = "a" * 60
        src = f"def test_{tail}():\n    pass\n"
        _, warns = _run(style_rules, Path("houyi/foo.py"), src)
        assert warns == []


class TestRawCjkGate:
    def test_raw_cjk_in_python_fails(self, style_rules):
        src = 'name = "\u4e2d\u6587"\n'
        errs, _ = _run(style_rules, Path("houyi/foo.py"), src)
        assert len(errs) == 1
        assert "raw CJK character" in errs[0]

    def test_unicode_escape_in_python_passes(self, style_rules):
        src = 'name = "\\u4e2d\\u6587"\n# gloss: zhong wen\n'
        errs, warns = _run(style_rules, Path("houyi/foo.py"), src)
        assert errs == []
        assert warns == []


class TestCjkEscapeWarn:
    def test_five_escapes_no_comment_warn(self, style_rules):
        src = 'a = "\\u4e00\\u4e01\\u4e02\\u4e03\\u4e04\\u4e05"\n'
        _, warns = _run(style_rules, Path("houyi/foo.py"), src)
        assert len(warns) == 1
        assert "CJK escapes" in warns[0]

    def test_escapes_with_comment_silent(self, style_rules):
        src = '# gloss: numerals\na = "\\u4e00\\u4e01\\u4e02\\u4e03\\u4e04\\u4e05"\n'
        _, warns = _run(style_rules, Path("houyi/foo.py"), src)
        assert warns == []


class TestScriptsHelpGate:
    def test_script_without_help_fails(self, style_rules):
        src = "print('hi')\n"
        errs, _ = _run(style_rules, Path("scripts/new_tool.py"), src)
        assert len(errs) == 1
        assert "-h/--help" in errs[0]

    def test_script_with_argparse_passes(self, style_rules):
        src = "import argparse\nparser = argparse.ArgumentParser()\n"
        errs, _ = _run(style_rules, Path("scripts/new_tool.py"), src)
        assert errs == []

    def test_shell_script_help_branch_ok(self, style_rules):
        src = 'if [[ "$1" == "-h" ]]; then echo help; fi\n'
        errs, _ = _run(style_rules, Path("scripts/new_tool.sh"), src)
        assert errs == []

    def test_non_script_file_skips_help(self, style_rules):
        src = "print('hi')\n"
        errs, _ = _run(style_rules, Path("houyi/foo.py"), src)
        assert errs == []

    def test_tests_scripts_init_skips_gate(self, style_rules):
        # tests/scripts/__init__.py has a parent named "scripts" but it is
        # a test fixture package, not a CLI entrypoint; the gate must not
        # demand -h/--help support from it.
        errs, _ = _run(style_rules, Path("tests/scripts/__init__.py"), "")
        assert errs == []
