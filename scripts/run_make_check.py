from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhaseResult:
    name: str
    elapsed_s: float
    returncode: int | None = None
    timed_out: bool = False
    started: bool = False
    details: list[str] | None = None


def _print_summary(results: list[PhaseResult], total_s: float, budget_s: float) -> None:
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("make check timing summary")
    for result in results:
        status = "not_run"
        if result.timed_out:
            status = "timed_out"
        elif result.started and result.returncode == 0:
            status = "passed"
        elif result.started and result.returncode is not None:
            status = f"failed({result.returncode})"
        print(f"  - {result.name}: {result.elapsed_s:.2f}s [{status}]")
        if result.details:
            for detail in result.details:
                print(f"    - {detail}")
    print(f"  - total: {total_s:.2f}s")
    print(f"  - budget: {budget_s:.2f}s")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _load_phase_details(file_path: Path) -> list[str]:
    if not file_path.exists():
        return []
    details: list[str] = []
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("total:") or line.startswith("status:"):
            continue
        if ":" not in line:
            continue
        name, elapsed = line.split(":", 1)
        details.append(f"{name}: {elapsed}s")
    return details


def _run_phase(command: list[str], *, cwd: Path, timeout: float, env: dict[str, str]) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        # Always SIGKILL the entire group so child processes (Vite, uv,
        # uvicorn) that are still gracefully shutting down are reaped and
        # their ports released.  Without this, a budget-timeout during
        # e2e-smoke leaves orphan listeners on the e2e ports.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            process.wait(timeout=3)
        raise exc


def _run_phase_async(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    """Start a phase as a background process (for parallel execution)."""
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=60.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    phases: list[tuple[str, list[str]]] = [
        ("check-unit", ["./scripts/check_code.sh"]),
        ("check-integration", ["./scripts/check_integration.sh"]),
        ("check-e2e-smoke", ["make", "--no-print-directory", "test-e2e-smoke"]),
    ]

    started_at = time.perf_counter()
    results = [PhaseResult(name=name, elapsed_s=0.0) for name, _ in phases]

    try:
        # Phase 1: check-unit (must run first and succeed)
        unit_result = results[0]
        _, unit_command = phases[0]
        elapsed_before = time.perf_counter() - started_at
        remaining = args.budget - elapsed_before
        if remaining <= 0:
            unit_result.timed_out = True
            total_s = time.perf_counter() - started_at
            _print_summary(results, total_s, args.budget)
            print(f"✗ make check exceeded {args.budget:.0f}s", file=sys.stderr)
            return 124

        phase_started = time.perf_counter()
        unit_result.started = True
        env = os.environ.copy()
        timing_file = Path(tempfile.mkstemp(prefix="houyi-check-unit-", suffix=".timings")[1])
        env["HOUYI_CHECK_TIMING_FILE"] = str(timing_file)
        env["HOUYI_CHECK_SUPPRESS_SUMMARY"] = "1"
        try:
            unit_result.returncode = _run_phase(unit_command, cwd=root, timeout=remaining, env=env)
        except subprocess.TimeoutExpired:
            unit_result.timed_out = True
            unit_result.elapsed_s = time.perf_counter() - phase_started
            unit_result.details = _load_phase_details(timing_file)
            total_s = time.perf_counter() - started_at
            _print_summary(results, total_s, args.budget)
            print(
                f"✗ make check exceeded {args.budget:.0f}s during {unit_result.name}",
                file=sys.stderr,
            )
            return 124

        unit_result.elapsed_s = time.perf_counter() - phase_started
        unit_result.details = _load_phase_details(timing_file)
        if unit_result.returncode != 0:
            total_s = time.perf_counter() - started_at
            _print_summary(results, total_s, args.budget)
            print(f"✗ make check failed in {unit_result.name}", file=sys.stderr)
            return unit_result.returncode

        # Phases 2 & 3: run check-integration and check-e2e-smoke in parallel
        elapsed_before = time.perf_counter() - started_at
        remaining = args.budget - elapsed_before
        if remaining <= 0:
            for r in results[1:]:
                r.timed_out = True
            total_s = time.perf_counter() - started_at
            _print_summary(results, total_s, args.budget)
            print(f"✗ make check exceeded {args.budget:.0f}s", file=sys.stderr)
            return 124

        parallel_started = time.perf_counter()
        int_result = results[1]
        e2e_result = results[2]
        _, int_command = phases[1]
        _, e2e_command = phases[2]

        int_result.started = True
        e2e_result.started = True
        int_env = os.environ.copy()
        e2e_env = os.environ.copy()

        # Suppress check_code.sh's own summary for integration phase
        int_env["HOUYI_CHECK_SUPPRESS_SUMMARY"] = "1"

        int_process = _run_phase_async(int_command, cwd=root, env=int_env)
        e2e_process = _run_phase_async(e2e_command, cwd=root, env=e2e_env)

        # Wait for both, with budget timeout
        failed_phase: str | None = None
        for process, result in [(int_process, int_result), (e2e_process, e2e_result)]:
            try:
                result.returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                result.timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    process.wait(timeout=3)
                if failed_phase is None:
                    failed_phase = result.name

            result.elapsed_s = time.perf_counter() - parallel_started

        # Check results of parallel phases
        for result in [int_result, e2e_result]:
            if result.timed_out:
                total_s = time.perf_counter() - started_at
                _print_summary(results, total_s, args.budget)
                print(
                    f"✗ make check exceeded {args.budget:.0f}s during {result.name}",
                    file=sys.stderr,
                )
                return 124
            if result.returncode != 0:
                total_s = time.perf_counter() - started_at
                _print_summary(results, total_s, args.budget)
                print(f"✗ make check failed in {result.name}", file=sys.stderr)
                return result.returncode

        total_s = time.perf_counter() - started_at
        if total_s > args.budget:
            _print_summary(results, total_s, args.budget)
            print(f"✗ make check exceeded {args.budget:.0f}s", file=sys.stderr)
            return 124

        _print_summary(results, total_s, args.budget)
        print("✓ make check passed")
        return 0
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
