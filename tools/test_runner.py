#!/usr/bin/env python3
"""run_tests — detect the project's test runner, run it SAFELY, return a STRUCTURED result.

Closes the "tests are manual" gap: instead of `terminal("pytest …")` + regex-scraping stdout,
the agent gets `{ok, runner, passed, failed, errors, total, status, exit_code, failures:[…],
raw_tail}`. Detects pytest / jest / vitest / go test by project markers (walk-up, like
``lint_extras``), runs via the SHARED backend exec (``file_tools._get_file_ops(task_id)`` →
``fops._exec`` — so docker/ssh/modal still work; NO parallel subprocess), and parses the output.

Safety (all required): refuses when a test HARNESS is active (``PYTEST_CURRENT_TEST`` /
``HERMES_ISOLATE_CHILD`` — never run while pytest is running, e.g. recursion / the gateway's own
suite); a hard per-run timeout (``verification.timeout_seconds``, the backend kills on expiry);
cwd = the detected PROJECT ROOT; capture-not-stream; bounded failure list + ``raw_tail``.
Graceful-degrade: no runner → ``{"ok": false, "reason": …}``; never raises, never hangs.

Config (``verification`` in DEFAULT_CONFIG, read via load_config): ``enabled``,
``runner_override``, ``timeout_seconds`` (60), ``max_failures_to_report`` (5).
The post-edit auto-trigger is a DEFERRED fast-follow (not shipped here).
"""

import json
import os
import re
import shlex
from typing import Any, Optional, Tuple

# A pytest short-summary failure line: ``FAILED path::test[param] - message`` / ``ERROR path - msg``.
_PYTEST_FAIL_RE = re.compile(r"^(?P<kind>FAILED|ERROR)\s+(?P<loc>\S+?)(?:\[.*?\])?\s+-\s+(?P<msg>.*)$")
# pytest summary counts (the AUTHORITATIVE pass/fail signal): ``2 failed, 1 passed in 0.01s``.
_PYTEST_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")

_RAW_TAIL_CHARS = 4000


def _verification_config() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("verification") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def verification_enabled(cfg: Optional[dict] = None) -> bool:
    cfg = cfg if cfg is not None else _verification_config()
    return bool(cfg.get("enabled", True))


def _timeout_seconds(cfg: dict) -> int:
    try:
        return max(1, int(cfg.get("timeout_seconds", 60)))
    except (TypeError, ValueError):
        return 60


def _max_failures(cfg: dict) -> int:
    try:
        return max(1, int(cfg.get("max_failures_to_report", 5)))
    except (TypeError, ValueError):
        return 5


# --------------------------------------------------------------------------
# runner detection (walk up for project markers)
# --------------------------------------------------------------------------

def _has_any(d: str, names) -> bool:
    return any(os.path.isfile(os.path.join(d, n)) for n in names)


def _file_contains(path: str, needle: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return needle in fh.read()
    except OSError:
        return False


def _detect_in_dir(d: str) -> Optional[str]:
    # go first (unambiguous), then JS (jest/vitest), then pytest.
    if os.path.isfile(os.path.join(d, "go.mod")):
        return "go"
    pkg = os.path.join(d, "package.json")
    if _has_any(d, ("jest.config.js", "jest.config.ts", "jest.config.cjs", "jest.config.mjs", "jest.config.json")) \
            or (os.path.isfile(pkg) and _file_contains(pkg, '"jest"')):
        return "jest"
    if _has_any(d, ("vitest.config.js", "vitest.config.ts", "vitest.config.mjs", "vitest.config.cjs", "vite.config.ts", "vite.config.js")) \
            and (os.path.isfile(pkg) and _file_contains(pkg, "vitest")):
        return "vitest"
    if os.path.isfile(pkg) and _file_contains(pkg, "vitest"):
        return "vitest"
    if _has_any(d, ("pytest.ini", "tox.ini", "conftest.py")):
        return "pytest"
    pp = os.path.join(d, "pyproject.toml")
    if os.path.isfile(pp) and _file_contains(pp, "[tool.pytest"):
        return "pytest"
    sc = os.path.join(d, "setup.cfg")
    if os.path.isfile(sc) and _file_contains(sc, "[tool:pytest]"):
        return "pytest"
    return None


def detect_runner(start_dir: str) -> Optional[Tuple[str, str]]:
    """Walk up from ``start_dir`` → ``(runner, project_root)`` or None."""
    d = os.path.abspath(start_dir) if start_dir else os.path.abspath(".")
    while True:
        runner = _detect_in_dir(d)
        if runner:
            return runner, d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


# --------------------------------------------------------------------------
# command + parse per runner
# --------------------------------------------------------------------------

def _build_command(runner: str, paths) -> str:
    targets = " ".join(shlex.quote(p) for p in paths) if paths else ""
    if runner == "pytest":
        return f"pytest -q -rA --tb=line {targets}".strip()
    if runner == "jest":
        return f"npx --no-install jest --json {targets}".strip()
    if runner == "vitest":
        return f"npx --no-install vitest run --reporter=json {targets}".strip()
    if runner == "go":
        return f"go test -json {targets or './...'}".strip()
    return ""


def _looks_timed_out(stdout: str, exit_code: int) -> bool:
    if exit_code in (124, 137, -9, -15):  # `timeout`/SIGKILL/SIGTERM conventions
        return True
    low = stdout.lower()
    return "timed out" in low or "timeout expired" in low


def _parse_pytest(stdout: str, exit_code: int, max_failures: int) -> dict:
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for n, kind in _PYTEST_COUNT_RE.findall(stdout):
        key = "error" if kind.startswith("error") else kind
        if key in counts:
            counts[key] += int(n)
    failures = []
    for line in stdout.splitlines():
        m = _PYTEST_FAIL_RE.match(line.strip())
        if not m:
            continue
        loc = m.group("loc")
        file_part, _, test = loc.partition("::")
        failures.append({"file": file_part, "test": test or None, "message": m.group("msg").strip()})
    total = counts["passed"] + counts["failed"] + counts["error"] + counts["skipped"]
    saw_summary = bool(_PYTEST_COUNT_RE.search(stdout))
    if not saw_summary and exit_code != 0:
        # no parseable summary + non-zero → tooling problem (pytest missing / collection error)
        status = "timeout" if _looks_timed_out(stdout, exit_code) else "error"
    else:
        status = "passed" if (counts["failed"] == 0 and counts["error"] == 0 and exit_code == 0) else "failed"
    return {
        "runner": "pytest", "status": status,
        "passed": counts["passed"], "failed": counts["failed"], "errors": counts["error"],
        "skipped": counts["skipped"], "total": total, "exit_code": exit_code,
        "failures": failures[:max_failures],
        "truncated_failures": len(failures) > max_failures,
        "raw_tail": stdout[-_RAW_TAIL_CHARS:],
    }


def _parse_generic_json(runner: str, stdout: str, exit_code: int, max_failures: int) -> dict:
    """Best-effort jest/vitest --json parse; vitest/jest share the {numPassedTests,...} shape."""
    data = None
    try:
        # jest/vitest may emit a leading non-JSON banner; grab the last JSON object.
        start = stdout.rfind("{")
        if start != -1:
            data = json.loads(stdout[start:]) if stdout[start:].strip().endswith("}") else json.loads(stdout)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        passed = int(data.get("numPassedTests", 0))
        failed = int(data.get("numFailedTests", 0))
        total = int(data.get("numTotalTests", passed + failed))
        status = "passed" if (failed == 0 and exit_code == 0) else "failed"
        return {
            "runner": runner, "status": status, "passed": passed, "failed": failed, "errors": 0,
            "skipped": int(data.get("numPendingTests", 0)), "total": total, "exit_code": exit_code,
            "failures": [], "raw_tail": stdout[-_RAW_TAIL_CHARS:],
        }
    status = "timeout" if _looks_timed_out(stdout, exit_code) else ("passed" if exit_code == 0 else "failed")
    return {"runner": runner, "status": status, "exit_code": exit_code, "raw_tail": stdout[-_RAW_TAIL_CHARS:]}


def _parse(runner: str, stdout: str, exit_code: int, max_failures: int) -> dict:
    if runner == "pytest":
        return _parse_pytest(stdout, exit_code, max_failures)
    if runner in ("jest", "vitest"):
        return _parse_generic_json(runner, stdout, exit_code, max_failures)
    # go test -json: count action lines (best-effort)
    passed = stdout.count('"Action":"pass"')
    failed = stdout.count('"Action":"fail"')
    status = "timeout" if _looks_timed_out(stdout, exit_code) else ("passed" if exit_code == 0 else "failed")
    return {
        "runner": runner, "status": status, "passed": passed, "failed": failed, "errors": 0,
        "total": passed + failed, "exit_code": exit_code, "failures": [], "raw_tail": stdout[-_RAW_TAIL_CHARS:],
    }


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------

def _harness_active() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("HERMES_ISOLATE_CHILD"))


def run_tests(args: dict, *, task_id: str = "default") -> str:
    """Detect + run the project's test suite, return a structured JSON result."""
    cfg = _verification_config()
    if not verification_enabled(cfg):
        return json.dumps({"ok": False, "reason": "verification disabled (verification.enabled=false)"})
    if _harness_active():
        return json.dumps({"ok": False, "reason": "refused: a test harness is active (won't run tests inside a test run)"})
    try:
        from tools.file_tools import _get_file_ops
        fops = _get_file_ops(task_id)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "reason": f"backend unavailable: {str(e)[:120]}"})

    start = getattr(getattr(fops, "env", None), "cwd", None) or getattr(fops, "cwd", None) or "."
    override = (args.get("runner") or cfg.get("runner_override") or "").strip() or None
    paths = args.get("paths") or None
    if isinstance(paths, str):
        paths = [paths]

    detected = detect_runner(start)
    if override and detected:
        runner, root = override, detected[1]
    elif override:
        runner, root = override, start
    elif detected:
        runner, root = detected
    else:
        return json.dumps({"ok": False, "reason": "no test runner detected (pytest/jest/vitest/go) in the project"})

    cmd = _build_command(runner, paths)
    if not cmd:
        return json.dumps({"ok": False, "reason": f"unsupported runner: {runner}"})
    try:
        res = fops._exec(cmd, cwd=root, timeout=_timeout_seconds(cfg))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "reason": f"run failed: {str(e)[:120]}"})
    out = getattr(res, "stdout", "") or ""
    code = getattr(res, "exit_code", 1)
    result = _parse(runner, out, code, _max_failures(cfg))
    result["ok"] = True
    result["project_root"] = root
    return json.dumps(result)


from tools.registry import registry  # noqa: E402

RUN_TESTS_SCHEMA = {
    "name": "run_tests",
    "description": (
        "Run the project's test suite (auto-detects pytest / jest / vitest / go test) and return a "
        "STRUCTURED result — {passed, failed, errors, total, status, failures:[{file, test, message}]} — "
        "so you reason about results instead of scraping stdout. Optional 'paths' narrows the run; "
        "'runner' overrides detection. Runs at the detected project root with a hard timeout; returns "
        "{ok:false, reason} when no runner is detected or a test harness is active."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional test files/dirs to run (default: the whole project).",
            },
            "runner": {
                "type": "string",
                "description": "Optional override: pytest | jest | vitest | go (default: auto-detect).",
            },
        },
        "required": [],
    },
}


def _run_tests_available() -> bool:
    """check_fn: expose run_tests only when enabled AND a runner is detectable from cwd."""
    if not verification_enabled():
        return False
    try:
        return detect_runner(os.getcwd()) is not None
    except Exception:  # noqa: BLE001
        return False


# Direct top-level registration (NOT a for-loop — a loop-nested register() is invisible to the
# AST auto-discovery in tools/registry.py::_module_registers_tools; that was the Plan-1 ship bug).
registry.register(
    name="run_tests",
    toolset="verification",
    schema=RUN_TESTS_SCHEMA,
    handler=lambda args, **kw: run_tests(args, task_id=kw.get("task_id", "default")),
    check_fn=_run_tests_available,
    is_async=False,
    emoji="🧪",
)
