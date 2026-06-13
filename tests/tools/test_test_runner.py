"""Tests for the run_tests tool (tools/test_runner.py).

Mocked unit tests (detection, parse, safety, graceful degrade) + a REAL-pytest integration
test (the Plan-1/2 real-binary lesson) that runs a tiny fixture suite end-to-end through the
backend exec. The harness-active guard is tested for REAL (PYTEST_CURRENT_TEST is set while
these run).
"""

import json
import os
import shutil
import subprocess
import types

import pytest

import tools.test_runner as tr


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_detect_pytest_via_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    assert tr.detect_runner(str(sub)) == ("pytest", str(tmp_path))


def test_detect_go_and_jest(tmp_path):
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    assert tr.detect_runner(str(tmp_path))[0] == "go"
    g = tmp_path / "js"
    g.mkdir()
    (g / "package.json").write_text('{"jest": {}}', encoding="utf-8")
    assert tr.detect_runner(str(g))[0] == "jest"


def test_detect_none(tmp_path):
    assert tr.detect_runner(str(tmp_path)) is None


# --------------------------------------------------------------------------
# pytest parse (the validated D1 format)
# --------------------------------------------------------------------------

_PYTEST_OUT = (
    "short test summary info\n"
    "FAILED test_sample.py::test_fail - AssertionError: math is broken\n"
    "FAILED test_sample.py::test_error - RuntimeError: boom\n"
    "2 failed, 1 passed in 0.01s\n"
)


def test_parse_pytest_counts_and_failures():
    r = tr._parse_pytest(_PYTEST_OUT, exit_code=1, max_failures=5)
    assert r["status"] == "failed"
    assert r["passed"] == 1 and r["failed"] == 2 and r["total"] == 3
    assert {f["test"] for f in r["failures"]} == {"test_fail", "test_error"}
    assert r["failures"][0]["file"] == "test_sample.py"
    assert "math is broken" in r["failures"][0]["message"]


def test_parse_pytest_clean_passes():
    r = tr._parse_pytest("3 passed in 0.02s\n", exit_code=0, max_failures=5)
    assert r["status"] == "passed" and r["failed"] == 0 and r["passed"] == 3


def test_parse_pytest_caps_failures():
    out = "".join(f"FAILED t.py::test_{i} - boom\n" for i in range(20)) + "20 failed in 1s\n"
    r = tr._parse_pytest(out, exit_code=1, max_failures=5)
    assert len(r["failures"]) == 5 and r["truncated_failures"] is True


def test_parse_pytest_tooling_error_is_status_error():
    # non-zero exit, no summary line (e.g. pytest missing / collection crash)
    r = tr._parse_pytest("ModuleNotFoundError: No module named pytest\n", exit_code=2, max_failures=5)
    assert r["status"] == "error"


# --------------------------------------------------------------------------
# run_tests: safety + graceful degrade (mock the backend)
# --------------------------------------------------------------------------

def _fake_fops(stdout, exit_code, cwd="/proj"):
    fops = types.SimpleNamespace(cwd=cwd, env=types.SimpleNamespace(cwd=cwd))
    fops._exec = lambda cmd, cwd=None, timeout=None: types.SimpleNamespace(stdout=stdout, exit_code=exit_code)
    return fops


def test_run_tests_refuses_when_harness_active(monkeypatch):
    # PYTEST_CURRENT_TEST is genuinely set right now → no patch → must refuse
    out = json.loads(tr.run_tests({}))
    assert out["ok"] is False and "harness is active" in out["reason"]


def test_run_tests_no_runner_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "_harness_active", lambda: False)
    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": _fake_fops("", 0, cwd=str(tmp_path)))
    out = json.loads(tr.run_tests({}))
    assert out["ok"] is False and "no test runner" in out["reason"]


def test_run_tests_structured_result(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    monkeypatch.setattr(tr, "_harness_active", lambda: False)
    monkeypatch.setattr("tools.file_tools._get_file_ops",
                        lambda task_id="default": _fake_fops(_PYTEST_OUT, 1, cwd=str(tmp_path)))
    out = json.loads(tr.run_tests({}))
    assert out["ok"] is True and out["runner"] == "pytest" and out["status"] == "failed"
    assert out["failed"] == 2 and out["passed"] == 1
    assert out["project_root"] == str(tmp_path)


def test_run_tests_disabled(monkeypatch):
    monkeypatch.setattr(tr, "_verification_config", lambda: {"enabled": False})
    out = json.loads(tr.run_tests({}))
    assert out["ok"] is False and "disabled" in out["reason"]


# --------------------------------------------------------------------------
# REAL pytest end-to-end (skip if absent)
# --------------------------------------------------------------------------

class _RealEnv:
    def __init__(self, cwd):
        self.cwd = cwd

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        # scrub the outer pytest's markers so the NESTED pytest runs cleanly
        env = dict(os.environ)
        for k in ("PYTEST_CURRENT_TEST", "PYTEST_PLUGINS", "HERMES_ISOLATE_CHILD"):
            env.pop(k, None)
        r = subprocess.run(command, shell=True, cwd=cwd or self.cwd, input=stdin_data,
                           capture_output=True, text=True, timeout=timeout, env=env)
        return {"output": (r.stdout or "") + (r.stderr or ""), "returncode": r.returncode}


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_real_pytest_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / "test_x.py").write_text(
        "def test_pass():\n    assert 1 == 1\n\ndef test_fail():\n    assert 1 == 2, 'nope'\n",
        encoding="utf-8",
    )
    from tools.file_operations import ShellFileOperations
    fops = ShellFileOperations(_RealEnv(str(tmp_path)))
    monkeypatch.setattr(tr, "_harness_active", lambda: False)
    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id="default": fops)

    out = json.loads(tr.run_tests({}))
    assert out["ok"] is True and out["runner"] == "pytest"
    assert out["status"] == "failed"
    assert out["passed"] == 1 and out["failed"] == 1
    assert any(f["test"] == "test_fail" for f in out["failures"])


# --------------------------------------------------------------------------
# discovery contract (the Plan-1 ship-bug guard) + cli reachability
# --------------------------------------------------------------------------

def test_run_tests_discovered_and_reaches_cli():
    from pathlib import Path
    from tools.registry import discover_builtin_tools, _module_registers_tools

    assert _module_registers_tools(Path("tools/test_runner.py")) is True
    assert "tools.test_runner" in discover_builtin_tools()

    import model_tools  # noqa: F401 — triggers real startup discovery
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset

    assert "verification" in _get_platform_tools({}, "cli")
    assert "run_tests" in resolve_toolset("verification")
