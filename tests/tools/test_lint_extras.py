"""Tests for the ruff/eslint lint overlay (tools/lint_extras.py).

Two layers:
1. Mocked-``exec_fn`` tests for the logic (config gates, project opt-in, output parsing,
   autofix bounding, graceful degrade).
2. REAL-ruff integration tests (ruff is hermes-agent's own dev dep, so present in CI) that
   exercise the actual CLI contract — guards against the "mocks hide an integration bug"
   trap. SKIP if ruff is somehow absent.
"""

import shutil
import subprocess
import types

import pytest

import tools.lint_extras as le


# --------------------------------------------------------------------------
# fake _exec (mimics file_operations._exec: stdin_data piped, stderr merged into stdout)
# --------------------------------------------------------------------------

def _fake_exec(stdout="", exit_code=0, capture=None):
    def _exec(cmd, *, stdin_data=None, timeout=None):
        if capture is not None:
            capture["cmd"] = cmd
            capture["stdin"] = stdin_data
        return types.SimpleNamespace(stdout=stdout, exit_code=exit_code)
    return _exec


# --------------------------------------------------------------------------
# config gates
# --------------------------------------------------------------------------

def test_should_lint_with_ruff_gated_off_by_config():
    assert le.should_lint_with_ruff("/p/a.py", has_command=lambda c: True, cfg={"enabled": True, "ruff_enabled": False}) is False
    assert le.should_lint_with_ruff("/p/a.py", has_command=lambda c: True, cfg={"enabled": False}) is False


def test_should_lint_with_ruff_needs_binary_and_project(monkeypatch):
    monkeypatch.setattr(le, "project_uses_ruff", lambda p: True)
    assert le.should_lint_with_ruff("/p/a.py", has_command=lambda c: False, cfg={}) is False  # no binary
    monkeypatch.setattr(le, "project_uses_ruff", lambda p: False)
    assert le.should_lint_with_ruff("/p/a.py", has_command=lambda c: True, cfg={}) is False   # no project config


def test_autofix_gates():
    assert le.autofix_enabled({"enabled": True, "autofix": False}) is False
    assert le.autofix_enabled({"enabled": True, "autofix": True}) is True
    assert le.autofix_enabled({"enabled": False, "autofix": True}) is False
    assert le.autofix_max_changes({"autofix_max_changes": 7}) == 7
    assert le.autofix_max_changes({}) == 50
    assert le.autofix_max_changes({"autofix_max_changes": "nope"}) == 50


# --------------------------------------------------------------------------
# project opt-in detection (walk up)
# --------------------------------------------------------------------------

def test_project_uses_ruff_walks_up_for_markers(tmp_path):
    (tmp_path / "ruff.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg" / "deep"
    sub.mkdir(parents=True)
    assert le.project_uses_ruff(str(sub / "a.py")) is True


def test_project_uses_ruff_pyproject_tool_section(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
    assert le.project_uses_ruff(str(tmp_path / "a.py")) is True


def test_project_uses_ruff_false_without_config(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert le.project_uses_ruff(str(tmp_path / "a.py")) is False


# --------------------------------------------------------------------------
# run_ruff parsing (mocked)
# --------------------------------------------------------------------------

def test_run_ruff_findings_only_and_drops_summary():
    out = "/p/a.py:1:8: F401 `os` imported but unused\nFound 1 error.\n[*] 1 fixable with the `--fix` option."
    ok, output, skipped = le.run_ruff("/p/a.py", "import os\n", exec_fn=_fake_exec(stdout=out, exit_code=1))
    assert ok is False and skipped is False
    assert output == "/p/a.py:1:8: F401 `os` imported but unused"  # summary trailers dropped


def test_run_ruff_clean_is_ok():
    ok, output, skipped = le.run_ruff("/p/a.py", "x = 1\n", exec_fn=_fake_exec(stdout="All checks passed!", exit_code=0))
    assert ok is True and output == "" and skipped is False


def test_run_ruff_unusable_is_skipped_not_error():
    # exit non-zero but no parseable finding line → tooling problem → skipped (write not flagged)
    ok, output, skipped = le.run_ruff("/p/a.py", "x=1\n", exec_fn=_fake_exec(stdout="ruff failed: invalid config", exit_code=2))
    assert skipped is True and ok is True


def test_run_ruff_never_raises_on_exec_error():
    def _boom(cmd, *, stdin_data=None, timeout=None):
        raise RuntimeError("backend down")
    ok, output, skipped = le.run_ruff("/p/a.py", "x=1\n", exec_fn=_boom)
    assert skipped is True and ok is True


def test_run_ruff_none_content_skips():
    assert le.run_ruff("/p/a.py", None, exec_fn=_fake_exec()) == (True, "", True)


# --------------------------------------------------------------------------
# D4 normalization (used by file_operations._check_lint_delta)
# --------------------------------------------------------------------------

def test_normalize_finding_line_strips_location():
    assert le.normalize_finding_line("/p/a.py:10:8: F401 `os` imported but unused") == "F401 `os` imported but unused"
    # same finding at a shifted line → same normalized key (the leak D4 fixes)
    assert le.normalize_finding_line("/p/a.py:14:8: F401 `os` imported but unused") == le.normalize_finding_line("/p/a.py:10:8: F401 `os` imported but unused")
    # a line without a location prefix passes through (py_compile/json behavior untouched)
    assert le.normalize_finding_line("SyntaxError: invalid syntax") == "SyntaxError: invalid syntax"


# --------------------------------------------------------------------------
# autofix bounding (mocked)
# --------------------------------------------------------------------------

def test_autofix_ruff_applies_within_cap():
    fixed = "x = 1\n"
    applied, content, count, summary = le.autofix_ruff("/p/a.py", "import os\nx = 1\n", exec_fn=_fake_exec(stdout=fixed, exit_code=0), max_changes=50)
    assert applied is True and content == fixed and count >= 1 and "applied" in summary


def test_autofix_ruff_skips_over_cap():
    fixed = "".join(f"line{i}\n" for i in range(100))
    applied, content, count, summary = le.autofix_ruff("/p/a.py", "x = 1\n", exec_fn=_fake_exec(stdout=fixed, exit_code=0), max_changes=10)
    assert applied is False and content is None and count > 10 and "exceeds cap" in summary


def test_autofix_ruff_noop_when_unchanged():
    same = "x = 1\n"
    applied, content, count, summary = le.autofix_ruff("/p/a.py", same, exec_fn=_fake_exec(stdout=same, exit_code=0), max_changes=50)
    assert applied is False and count == 0


def test_autofix_ruff_never_raises():
    def _boom(cmd, *, stdin_data=None, timeout=None):
        raise RuntimeError("x")
    applied, content, count, summary = le.autofix_ruff("/p/a.py", "x=1\n", exec_fn=_boom, max_changes=50)
    assert applied is False and content is None


# --------------------------------------------------------------------------
# REAL ruff — validate the actual CLI contract (not just mocks)
# --------------------------------------------------------------------------

def _real_exec(cmd, *, stdin_data=None, timeout=None):
    """Mimics file_operations._exec: shell, stdin piped, stderr MERGED into stdout."""
    r = subprocess.run(cmd, shell=True, input=stdin_data, capture_output=True, text=True, timeout=timeout)
    return types.SimpleNamespace(exit_code=r.returncode, stdout=(r.stdout or "") + (r.stderr or ""))


def _ruff_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff.lint]\nselect = ['F']\n", encoding="utf-8")
    return tmp_path


ruff_real = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")


@ruff_real
def test_real_ruff_reports_unused_import(tmp_path):
    proj = _ruff_project(tmp_path)
    ok, output, skipped = le.run_ruff(str(proj / "a.py"), "import os\nx = 1\n", exec_fn=_real_exec)
    assert skipped is False and ok is False
    assert "F401" in output and ":" in output  # a real path:line:col: F401 line
    assert "Found" not in output  # summary trailer filtered out


@ruff_real
def test_real_ruff_clean_file_ok(tmp_path):
    proj = _ruff_project(tmp_path)
    ok, output, skipped = le.run_ruff(str(proj / "a.py"), "x = 1\n", exec_fn=_real_exec)
    assert ok is True and output == "" and skipped is False


@ruff_real
def test_real_ruff_autofix_removes_unused_import(tmp_path):
    proj = _ruff_project(tmp_path)
    applied, fixed, count, summary = le.autofix_ruff(str(proj / "a.py"), "import os\nx = 1\n", exec_fn=_real_exec, max_changes=50)
    assert applied is True
    assert fixed is not None and "import os" not in fixed and "x = 1" in fixed
    assert "Found" not in fixed and "fixed" not in fixed  # clean fixed code, no summary pollution
    assert count >= 1


@ruff_real
def test_real_ruff_autofix_respects_cap(tmp_path):
    proj = _ruff_project(tmp_path)
    applied, fixed, count, summary = le.autofix_ruff(str(proj / "a.py"), "import os\nx = 1\n", exec_fn=_real_exec, max_changes=0)
    assert applied is False and fixed is None and "exceeds cap" in summary
