"""End-to-end tests for the ruff lint + autofix hook in the write path.

These drive ``ShellFileOperations.write_file`` through a REAL local backend (a small
subprocess shim mirroring the production backend: stdin piped, stderr merged into output)
against REAL ruff — so the hook, the D4 delta normalization, and the autofix re-write are
exercised through the actual code path, not mocks. SKIP if ruff is absent.
"""

import subprocess
import shutil

import pytest

import tools.lint_extras as le
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")


class _RealEnv:
    """Local terminal backend: runs the command for real (stdin piped, stderr→output)."""

    def __init__(self, cwd):
        self.cwd = cwd

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        r = subprocess.run(
            command, shell=True, cwd=cwd or self.cwd,
            input=stdin_data, capture_output=True, text=True, timeout=timeout,
        )
        return {"output": (r.stdout or "") + (r.stderr or ""), "returncode": r.returncode}


def _ruff_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff.lint]\nselect = ['F']\n", encoding="utf-8")
    return tmp_path


def _ops(proj):
    return ShellFileOperations(_RealEnv(str(proj)))


def test_write_surfaces_ruff_finding(tmp_path):
    proj = _ruff_project(tmp_path)
    res = _ops(proj).write_file(str(proj / "a.py"), "import os\nx = 1\n")  # F401: os unused
    assert res.error is None
    assert res.lint and res.lint["status"] == "error"
    assert "F401" in res.lint["output"]


def test_clean_write_no_lint_error(tmp_path):
    proj = _ruff_project(tmp_path)
    res = _ops(proj).write_file(str(proj / "a.py"), "x = 1\n")
    # clean ruff → lint status ok/skipped, no F-codes surfaced
    assert res.error is None
    assert not (res.lint and res.lint.get("status") == "error" and "F" in res.lint.get("output", ""))


def test_delta_suppresses_preexisting_after_line_shift(tmp_path):
    proj = _ruff_project(tmp_path)
    ops = _ops(proj)
    path = str(proj / "a.py")
    ops.write_file(path, "import os\nx = 1\n")                       # pre-existing F401: os
    # add a NEW unused import AND shift os's line down (a blank line first)
    res = ops.write_file(path, "\nimport os\nimport sys\nx = 1\n")  # F401 os (shifted) + F401 sys (new)
    out = (res.lint or {}).get("output", "")
    assert "sys" in out                  # the new finding surfaces
    assert "import os" not in out or out.count("F401") == 1  # pre-existing os suppressed despite line shift


def test_autofix_off_is_byte_identical(tmp_path):
    proj = _ruff_project(tmp_path)
    path = str(proj / "a.py")
    _ops(proj).write_file(path, "import os\nx = 1\n")
    assert (proj / "a.py").read_text(encoding="utf-8") == "import os\nx = 1\n"  # unchanged on disk
    # and no autofix field reported (default off)
    res2 = _ops(proj).write_file(str(proj / "b.py"), "import os\nx = 1\n")
    assert res2.autofix is None


def test_autofix_on_fixes_on_disk_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(le, "lint_config", lambda: {
        "enabled": True, "ruff_enabled": True, "autofix": True, "autofix_max_changes": 50,
    })
    proj = _ruff_project(tmp_path)
    path = str(proj / "a.py")
    res = _ops(proj).write_file(path, "import os\nx = 1\n")
    on_disk = (proj / "a.py").read_text(encoding="utf-8")
    assert "import os" not in on_disk and "x = 1" in on_disk      # ruff --fix removed the import
    assert res.autofix and res.autofix["fixed_count"] >= 1
    assert "applied" in res.autofix["summary"]


def test_autofix_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(le, "lint_config", lambda: {
        "enabled": True, "ruff_enabled": True, "autofix": True, "autofix_max_changes": 0,
    })
    proj = _ruff_project(tmp_path)
    path = str(proj / "a.py")
    res = _ops(proj).write_file(path, "import os\nx = 1\n")
    # cap 0 → fix NOT applied; file unchanged; the skip is reported
    assert (proj / "a.py").read_text(encoding="utf-8") == "import os\nx = 1\n"
    assert res.autofix and res.autofix.get("applied") is False and "exceeds cap" in res.autofix["summary"]


def test_ruff_absent_falls_back_to_py_compile(tmp_path, monkeypatch):
    # force the ruff route off → today's in-process py_compile path, no crash
    monkeypatch.setattr(le, "should_lint_with_ruff", lambda *a, **k: False)
    proj = _ruff_project(tmp_path)
    res = _ops(proj).write_file(str(proj / "a.py"), "import os\nx = 1\n")
    assert res.error is None  # py_compile: syntactically valid → no error, no crash
