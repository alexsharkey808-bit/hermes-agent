#!/usr/bin/env python3
"""Fork-owned overlay: edit-time ruff/eslint lint + bounded, safe auto-fix.

Extends the existing lint-on-write path in ``tools/file_operations.py`` WITHOUT bloating
that hot, upstream-churned file: it calls the small, pure functions here. This module must
NOT import ``file_operations`` (cycle) — it takes an ``exec_fn`` callable (the handler's
backend-respecting ``_exec(cmd, stdin_data=...)``) and an optional ``has_command`` callable.

Activation (D1): ruff/eslint run only when the binary is on PATH AND the project opts in
(a ruff/eslint config is present up-tree) AND the master ``lint.ruff_enabled`` /
``lint.eslint_enabled`` gate is on. Otherwise the caller keeps today's behavior. Everything
degrades gracefully — absent binary, no project config, unusable linter, or any exception →
a ``skipped`` outcome, never a raise, never a blocked write.

Auto-fix (D2): ``lint.autofix`` default OFF. SAFE fixes only (``ruff check --fix`` /
``eslint --fix`` — NEVER ``--unsafe-fixes``). One pass; bounded by ``lint.autofix_max_changes``;
the caller re-reads + reports. Config is read via ``hermes_cli.config.load_config`` (mtime-cached).

ruff CLI contract (validated against ruff 0.15.10):
  - ``ruff check --stdin-filename <path> --output-format concise --no-fix -`` (content on stdin)
    → findings ``path:line:col: CODE msg`` on STDOUT, exit 1 on violations / 0 clean.
  - ``ruff check --stdin-filename <path> --fix - 2>/dev/null`` → the FIXED source on STDOUT.
    The trailing ``2>/dev/null`` is REQUIRED because the handler's ``_exec`` merges stderr into
    stdout (``stderr=subprocess.STDOUT``); ruff writes its summary to stderr, which would
    otherwise corrupt the fixed content.
"""

import os
import re
import shlex
from typing import Any, Callable, Optional, Tuple

# A finding line: ``<path>:<line>:<col>: <rest>`` (ruff concise, eslint unix). Used both to
# keep only real findings (dropping summary trailers) and as the D4 normalization anchor.
_FINDING_RE = re.compile(r"^(?P<loc>.+?:\d+:\d+:)\s*(?P<rest>.*)$")

_RUFF_CONFIG_FILES = ("ruff.toml", ".ruff.toml")
_ESLINT_CONFIG_FILES = (
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
    ".eslintrc.yaml", ".eslintrc.yml", "eslint.config.js", "eslint.config.mjs",
    "eslint.config.cjs", "eslint.config.ts",
)

ExecFn = Callable[..., Any]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def lint_config() -> dict:
    """Read the ``lint`` config section (mtime-cached via load_config). Never raises."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("lint") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _flag(cfg: dict, key: str, default: bool) -> bool:
    return bool(cfg.get("enabled", True)) and bool(cfg.get(key, default))


def autofix_enabled(cfg: Optional[dict] = None) -> bool:
    cfg = cfg if cfg is not None else lint_config()
    return _flag(cfg, "autofix", False)


def autofix_max_changes(cfg: Optional[dict] = None) -> int:
    cfg = cfg if cfg is not None else lint_config()
    try:
        return max(0, int(cfg.get("autofix_max_changes", 50)))
    except (TypeError, ValueError):
        return 50


# --------------------------------------------------------------------------
# project opt-in detection (walk up from the file's directory)
# --------------------------------------------------------------------------

def _walk_up(start_dir: str, predicate: Callable[[str], bool]) -> bool:
    d = os.path.abspath(start_dir) if start_dir else os.path.abspath(".")
    while True:
        try:
            if predicate(d):
                return True
        except OSError:
            pass
        parent = os.path.dirname(d)
        if parent == d:
            return False
        d = parent


def _pyproject_has_ruff(d: str) -> bool:
    pp = os.path.join(d, "pyproject.toml")
    if not os.path.isfile(pp):
        return False
    try:
        with open(pp, encoding="utf-8") as fh:
            return "[tool.ruff" in fh.read()
    except OSError:
        return False


def _package_json_has_eslint(d: str) -> bool:
    pkg = os.path.join(d, "package.json")
    if not os.path.isfile(pkg):
        return False
    try:
        with open(pkg, encoding="utf-8") as fh:
            return "eslintConfig" in fh.read()
    except OSError:
        return False


def project_uses_ruff(path: str) -> bool:
    start = os.path.dirname(os.path.abspath(path))
    return _walk_up(start, lambda d: any(os.path.isfile(os.path.join(d, m)) for m in _RUFF_CONFIG_FILES) or _pyproject_has_ruff(d))


def project_uses_eslint(path: str) -> bool:
    start = os.path.dirname(os.path.abspath(path))
    return _walk_up(start, lambda d: any(os.path.isfile(os.path.join(d, m)) for m in _ESLINT_CONFIG_FILES) or _package_json_has_eslint(d))


def _available(cmd: str, has_command: Optional[Callable[[str], bool]]) -> bool:
    if has_command is not None:
        try:
            return bool(has_command(cmd))
        except Exception:  # noqa: BLE001
            return False
    import shutil
    return shutil.which(cmd) is not None


def should_lint_with_ruff(path: str, *, has_command: Optional[Callable[[str], bool]] = None, cfg: Optional[dict] = None) -> bool:
    cfg = cfg if cfg is not None else lint_config()
    if not _flag(cfg, "ruff_enabled", True):
        return False
    return _available("ruff", has_command) and project_uses_ruff(path)


def should_lint_with_eslint(path: str, *, has_command: Optional[Callable[[str], bool]] = None, cfg: Optional[dict] = None) -> bool:
    cfg = cfg if cfg is not None else lint_config()
    if not _flag(cfg, "eslint_enabled", True):
        return False
    return _available("eslint", has_command) and project_uses_eslint(path)


# --------------------------------------------------------------------------
# linting (content via stdin → only-findings output)
# --------------------------------------------------------------------------

def _findings_only(output: str) -> str:
    """Keep only ``path:line:col: …`` finding lines, dropping summary trailers
    (``Found N errors``, ``[*] N fixable``, ``All checks passed!``)."""
    return "\n".join(ln for ln in output.splitlines() if _FINDING_RE.match(ln.strip()))


def normalize_finding_line(line: str) -> str:
    """Strip a leading ``<path>:<line>:<col>:`` so a finding compares equal across an edit
    that only shifted its line number (D4). Lines without that prefix pass through unchanged
    (so py_compile/json/yaml behavior is untouched)."""
    m = _FINDING_RE.match(line.strip())
    return m.group("rest").strip() if m else line.strip()


def _run(exec_fn: ExecFn, cmd: str, content: str) -> Tuple[int, str]:
    res = exec_fn(cmd, stdin_data=content, timeout=30)
    return getattr(res, "exit_code", 1), (getattr(res, "stdout", "") or "")


def run_ruff(path: str, content: Optional[str], *, exec_fn: ExecFn) -> Tuple[bool, str, bool]:
    """Lint ``content`` with ruff (stdin). Returns ``(ok, output, skipped)``; never raises.

    ``output`` is the finding lines only (``path:line:col: CODE msg``). A linter that can't
    run (bad config etc.) → ``skipped`` so the write isn't flagged.
    """
    if content is None:
        return (True, "", True)
    cmd = f"ruff check --stdin-filename {shlex.quote(path)} --output-format concise --no-fix -"
    try:
        code, out = _run(exec_fn, cmd, content)
    except Exception:  # noqa: BLE001
        return (True, "", True)
    findings = _findings_only(out)
    if code == 0:
        return (True, "", False)
    if not findings:
        # exit non-zero but no parseable findings → tooling/config problem, not a lint failure
        return (True, out.strip(), True)
    return (False, findings, False)


def run_eslint(path: str, content: Optional[str], *, exec_fn: ExecFn) -> Tuple[bool, str, bool]:
    """Lint ``content`` with eslint (stdin, unix format). Same contract as ``run_ruff``.

    NOTE: validated against ruff but NOT against a real eslint binary in this build — the
    graceful-degrade keeps it safe; the live-smoke confirms it where eslint is installed.
    """
    if content is None:
        return (True, "", True)
    cmd = f"eslint --stdin --stdin-filename {shlex.quote(path)} --format unix"
    try:
        code, out = _run(exec_fn, cmd, content)
    except Exception:  # noqa: BLE001
        return (True, "", True)
    findings = _findings_only(out)
    if code == 0:
        return (True, "", False)
    if not findings:
        return (True, out.strip(), True)
    return (False, findings, False)


# --------------------------------------------------------------------------
# bounded, safe auto-fix (content via stdin → fixed content)
# --------------------------------------------------------------------------

def _changed_line_count(before: str, after: str) -> int:
    import difflib
    b, a = before.splitlines(), after.splitlines()
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def _autofix(linter: str, fix_cmd: str, content: Optional[str], *, exec_fn: ExecFn, max_changes: int) -> Tuple[bool, Optional[str], int, str]:
    if content is None:
        return (False, None, 0, "")
    try:
        # ``2>/dev/null``: _exec merges stderr→stdout; the linter writes its summary to
        # stderr, which would corrupt the fixed source on stdout. SAFE fixes only.
        res = exec_fn(f"{fix_cmd} 2>/dev/null", stdin_data=content, timeout=30)
    except Exception:  # noqa: BLE001
        return (False, None, 0, f"autofix skipped ({linter} error)")
    fixed = getattr(res, "stdout", None)
    if fixed is None or fixed == content:
        return (False, None, 0, "no fixable issues")
    changed = _changed_line_count(content, fixed)
    if changed == 0:
        return (False, None, 0, "no fixable issues")
    if changed > max_changes:
        return (False, None, changed, f"autofix skipped: {changed} changed lines exceeds cap ({max_changes})")
    return (True, fixed, changed, f"{linter} --fix applied ({changed} line(s) changed)")


def autofix_ruff(path: str, content: Optional[str], *, exec_fn: ExecFn, max_changes: int) -> Tuple[bool, Optional[str], int, str]:
    """SAFE ruff auto-fix of ``content`` (stdin). Returns ``(applied, fixed_content,
    fixed_count, summary)``. Bounded (skips if > ``max_changes`` lines); never raises;
    never ``--unsafe-fixes``."""
    cmd = f"ruff check --stdin-filename {shlex.quote(path)} --fix -"
    return _autofix("ruff", cmd, content, exec_fn=exec_fn, max_changes=max_changes)


def autofix_eslint(path: str, content: Optional[str], *, exec_fn: ExecFn, max_changes: int) -> Tuple[bool, Optional[str], int, str]:
    cmd = f"eslint --stdin --stdin-filename {shlex.quote(path)} --fix-dry-run --format json"
    # eslint --fix-dry-run writes a JSON report (not the fixed source) — unvalidated here;
    # keep it conservative: only the ruff path is exercised in this build. Degrade to no-op
    # if the output isn't usable fixed source.
    if content is None:
        return (False, None, 0, "")
    try:
        res = exec_fn(f"{cmd} 2>/dev/null", stdin_data=content, timeout=30)
    except Exception:  # noqa: BLE001
        return (False, None, 0, "autofix skipped (eslint error)")
    import json
    try:
        report = json.loads(getattr(res, "stdout", "") or "[]")
        fixed = report[0].get("output") if report and isinstance(report[0], dict) else None
    except (ValueError, IndexError, AttributeError, TypeError):
        fixed = None
    if not fixed or fixed == content:
        return (False, None, 0, "no fixable issues")
    changed = _changed_line_count(content, fixed)
    if changed == 0 or changed > max_changes:
        reason = "no fixable issues" if changed == 0 else f"autofix skipped: {changed} changed lines exceeds cap ({max_changes})"
        return (False, None, changed, reason)
    return (True, fixed, changed, f"eslint --fix applied ({changed} line(s) changed)")
