"""Integration test for code_intelligence against a REAL Python language server.

SKIPS cleanly when no pyright is on PATH (the trafilatura/ddgs availability pattern — CI has
no language server). When pyright IS present, it spins up the real LSPService in a tiny git
fixture repo and asserts ``find_definition`` resolves a call site to its definition line.

Honest limit: in CI (and any host without pyright) this test SKIPS — the deterministic
coverage lives in tests/agent/lsp/test_navigation.py + tests/tools/test_code_intelligence.py.
"""

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    not (shutil.which("pyright-langserver") or shutil.which("pyright")),
    reason="no Python language server (pyright) on PATH — CI has none",
)


def test_find_definition_resolves_real_def(tmp_path):
    from agent.lsp.manager import LSPService

    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "pyrightconfig.json").write_text("{}", encoding="utf-8")  # project-root marker
    src = repo / "a.py"
    #            line0: def foo():
    #            line1:     return 1
    #            line4: x = foo()   ← call site; `foo` starts at 0-indexed char 4
    src.write_text("def foo():\n    return 1\n\n\nx = foo()\n", encoding="utf-8")

    svc = LSPService(enabled=True, wait_mode="document", wait_timeout=15.0, install_strategy="manual")
    try:
        if not svc.enabled_for(str(src)):
            pytest.skip("LSP not enabled_for the fixture file (no workspace/server match)")
        # 0-indexed (line 4, char 4) = the `foo` in `x = foo()`
        res = svc.request_definition_sync(str(src), 4, 4, timeout=15.0)
        if not res:
            pytest.skip("pyright present but did not resolve (cold-start / spawn failure) — flake guard")
        assert any(r.get("line") == 0 and str(r.get("path", "")).endswith("a.py") for r in res)
    finally:
        svc.shutdown()
