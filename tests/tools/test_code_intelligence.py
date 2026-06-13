"""Tool-level tests for the code_intelligence tools.

The service is mocked at the seam where the tools LOOK IT UP
(``tools.code_intelligence.get_service``), NOT ``agent.lsp.get_service``. Covers: JSON-string
return, success/​failure shapes, graceful degrade (no service → {"ok": false}), the agent-facing
1-indexed ↔ LSP 0-indexed conversion, references forwarding ``include_declaration``, and a
``request_*_sync`` raising being caught.
"""

import json
from unittest.mock import MagicMock

import tools.code_intelligence as ci


def _fake_svc():
    svc = MagicMock()
    svc.request_definition_sync = MagicMock(return_value=[{"path": "/ws/a.py", "line": 9, "character": 4}])
    svc.request_references_sync = MagicMock(return_value=[{"path": "/ws/a.py", "line": 1, "character": 0}])
    svc.document_symbols_sync = MagicMock(return_value=[{"name": "C", "kind": 5, "line": 0, "character": 6}])
    svc.workspace_symbols_sync = MagicMock(return_value=[
        {"name": "foo", "kind": 12, "line": 2, "character": 0, "path": "/ws/a.py"},
    ])
    return svc


def test_find_definition_success_shape_and_1indexed_output(monkeypatch):
    monkeypatch.setattr(ci, "get_service", lambda: _fake_svc())
    out = ci.find_definition({"path": "/ws/b.py", "line": 3, "character": 5})
    assert isinstance(out, str)                       # JSON STRING
    payload = json.loads(out)
    assert payload["ok"] is True
    # service is 0-indexed; tool presents 1-indexed (9→10, 4→5)
    assert payload["results"] == [{"path": "/ws/a.py", "line": 10, "character": 5}]


def test_input_index_converted_to_zero_indexed(monkeypatch):
    svc = _fake_svc()
    monkeypatch.setattr(ci, "get_service", lambda: svc)
    ci.find_definition({"path": "/ws/b.py", "line": 3, "character": 5})
    # 1-indexed (3,5) → 0-indexed (2,4) on the way into the service
    assert svc.request_definition_sync.call_args.args == ("/ws/b.py", 2, 4)


def test_index_conversion_clamps_at_zero(monkeypatch):
    svc = _fake_svc()
    monkeypatch.setattr(ci, "get_service", lambda: svc)
    ci.find_definition({"path": "/ws/b.py", "line": 1, "character": 1})
    assert svc.request_definition_sync.call_args.args == ("/ws/b.py", 0, 0)  # clamped, not -1


def test_graceful_degrade_when_no_service(monkeypatch):
    monkeypatch.setattr(ci, "get_service", lambda: None)
    for fn, args in (
        (ci.find_definition, {"path": "/x", "line": 1, "character": 1}),
        (ci.find_references, {"path": "/x", "line": 1, "character": 1}),
        (ci.document_symbols, {"path": "/x"}),
        (ci.workspace_symbols, {"query": "q"}),
    ):
        payload = json.loads(fn(args))
        assert payload["ok"] is False and "reason" in payload  # no raise, clear reason


def test_references_forwards_include_declaration(monkeypatch):
    svc = _fake_svc()
    monkeypatch.setattr(ci, "get_service", lambda: svc)
    ci.find_references({"path": "/ws/b.py", "line": 2, "character": 2, "include_declaration": False})
    assert svc.request_references_sync.call_args.kwargs["include_declaration"] is False


def test_request_raising_is_caught(monkeypatch):
    svc = _fake_svc()
    svc.request_definition_sync = MagicMock(side_effect=RuntimeError("server crashed"))
    monkeypatch.setattr(ci, "get_service", lambda: svc)
    payload = json.loads(ci.find_definition({"path": "/x", "line": 1, "character": 1}))
    assert payload["ok"] is False and "reason" in payload  # caught, no stack trace leak


def test_document_symbols_maps_kind_name_and_1indexes(monkeypatch):
    monkeypatch.setattr(ci, "get_service", lambda: _fake_svc())
    payload = json.loads(ci.document_symbols({"path": "/ws/a.py"}))
    assert payload["ok"] is True
    assert payload["results"] == [{"name": "C", "kind": "Class", "line": 1, "character": 7}]  # kind 5→Class, 0→1


def test_workspace_symbols_shape(monkeypatch):
    monkeypatch.setattr(ci, "get_service", lambda: _fake_svc())
    payload = json.loads(ci.workspace_symbols({"query": "foo"}))
    assert payload["ok"] is True
    assert payload["results"] == [
        {"name": "foo", "kind": "Function", "line": 3, "character": 1, "path": "/ws/a.py"},
    ]


def test_results_capped_with_truncated_flag(monkeypatch):
    svc = MagicMock()
    svc.request_references_sync = MagicMock(
        return_value=[{"path": f"/ws/{i}.py", "line": i, "character": 0} for i in range(ci._MAX_RESULTS + 50)]
    )
    monkeypatch.setattr(ci, "get_service", lambda: svc)
    payload = json.loads(ci.find_references({"path": "/x", "line": 1, "character": 1}))
    assert len(payload["results"]) == ci._MAX_RESULTS
    assert payload["truncated"] is True
    assert payload["total"] == ci._MAX_RESULTS + 50


# --------------------------------------------------------------------------
# E2E discovery contract — the for-loop-registration bug these tests now guard
# (direct-import tests passed but the module was never auto-discovered at startup,
# so the tools were invisible in production). Exercise the REAL discovery path.
# --------------------------------------------------------------------------

def test_module_found_by_auto_discovery():
    from pathlib import Path
    from tools.registry import discover_builtin_tools, _module_registers_tools

    # AST scan must see a TOP-LEVEL registry.register(...) — a loop-nested call returns False
    assert _module_registers_tools(Path("tools/code_intelligence.py")) is True
    assert "tools.code_intelligence" in discover_builtin_tools()


def test_tools_register_and_reach_cli_via_real_discovery():
    import model_tools  # noqa: F401 — import triggers discover_builtin_tools() at startup
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset
    from tools.registry import registry

    nav = {"find_definition", "find_references", "document_symbols", "workspace_symbols"}
    # the 4 tools actually register (check_fn default ON in the isolated test config)
    registered = {d["function"]["name"] for d in registry.get_definitions(nav)}
    assert registered == nav
    # NB: _get_platform_tools returns TOOLSET names (not tool names) — reachability is the
    # toolset being enabled for cli AND resolving to the 4 nav tools.
    assert "code_intelligence" in _get_platform_tools({}, "cli")
    assert nav <= set(resolve_toolset("code_intelligence"))
