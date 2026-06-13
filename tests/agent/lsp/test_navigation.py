"""Unit tests for the additive LSP code-navigation methods on LSPService.

No language server: a fake LSPClient is injected via ``_get_or_spawn`` and its
``_send_request_with_retry`` returns canned LSP payloads. The real sync-over-async
dispatch (the background loop) runs — only the client I/O is faked. Covers the LSP
result-shape POLYMORPHISM (the key correctness risk): definition returns
Location | Location[] | LocationLink[]; documentSymbol returns hierarchical
DocumentSymbol[] OR flat SymbolInformation[]; workspace/symbol returns
SymbolInformation[] | WorkspaceSymbol[]; null → [].
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.lsp.manager import LSPService


@pytest.fixture
def svc():
    s = LSPService(enabled=True, wait_mode="document", wait_timeout=2.0, install_strategy="auto")
    try:
        yield s
    finally:
        s.shutdown()


def _fake_client(*, send_return=None, send_raises=None):
    c = MagicMock()
    c.server_id = "pyright"
    c.workspace_root = "/ws"
    c.is_running = True
    c.open_file = AsyncMock(return_value=1)
    c.shutdown = AsyncMock()  # awaited by _shutdown_async in fixture teardown
    if send_raises is not None:
        c._send_request_with_retry = AsyncMock(side_effect=send_raises)
    else:
        c._send_request_with_retry = AsyncMock(return_value=send_return)
    return c


def _inject(svc, monkeypatch, client):
    monkeypatch.setattr(svc, "enabled_for", lambda fp: True)
    monkeypatch.setattr(svc, "_get_or_spawn", AsyncMock(return_value=client))


# --------------------------------------------------------------------------
# definition / references — request shape + Location vs LocationLink + lists
# --------------------------------------------------------------------------

def test_definition_sends_request_and_normalizes_single_location(svc, monkeypatch):
    client = _fake_client(send_return={
        "uri": "file:///ws/a.py",
        "range": {"start": {"line": 9, "character": 4}, "end": {"line": 9, "character": 7}},
    })
    _inject(svc, monkeypatch, client)

    out = svc.request_definition_sync("/ws/b.py", 2, 3)

    args = client._send_request_with_retry.call_args.args
    assert args[0] == "textDocument/definition"
    assert args[1]["position"] == {"line": 2, "character": 3}  # 0-indexed pass-through
    assert args[1]["textDocument"]["uri"].startswith("file://")
    assert out == [{"path": "/ws/a.py", "line": 9, "character": 4}]  # single → list


def test_references_sets_include_declaration_and_handles_locationlink_list(svc, monkeypatch):
    client = _fake_client(send_return=[
        {"uri": "file:///ws/a.py", "range": {"start": {"line": 1, "character": 0}}},
        # LocationLink shape — targetUri + targetSelectionRange (NOT uri/range)
        {"targetUri": "file:///ws/c.py", "targetSelectionRange": {"start": {"line": 5, "character": 8}}},
    ])
    _inject(svc, monkeypatch, client)

    out = svc.request_references_sync("/ws/b.py", 0, 0, include_declaration=True)

    args = client._send_request_with_retry.call_args.args
    assert args[0] == "textDocument/references"
    assert args[1]["context"]["includeDeclaration"] is True
    assert out == [
        {"path": "/ws/a.py", "line": 1, "character": 0},
        {"path": "/ws/c.py", "line": 5, "character": 8},  # LocationLink normalized
    ]


# --------------------------------------------------------------------------
# documentSymbol — hierarchical DocumentSymbol vs flat SymbolInformation
# --------------------------------------------------------------------------

def test_document_symbols_hierarchical_flattens_children(svc, monkeypatch):
    client = _fake_client(send_return=[{
        "name": "C", "kind": 5,
        "range": {"start": {"line": 0, "character": 0}},
        "selectionRange": {"start": {"line": 0, "character": 6}},
        "children": [
            {"name": "m", "kind": 6, "selectionRange": {"start": {"line": 1, "character": 4}}},
        ],
    }])
    _inject(svc, monkeypatch, client)

    out = svc.document_symbols_sync("/ws/a.py")

    by_name = {s["name"]: s for s in out}
    assert by_name["C"]["line"] == 0 and "container" not in by_name["C"]
    assert by_name["m"]["line"] == 1 and by_name["m"]["container"] == "C"  # child flattened


def test_document_symbols_flat_symbolinformation(svc, monkeypatch):
    client = _fake_client(send_return=[{
        "name": "foo", "kind": 12,
        "location": {"uri": "file:///ws/a.py", "range": {"start": {"line": 3, "character": 0}}},
    }])
    _inject(svc, monkeypatch, client)

    out = svc.document_symbols_sync("/ws/a.py")

    assert out == [{"name": "foo", "kind": 12, "line": 3, "character": 0, "path": "/ws/a.py"}]


# --------------------------------------------------------------------------
# workspace/symbol — merge across running clients + WorkspaceSymbol + dedup
# --------------------------------------------------------------------------

def test_workspace_symbols_merges_running_clients(svc):
    c1 = _fake_client(send_return=[
        {"name": "foo", "kind": 12, "location": {"uri": "file:///ws/a.py", "range": {"start": {"line": 1, "character": 0}}}},
    ])
    c2 = _fake_client(send_return=[
        {"name": "bar", "kind": 5, "location": {"uri": "file:///ws/b.py", "range": {"start": {"line": 2, "character": 0}}}},
    ])
    with svc._state_lock:
        svc._clients[("pyright", "/ws")] = c1
        svc._clients[("gopls", "/ws")] = c2

    out = svc.workspace_symbols_sync("f")

    assert sorted(s["name"] for s in out) == ["bar", "foo"]
    assert c1._send_request_with_retry.call_args.args[0] == "workspace/symbol"
    assert c1._send_request_with_retry.call_args.args[1] == {"query": "f"}


def test_workspace_symbols_path_hint_selects_one_server(svc, monkeypatch):
    client = _fake_client(send_return=[
        {"name": "baz", "kind": 6, "location": {"uri": "file:///ws/x.py", "range": {"start": {"line": 0, "character": 0}}}},
    ])
    monkeypatch.setattr(svc, "_get_or_spawn", AsyncMock(return_value=client))

    out = svc.workspace_symbols_sync("b", path="/ws/x.py")

    assert [s["name"] for s in out] == ["baz"]
    svc._get_or_spawn.assert_awaited()  # path hint forced server selection


def test_workspace_symbols_no_running_clients_returns_empty(svc):
    # _enabled but no clients up and no path → [] (servers spawn lazily), not an error
    assert svc.workspace_symbols_sync("anything") == []


# --------------------------------------------------------------------------
# graceful degrade — never raise
# --------------------------------------------------------------------------

def test_request_error_returns_empty(svc, monkeypatch):
    _inject(svc, monkeypatch, _fake_client(send_raises=RuntimeError("boom")))
    assert svc.request_definition_sync("/ws/b.py", 0, 0) == []


def test_null_result_returns_empty(svc, monkeypatch):
    _inject(svc, monkeypatch, _fake_client(send_return=None))
    assert svc.request_references_sync("/ws/b.py", 0, 0) == []


def test_disabled_for_file_returns_empty_without_request(svc, monkeypatch):
    monkeypatch.setattr(svc, "enabled_for", lambda fp: False)
    client = _fake_client(send_return={"uri": "file:///ws/a.py", "range": {"start": {"line": 0, "character": 0}}})
    monkeypatch.setattr(svc, "_get_or_spawn", AsyncMock(return_value=client))
    assert svc.document_symbols_sync("/ws/a.py") == []
    client._send_request_with_retry.assert_not_awaited()


def test_no_client_returns_empty(svc, monkeypatch):
    monkeypatch.setattr(svc, "enabled_for", lambda fp: True)
    monkeypatch.setattr(svc, "_get_or_spawn", AsyncMock(return_value=None))
    assert svc.document_symbols_sync("/ws/a.py") == []
