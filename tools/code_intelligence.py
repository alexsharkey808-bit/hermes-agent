#!/usr/bin/env python3
"""Code-intelligence tools — semantic navigation over Hermes's existing LSP.

Four read-only tools (`find_definition`, `find_references`, `document_symbols`,
`workspace_symbols`) that drive the language servers Hermes already runs
(`agent/lsp/`, pyright/gopls/rust-analyzer/tsserver) via the additive
`LSPService.request_*_sync` methods. No new code-intelligence engine.

Gated by ``code_intelligence.enabled`` (default ON; ``lsp.enabled`` is the master
gate) and GRACEFULLY DEGRADING: no language server, a non-git workspace, a disabled
LSP, or a timeout yields ``{"ok": false, "reason": ...}`` — never an exception, never a
blocked call. Handlers return JSON strings.

Index convention (the classic off-by-one trap, made explicit): the agent-facing API is
**1-indexed** for ``line`` and ``character`` (matching `read_file`'s displayed numbers).
The handlers convert to the LSP's 0-indexed positions on the way in, and back to
1-indexed on the way out. The `LSPService.request_*_sync` methods speak pure LSP
0-indexing.
"""

import json
import logging

from agent.lsp import get_service
from tools.registry import registry

logger = logging.getLogger(__name__)

_MAX_RESULTS = 200

# LSP SymbolKind (spec §3.17.11) — int → human name for the agent.
_SYMBOL_KIND = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class", 6: "Method",
    7: "Property", 8: "Field", 9: "Constructor", 10: "Enum", 11: "Interface",
    12: "Function", 13: "Variable", 14: "Constant", 15: "String", 16: "Number",
    17: "Boolean", 18: "Array", 19: "Object", 20: "Key", 21: "Null", 22: "EnumMember",
    23: "Struct", 24: "Event", 25: "Operator", 26: "TypeParameter",
}

_UNAVAILABLE = json.dumps({
    "ok": False,
    "reason": "code intelligence unavailable (no language server, not a git workspace, or lsp disabled)",
})


def _code_intelligence_enabled() -> bool:
    """check_fn: expose the tools only when code_intelligence.enabled is truthy."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        ci = cfg.get("code_intelligence") if isinstance(cfg, dict) else None
        return bool((ci or {}).get("enabled", True))
    except Exception:  # noqa: BLE001
        return False


def _to_zero(value) -> int:
    """1-indexed agent input → 0-indexed LSP (clamped at 0)."""
    try:
        return max(0, int(value) - 1)
    except (TypeError, ValueError):
        return 0


def _locations_payload(results: list) -> str:
    capped = results[:_MAX_RESULTS]
    out = [
        {
            "path": r.get("path"),
            "line": int(r.get("line", 0)) + 1,          # 0-indexed → 1-indexed
            "character": int(r.get("character", 0)) + 1,
        }
        for r in capped
    ]
    payload = {"ok": True, "results": out}
    if len(results) > _MAX_RESULTS:
        payload["truncated"] = True
        payload["total"] = len(results)
    return json.dumps(payload)


def _symbols_payload(results: list) -> str:
    capped = results[:_MAX_RESULTS]
    out = []
    for r in capped:
        entry = {
            "name": r.get("name"),
            "kind": _SYMBOL_KIND.get(r.get("kind"), r.get("kind")),
            "line": int(r.get("line", 0)) + 1,
            "character": int(r.get("character", 0)) + 1,
        }
        if r.get("path") is not None:
            entry["path"] = r["path"]
        if r.get("container"):
            entry["container"] = r["container"]
        out.append(entry)
    payload = {"ok": True, "results": out}
    if len(results) > _MAX_RESULTS:
        payload["truncated"] = True
        payload["total"] = len(results)
    return json.dumps(payload)


def find_definition(args: dict) -> str:
    svc = get_service()
    if svc is None:
        return _UNAVAILABLE
    try:
        return _locations_payload(
            svc.request_definition_sync(
                args.get("path") or "", _to_zero(args.get("line")), _to_zero(args.get("character"))
            )
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("find_definition failed: %s", e)
        return json.dumps({"ok": False, "reason": str(e)[:200]})


def find_references(args: dict) -> str:
    svc = get_service()
    if svc is None:
        return _UNAVAILABLE
    try:
        return _locations_payload(
            svc.request_references_sync(
                args.get("path") or "",
                _to_zero(args.get("line")),
                _to_zero(args.get("character")),
                include_declaration=bool(args.get("include_declaration", True)),
            )
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("find_references failed: %s", e)
        return json.dumps({"ok": False, "reason": str(e)[:200]})


def document_symbols(args: dict) -> str:
    svc = get_service()
    if svc is None:
        return _UNAVAILABLE
    try:
        return _symbols_payload(svc.document_symbols_sync(args.get("path") or ""))
    except Exception as e:  # noqa: BLE001
        logger.debug("document_symbols failed: %s", e)
        return json.dumps({"ok": False, "reason": str(e)[:200]})


def workspace_symbols(args: dict) -> str:
    svc = get_service()
    if svc is None:
        return _UNAVAILABLE
    try:
        return _symbols_payload(
            svc.workspace_symbols_sync(args.get("query") or "", path=args.get("path") or None)
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("workspace_symbols failed: %s", e)
        return json.dumps({"ok": False, "reason": str(e)[:200]})


_POSITION_PARAMS = {
    "path": {"type": "string", "description": "Absolute or workspace-relative path to the source file."},
    "line": {"type": "integer", "description": "1-indexed line number (as shown by read_file) of the symbol."},
    "character": {"type": "integer", "description": "1-indexed column of the symbol on that line."},
}

FIND_DEFINITION_SCHEMA = {
    "name": "find_definition",
    "description": (
        "Jump to where the symbol at a file position is DEFINED, using the language server "
        "(semantic, not text search). line/character are 1-indexed. Returns "
        "{ok, results:[{path, line, character}]} (1-indexed), or {ok:false, reason} when no "
        "language server is available."
    ),
    "parameters": {"type": "object", "properties": dict(_POSITION_PARAMS), "required": ["path", "line", "character"]},
}

FIND_REFERENCES_SCHEMA = {
    "name": "find_references",
    "description": (
        "Find all references to the symbol at a file position across the workspace (semantic). "
        "line/character are 1-indexed. Returns {ok, results:[{path, line, character}]} (1-indexed, "
        "capped at 200 with a truncated flag), or {ok:false, reason} when unavailable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_POSITION_PARAMS,
            "include_declaration": {
                "type": "boolean",
                "description": "Include the declaration itself in the results (default true).",
            },
        },
        "required": ["path", "line", "character"],
    },
}

DOCUMENT_SYMBOLS_SCHEMA = {
    "name": "document_symbols",
    "description": (
        "List the symbols (classes, functions, methods, variables) defined in a file, via the "
        "language server. Returns {ok, results:[{name, kind, line, character, container?}]} "
        "(1-indexed; nested symbols flattened with their container), or {ok:false, reason}."
    ),
    "parameters": {
        "type": "object",
        "properties": {"path": _POSITION_PARAMS["path"]},
        "required": ["path"],
    },
}

WORKSPACE_SYMBOLS_SCHEMA = {
    "name": "workspace_symbols",
    "description": (
        "Search the whole workspace for symbols matching a query string (semantic, fuzzy). "
        "Returns {ok, results:[{name, kind, path, line, character, container?}]} (1-indexed). "
        "Only covers language servers already running (they start on first file open) unless you "
        "pass a 'path' hint to select/start a specific language's server; empty results are "
        "{ok:true, results:[]}, not an error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Symbol name (or fragment) to search for."},
            "path": {
                "type": "string",
                "description": "Optional file path hint to pick which language server answers.",
            },
        },
        "required": ["query"],
    },
}

for _schema, _handler, _emoji in (
    (FIND_DEFINITION_SCHEMA, find_definition, "🧭"),
    (FIND_REFERENCES_SCHEMA, find_references, "🔗"),
    (DOCUMENT_SYMBOLS_SCHEMA, document_symbols, "🗂️"),
    (WORKSPACE_SYMBOLS_SCHEMA, workspace_symbols, "🔭"),
):
    registry.register(
        name=_schema["name"],
        toolset="code_intelligence",
        schema=_schema,
        handler=(lambda h: lambda args, **kw: h(args))(_handler),
        check_fn=_code_intelligence_enabled,
        is_async=False,
        emoji=_emoji,
    )
