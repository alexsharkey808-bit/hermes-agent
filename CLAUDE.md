# CLAUDE.md — hermes-agent

This subtree is its own git repo. Its canonical, exhaustive dev guide is **`AGENTS.md`**
(this folder) — import order, the `scripts/run_tests.sh` test wrapper, dependency pinning,
the full file map, and the prompt-caching / profile rules all live there. Claude Code does
not auto-load `AGENTS.md` on its own, so it is imported below. Read it before any non-trivial
change in `hermes-agent/`.

@AGENTS.md

Workspace-wide rules (when this repo is checked out inside `~/.hermes`) live in the parent
`../CLAUDE.md`; don't duplicate them here.
