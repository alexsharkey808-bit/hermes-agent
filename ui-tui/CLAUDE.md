# CLAUDE.md — ui-tui

Ink/React terminal UI behind `hermes --tui`. Dev commands and the full layout are in
`README.md` (imported below). Process model: TypeScript (Ink) owns the screen; Python
(`../tui_gateway/`) owns sessions, tools, and model calls; they talk over stdio
newline-delimited JSON-RPC.

@README.md

**CRITICAL — never re-implement the primary chat experience in React.** The main transcript,
composer/input flow, and PTY-backed terminal belong to the embedded `hermes --tui` and surface
in the dashboard automatically. Build only complementary sidebar widgets, inspectors, and
status panels here — never a replacement transcript or composer.

Workspace TUI notes are also in `../../CLAUDE.md` ("TUI Development").
