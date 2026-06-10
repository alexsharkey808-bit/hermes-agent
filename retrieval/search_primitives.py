"""Search primitives — the enforced security boundary (rr01-F1, Threat Model).

Model-authored orchestration is UNTRUSTED (its inputs are attacker-controlled web content). We do
NOT rely on a Python sandbox as the boundary. Instead, safety is enforced HERE, where code touches
the outside world: every `web_extract` carries an egress allow-list and per-run rate + byte caps.
Even a prompt-injected orchestrator can only call these primitives, capped.
"""

from dataclasses import dataclass
from urllib.parse import urlparse


class EgressDenied(Exception):
    """Raised when a primitive call violates the egress allow-list or a per-run cap."""


@dataclass
class Caps:
    max_calls: int = 50
    max_bytes: int = 5_000_000


def make_primitives(web_client, *, allow: list[str], caps: Caps) -> dict:
    """Build `{web_search, web_extract}` primitives carrying the egress allow-list + per-run caps.

    State (call count, byte total) is closed over per call to `make_primitives`, so each research
    run gets a fresh, independent budget.
    """
    state = {"calls": 0, "bytes": 0}

    def _guard(url: str) -> None:
        host = urlparse(url).hostname or ""
        if not any(host == a or host.endswith("." + a) for a in allow):
            raise EgressDenied(f"host not allowed: {host}")
        state["calls"] += 1
        if state["calls"] > caps.max_calls:
            raise EgressDenied("rate cap exceeded")

    def web_search(query: str, *, k: int = 10) -> list[dict]:
        return web_client.search(query, k=k)

    def web_extract(url: str) -> str:
        _guard(url)
        doc = web_client.fetch(url)
        state["bytes"] += len(doc)
        if state["bytes"] > caps.max_bytes:
            raise EgressDenied("byte cap exceeded")
        return doc

    return {"web_search": web_search, "web_extract": web_extract}
