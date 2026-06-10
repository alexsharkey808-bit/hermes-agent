"""Local-corpus retrieval (DCI, Phase 1).

grep-first with an *empirical* vector fallback: try ripgrep; escalate to vector/semantic
search only when grep returns zero or low-confidence hits (rr01-F4). No guessed file-count
threshold.
"""

import json
import subprocess
from dataclasses import dataclass


@dataclass
class Hit:
    path: str
    line_number: int
    text: str
    score: float = 1.0


def grep_corpus(query: str, *, root: str, max_results: int = 100) -> list[Hit]:
    """ripgrep over `root` for the LITERAL `query`.

    Uses an args list + `-F` fixed-string (never shell=True), so query metacharacters
    cannot inject a regex or a shell command. `--max-count` bounds matches *per file*;
    the in-loop break enforces the *global* `max_results` cap across files.
    """
    cmd = ["rg", "--json", "-F", "--smart-case", "--max-count", str(max_results),
           "-e", query, root]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode not in (0, 1):  # 1 = no matches, not an error
        raise RuntimeError(f"ripgrep failed ({proc.returncode}): {proc.stderr.strip()}")
    hits: list[Hit] = []
    for line in proc.stdout.splitlines():
        evt = json.loads(line)
        if evt.get("type") != "match":
            continue
        d = evt["data"]
        hits.append(Hit(path=d["path"]["text"], line_number=d["line_number"],
                        text=d["lines"]["text"].rstrip("\n")))
        if len(hits) >= max_results:
            break
    return hits


def needs_vector_fallback(grep_hits: list[dict], *, min_hits: int = 1,
                          min_score: float = 0.5) -> bool:
    """Empirical fallback (rr01-F4): escalate to vector search only when grep returns
    too few strong hits.

    A hit is "strong" if its score is >= min_score (a hit with no score is treated as a
    confident literal match, score 1.0). Returns True when fewer than `min_hits` strong
    hits remain -> caller should escalate to vector/semantic search.
    """
    strong = [h for h in grep_hits if h.get("score", 1.0) >= min_score]
    return len(strong) < min_hits
