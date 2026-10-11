"""Keyless answers: pick the source passages that best answer the question and cite them.

Each top retrieved chunk is cut into candidate units. Prose becomes sentences; table and list
regions (where PDF extraction wraps cells over several lines) become short windows of lines,
labelled with the region's first line (usually the table header). The cross-encoder that reranks
chunks then scores every unit against the question, and the best non-overlapping ones are returned
in source order with their citations. Nothing is generated, so the answer can only contain what the
documents say."""

import re
from dataclasses import dataclass

import numpy as np

from app.ingest.embed import rerank_scores
from app.retrieval.strategies import Hit

MIN_RELEVANCE = -6.0  # cross-encoder logit below which nothing really answers the question
MARGIN = 1.5  # keep units scoring within this of the best one
MAX_UNITS = 3
SOURCES_CONSIDERED = 3
WINDOW_LINES = 3
WINDOW_CHARS = 110

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_END = (".", "!", "?", ":", ";")


@dataclass(frozen=True)
class Unit:
    text: str
    first: int  # line span in the reflowed chunk, for overlap checks
    last: int
    sentence: bool


def reflow(text: str) -> list[str]:
    """Re-joins lines that a PDF wrapped mid-sentence; short lines (headings, table rows) stay."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    width = max(len(ln) for ln in lines)
    out: list[str] = []
    joining = False
    for ln in lines:
        if joining and out:
            out[-1] += " " + ln
        else:
            out.append(ln)
        joining = width >= 60 and len(ln) >= 0.7 * width and not ln.endswith(_END)
    return out


def _is_prose(line: str) -> bool:
    return len(line) >= 40 and line.endswith(_END)


def units(text: str) -> list[Unit]:
    lines = [ln.lstrip("-• ").strip() for ln in reflow(text)]
    out: list[Unit] = []
    i = 0
    while i < len(lines):
        if _is_prose(lines[i]):
            out.extend(Unit(s, i, i, True) for s in _SENT.split(lines[i]) if len(s) >= 12)
            i += 1
            continue
        j = i
        while j < len(lines) and not _is_prose(lines[j]):
            j += 1
        label, rows = lines[i], range(i + 1, j)
        if not rows and lines[i].endswith(_END) and len(lines[i]) >= 12:
            out.append(Unit(lines[i], i, i, True))  # a short sentence on its own
        for r in rows:
            window, last = lines[r], r
            while (
                last + 1 < j
                and last - r + 1 < WINDOW_LINES
                and len(window) < WINDOW_CHARS
                and not window.endswith(_END)
            ):
                last += 1
                window += " " + lines[last]
            if re.search(r"[A-Za-z]", window) and len(window) >= 8:
                out.append(Unit(f"{window} ({label})", r, last, False))
        i = j
    return out


def split_sentences(text: str) -> list[str]:
    return [u.text for u in units(text)]


def extractive_answer(question: str, hits: list[Hit]) -> tuple[str, float]:
    """Returns (answer with [n] citations, best relevance score)."""
    cands: list[tuple[int, int, Unit]] = []  # (source number, position, unit)
    per_source: dict[int, list[Unit]] = {}
    for n, h in enumerate(hits[:SOURCES_CONSIDERED], start=1):
        per_source[n] = units(h.text)
        cands.extend((n, i, u) for i, u in enumerate(per_source[n]))
    if not cands:
        return "I couldn't find anything relevant in the selected documents.", float("-inf")
    rel = rerank_scores(question, [c[2].text for c in cands])
    scores = [r + 0.2 * (SOURCES_CONSIDERED - n) for r, (n, _, _) in zip(rel, cands, strict=True)]
    best = max(rel)
    if best < MIN_RELEVANCE:
        return "I couldn't find a passage that answers this in the selected documents.", best

    top = max(scores)
    chosen: list[tuple[int, int, str]] = []
    taken: dict[int, set[int]] = {}
    for i in np.argsort(scores)[::-1]:
        if len(chosen) == MAX_UNITS or scores[i] < top - MARGIN:
            break
        n, pos, u = cands[i]
        span = set(range(u.first, u.last + 1))
        if not u.sentence and span <= taken.get(n, set()):
            continue  # table window already covered by a better one
        text = u.text
        # A short topic sentence ("Q2 was difficult.") is usually followed by the actual reason.
        nxt = per_source[n][pos + 1] if pos + 1 < len(per_source[n]) else None
        if u.sentence and len(text) < 70 and nxt and nxt.sentence:
            text = f"{text} {nxt.text}"
        taken.setdefault(n, set()).update(span)
        chosen.append((n, pos, text))
    chosen.sort()
    answer, seen = [], set()
    for n, _, text in chosen:
        if text not in seen:
            seen.add(text)
            answer.append(f"{text.rstrip()} [{n}]")
    return " ".join(answer), best
