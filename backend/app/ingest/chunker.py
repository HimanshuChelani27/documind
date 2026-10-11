"""Structure-aware chunking that never crosses a page boundary.

Each page is split into chunks of about `size` characters. Split points are chosen by priority:
a heading (larger font than the body) → a paragraph end → a sentence end → a line break → a space.
Chunks inside the same section overlap by about `overlap` characters (snapped to a sentence or
word start) so a fact cut at a boundary still appears whole in one chunk. Every chunk keeps the
page number, the nearest heading and its character offset in the page text, so citations can
point at the exact passage.
"""

import re
from dataclasses import dataclass

from app.ingest.pdf import Line, Page

SENTENCE_END = re.compile(r"[.!?:;][\"')\]]?(?=\s)")


@dataclass
class ChunkData:
    page: int
    heading: str | None
    char_start: int
    text: str


def is_heading(line: Line, body_size: float) -> bool:
    t = line.text
    if len(t) > 90 or len(t) < 3 or t.endswith((".", ",", ";")) or t.startswith(("-", "•", "|")):
        return False
    return line.size >= body_size + 0.75 or (line.bold and line.size >= body_size and len(t) < 60)


def _boundaries(text: str, line_starts: list[int], heading_starts: set[int]) -> dict[int, int]:
    """Map candidate split position → priority (higher is better). Position = start of next chunk."""
    cands: dict[int, int] = {}
    for m in re.finditer(r" ", text):
        cands[m.end()] = 1
    for pos in line_starts:
        cands[pos] = 2
    for m in SENTENCE_END.finditer(text):
        end = m.end()
        while end < len(text) and text[end] in " \n":
            end += 1
        nl = "\n" in text[m.end() : end]
        cands[end] = max(cands.get(end, 0), 4 if nl else 3)
    for pos in heading_starts:
        cands[pos] = 5
    return cands


def chunk_page(page: Page, body_size: float, size: int, overlap: int, heading: str | None):
    """Returns (chunks, last heading seen) so headings carry across pages."""
    text = page.text
    if not text.strip():
        return [], heading
    line_starts, headings, pos = [], {}, 0
    for line in page.lines:
        line_starts.append(pos)
        if is_heading(line, body_size):
            headings[pos] = line.text
        pos += len(line.text) + 1
    cands = _boundaries(text, line_starts[1:], set(headings) - {0})

    def heading_at(p: int) -> str | None:
        found = heading
        for hp in sorted(headings):
            if hp <= p:
                found = headings[hp]
        return found

    chunks: list[ChunkData] = []
    start = 0
    while start < len(text):
        if len(text) - start <= size:
            end = len(text)
        else:
            lo, hi = start + size // 2, start + size
            window = [(prio, p) for p, prio in cands.items() if lo <= p <= hi]
            # best priority, then the latest position at that priority
            end = max(window)[1] if window else hi
        body = text[start:end].strip()
        if body:
            offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            chunks.append(ChunkData(page.number, heading_at(offset), offset, body))
        if end >= len(text):
            break
        if cands.get(end) == 5:  # new section: no overlap
            start = end
            continue
        # Overlap: step back, then snap forward to a sentence start (or at least a word start).
        back = max(start + 1, end - overlap)
        snap = [p for p, prio in cands.items() if back <= p < end and prio >= 3]
        words = [p for p, prio in cands.items() if back <= p < end and prio >= 1]
        start = min(snap) if snap else (min(words) if words else end)
    last = heading_at(len(text))
    return chunks, last


def chunk_pages(pages: list[Page], body_size: float, size: int = 800, overlap: int = 120) -> list[ChunkData]:
    out: list[ChunkData] = []
    heading = None
    for page in pages:
        chunks, heading = chunk_page(page, body_size, size, overlap, heading)
        out.extend(chunks)
    return out
