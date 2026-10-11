"""PDF → per-page text with line-level font sizes (used to spot headings).

Uses pypdf (BSD licensed). pypdf's text visitor reports the font size of every text run, and the
runs concatenate to exactly the extracted text, so each line can be tagged with its largest font.
"""

import contextlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from pypdf import PdfReader

PDF_MAGIC = b"%PDF-"


class PdfError(ValueError):
    pass


@dataclass
class Line:
    text: str
    size: float
    bold: bool


@dataclass
class Page:
    number: int  # 1-based
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class ParsedPdf:
    title: str | None
    pages: list[Page]
    body_size: float


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(PDF_MAGIC)


def _scale(m) -> float:
    return math.hypot(m[0], m[1]) or 1.0


def _page_lines(page) -> list[Line]:
    runs: list[tuple[str, float, bool]] = []

    def visit(text, cm, tm, font_dict, font_size):
        if text:
            name = str((font_dict or {}).get("/BaseFont", ""))
            runs.append((text, round((font_size or 0) * _scale(tm) * _scale(cm), 1), "Bold" in name))

    page.extract_text(visitor_text=visit)
    lines: list[Line] = []
    cur, size, bold = "", 0.0, True
    for text, s, b in runs:
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append(Line(cur, size, bold))
                cur, size, bold = "", 0.0, True
            if part.strip():
                size = max(size, s)
                bold = bold and b
            cur += part
    lines.append(Line(cur, size, bold))
    out = []
    for line in lines:
        text = re.sub(r"[ \t ]+", " ", line.text).strip()
        if text:
            out.append(Line(text, line.size, line.bold and bool(text)))
    return out


def _strip_running_lines(pages: list[Page]) -> None:
    """Drop headers/footers: short lines that repeat (ignoring digits) on most pages."""
    if len(pages) < 3:
        return

    def key(text: str) -> str:
        return re.sub(r"\d+", "#", text.lower())

    seen: Counter[str] = Counter()
    for p in pages:
        edge = p.lines[:2] + p.lines[-2:]
        seen.update({key(line.text) for line in edge if len(line.text) < 100})
    running = {k for k, n in seen.items() if n >= max(3, len(pages) * 0.6)}
    for p in pages:
        p.lines = [line for line in p.lines if key(line.text) not in running]


def parse_pdf(data: bytes, max_pages: int | None = None) -> ParsedPdf:
    if not is_pdf(data):
        raise PdfError("Not a PDF file")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfError("Encrypted PDFs are not supported")
        n = len(reader.pages)
    except PdfError:
        raise
    except Exception as e:  # pypdf raises a variety of errors on malformed files
        raise PdfError(f"Could not read PDF: {type(e).__name__}") from e
    if max_pages and n > max_pages:
        raise PdfError(f"PDF has {n} pages; the limit is {max_pages}")
    pages = [Page(i + 1, _page_lines(p)) for i, p in enumerate(reader.pages)]
    if not any(p.lines for p in pages):
        raise PdfError("No extractable text (scanned PDFs need OCR, which DocuMind doesn't do)")

    _strip_running_lines(pages)
    sizes: Counter[float] = Counter()
    for p in pages:
        for line in p.lines:
            sizes[line.size] += len(line.text)
    body = sizes.most_common(1)[0][0] if sizes else 10.0

    meta_title = None
    with contextlib.suppress(Exception):  # broken metadata shouldn't fail ingestion
        meta_title = (reader.metadata or {}).get("/Title")
    return ParsedPdf(title=str(meta_title).strip() if meta_title else None, pages=pages, body_size=body)
