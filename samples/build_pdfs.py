"""Build the sample PDFs from the markdown sources in samples/src.

Each `---page---` marker starts a new PDF page, so the page numbers that eval/gold.jsonl
refers to are fixed by the source, not by layout. The script fails if any authored page
overflows onto a second physical page.

    cd backend && uv run python ../samples/build_pdfs.py
"""

import re
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

HERE = Path(__file__).parent
SRC = HERE / "src"
PAGE_BREAK = "---page---"
TEAL = (31, 122, 99)


class Doc(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.set_margins(20, 18, 20)
        self.set_auto_page_break(True, margin=18)
        self.set_title(title)
        self.set_author("DocuMind sample corpus")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, self.doc_title, align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R")


def _table(pdf: Doc, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r"-{3,}", c) for c in r)]
    pdf.set_font("Helvetica", size=9)
    pdf.set_draw_color(200, 200, 200)
    with pdf.table(
        line_height=5.2,
        headings_style=FontFace(emphasis="BOLD", fill_color=(232, 244, 240)),
        padding=1.4,
    ) as table:
        for r in cells:
            row = table.row()
            for c in r:
                row.cell(c.replace("**", ""))
    pdf.ln(2.5)


def render_page(pdf: Doc, block: str) -> None:
    lines = block.strip("\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            _table(pdf, rows)
            continue
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 19)
            pdf.set_text_color(*TEAL)
            pdf.multi_cell(0, 9, line[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(1.5)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(20, 20, 20)
            pdf.multi_cell(0, 7, line[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*TEAL)
            pdf.multi_cell(0, 6, line[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif m := re.match(r"^(- |\d+\. )(.*)$", line):
            bullet = "-" if m.group(1) == "- " else m.group(1).strip()
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(pdf.l_margin + 3)
            pdf.cell(6, 5.2, bullet)
            pdf.multi_cell(0, 5.2, m.group(2), markdown=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5.2, line, markdown=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.2)
        i += 1


def build(src: Path) -> Path:
    text = src.read_text()
    title = text.split("\n", 1)[0].lstrip("# ").strip()
    pages = text.split(PAGE_BREAK)
    pdf = Doc(title)
    for n, block in enumerate(pages, start=1):
        pdf.add_page()
        render_page(pdf, block)
        if pdf.page_no() != n:
            sys.exit(f"{src.name}: authored page {n} overflows; shorten it")
    out = HERE / f"{src.stem}.pdf"
    pdf.output(str(out))
    print(f"{out.name}: {len(pages)} pages")
    return out


if __name__ == "__main__":
    for md in sorted(SRC.glob("*.md")):
        build(md)
