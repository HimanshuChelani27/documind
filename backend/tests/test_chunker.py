from app.ingest.chunker import chunk_pages, is_heading
from app.ingest.pdf import Line, Page, parse_pdf
from tests.conftest import SAMPLES


def _page(n: int, paragraphs: list[str], heading: str | None = None) -> Page:
    lines = [Line(heading, 14.0, True)] if heading else []
    lines += [Line(p, 10.0, False) for p in paragraphs]
    return Page(n, lines)


def test_chunks_keep_page_numbers_and_never_cross_pages():
    sentence = "The pump must be connected through a residual current device of thirty milliamps. "
    pages = [_page(1, [sentence * 12], "Safety"), _page(2, [sentence * 3], "Electrical")]
    chunks = chunk_pages(pages, 10.0, size=400, overlap=80)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert len(c.text) <= 400
        page_text = pages[c.page - 1].text
        assert page_text[c.char_start : c.char_start + len(c.text)] == c.text  # offset is exact
    assert chunks[-1].heading == "Electrical"


def test_overlap_between_consecutive_chunks_in_a_section():
    text = " ".join(f"Sentence number {i} talks about policy item {i}." for i in range(40))
    chunks = chunk_pages([_page(1, [text])], 10.0, size=300, overlap=80)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.char_start < a.char_start + len(a.text)  # b starts inside a
        assert b.text.startswith("Sentence")  # snapped to a sentence start


def test_heading_starts_a_new_chunk_without_overlap():
    body = "Plain words in a paragraph that keeps going for a while. " * 6
    page = Page(1, [Line(body, 10, False), Line("Travel rules", 12, True), Line(body, 10, False)])
    chunks = chunk_pages([page], 10.0, size=450, overlap=100)
    starts = [c for c in chunks if c.text.startswith("Travel rules")]
    assert starts and starts[0].heading == "Travel rules"


def test_heading_detection():
    assert is_heading(Line("6. Business travel", 14, True), 10)
    assert is_heading(Line("Rules for the per diem", 11, True), 10)
    assert not is_heading(Line("Trip type Per diem (per day)", 9, True), 10)  # bold table header
    assert not is_heading(Line("A normal sentence that ends.", 10, False), 10)


def test_sample_pdf_parsing_strips_running_footer():
    parsed = parse_pdf((SAMPLES / "northwind_employee_handbook.pdf").read_bytes())
    assert len(parsed.pages) == 12
    assert parsed.body_size == 10.0
    assert all("Page " not in line.text for p in parsed.pages for line in p.lines[-1:])
    chunks = chunk_pages(parsed.pages, parsed.body_size)
    per_diem = [c for c in chunks if "USD 85" in c.text]
    assert per_diem and per_diem[0].page == 7
