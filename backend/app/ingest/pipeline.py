"""parse → chunk → embed → index, with progress callbacks for the ingestion SSE stream."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from app.config import get_settings
from app.ingest import indexer
from app.ingest.chunker import chunk_pages
from app.ingest.embed import embed_passages
from app.ingest.pdf import parse_pdf
from app.models import Chunk, Document

logger = logging.getLogger("documind.ingest")

Progress = Callable[[str, float], None]
EMBED_BATCH = 16


def _noop(stage: str, progress: float) -> None:
    pass


def _title(parsed, filename: str) -> str:
    if parsed.title and len(parsed.title) > 3:
        return parsed.title[:200]
    for page in parsed.pages[:1]:
        for line in page.lines[:3]:
            if line.size > parsed.body_size:
                return line.text[:200]
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()[:200] or "Untitled"


def ingest(session: Session, doc: Document, data: bytes, progress: Progress = _noop) -> Document:
    """Ingests `data` into an existing (processing) Document row. Marks it ready or failed."""
    s = get_settings()
    try:
        progress("parse", 0.05)
        parsed = parse_pdf(data, max_pages=s.max_pages if doc.collection != "library" else None)
        doc.title, doc.pages = _title(parsed, doc.filename), len(parsed.pages)
        progress("chunk", 0.2)
        pieces = chunk_pages(parsed.pages, parsed.body_size, s.chunk_size, s.chunk_overlap)
        if not pieces:
            raise ValueError("No text found to index")
        recs = [
            indexer.ChunkRec(
                f"{doc.id}:{i}", doc.id, doc.title, doc.filename, c.page, c.heading, c.char_start, c.text
            )
            for i, c in enumerate(pieces)
        ]
        vectors = []
        for i in range(0, len(recs), EMBED_BATCH):
            progress("embed", 0.25 + 0.6 * i / len(recs))
            vectors.append(embed_passages([r.search_text for r in recs[i : i + EMBED_BATCH]]))
        indexer.save_vectors(doc.collection, doc.id, np.concatenate(vectors))
        progress("index", 0.9)
        for i, c in enumerate(pieces):
            session.add(
                Chunk(
                    id=f"{doc.id}:{i}",
                    doc_id=doc.id,
                    collection=doc.collection,
                    ordinal=i,
                    page=c.page,
                    heading=c.heading,
                    char_start=c.char_start,
                    text=c.text,
                )
            )
        doc.chunks, doc.status = len(pieces), "ready"
        session.add(doc)
        session.commit()
        indexer.rebuild(session, doc.collection)
        progress("done", 1.0)
    except Exception as e:
        session.rollback()
        logger.warning("Ingestion of %s failed: %s", doc.filename, e)
        doc.status, doc.error = "failed", str(e)[:300] if isinstance(e, ValueError) else "Ingestion failed"
        session.add(doc)
        session.commit()
        progress("failed", 1.0)
    session.refresh(doc)
    return doc


def seed_library(session: Session) -> int:
    """Ingests the sample PDFs into the shared read-only library (once)."""
    samples = get_settings().samples_dir
    added = 0
    existing = {d.filename for d in session.exec(select(Document).where(Document.collection == "library"))}
    for pdf in sorted(samples.glob("*.pdf")):
        if pdf.name in existing:
            continue
        data = pdf.read_bytes()
        doc = Document(collection="library", filename=pdf.name, title=pdf.stem, size_bytes=len(data))
        session.add(doc)
        session.commit()
        ingest(session, doc, data)
        added += 1
        logger.info("Seeded %s (%d chunks)", pdf.name, doc.chunks)
    return added
