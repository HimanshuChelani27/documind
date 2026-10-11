"""Per-collection search indexes: FAISS (dense) + bm25s (sparse), persisted under data/indexes/.

Each document's chunk vectors are stored once (vectors/<doc id>.npy) at ingestion. Adding or
deleting a document rebuilds the collection's FAISS and BM25 indexes from those stored vectors
and the chunk rows, which is cheap at demo scale and keeps every index exactly in sync.
"""

import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import bm25s
import faiss
import numpy as np
from sqlmodel import Session, select

from app.config import get_settings
from app.models import Chunk, Document


@dataclass(frozen=True)
class ChunkRec:
    id: str
    doc_id: str
    doc_title: str
    filename: str
    page: int
    heading: str | None
    char_start: int
    text: str

    @property
    def search_text(self) -> str:
        """Text that is embedded/indexed: the heading adds context the passage may lack."""
        if self.heading and not self.text.startswith(self.heading):
            return f"{self.heading}\n{self.text}"
        return self.text


@dataclass
class CollectionIndex:
    collection: str
    chunks: list[ChunkRec]
    vectors: np.ndarray
    dense: faiss.Index
    bm25: bm25s.BM25 | None

    def __len__(self) -> int:
        return len(self.chunks)


_cache: dict[str, CollectionIndex] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(collection: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(collection, threading.Lock())


def collection_dir(collection: str) -> Path:
    return get_settings().index_dir / collection


def vectors_path(collection: str, doc_id: str) -> Path:
    return collection_dir(collection) / "vectors" / f"{doc_id}.npy"


def save_vectors(collection: str, doc_id: str, vectors: np.ndarray) -> None:
    path = vectors_path(collection, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors.astype(np.float32))


def tokenize(texts: list[str]):
    return bm25s.tokenize(texts, lower=True, stopwords="en", show_progress=False)


def _load_chunks(session: Session, collection: str) -> list[ChunkRec]:
    rows = session.exec(
        select(Chunk, Document)
        .join(Document, Chunk.doc_id == Document.id)  # type: ignore[arg-type]
        .where(Chunk.collection == collection, Document.status == "ready")
        .order_by(Document.created_at, Chunk.doc_id, Chunk.ordinal)
    ).all()
    return [
        ChunkRec(c.id, c.doc_id, d.title, d.filename, c.page, c.heading, c.char_start, c.text)
        for c, d in rows
    ]


def _build(collection: str, chunks: list[ChunkRec]) -> CollectionIndex:
    dim = 384
    parts, by_doc = [], {}
    for rec in chunks:
        by_doc.setdefault(rec.doc_id, []).append(rec)
    for doc_id, recs in by_doc.items():
        vecs = np.load(vectors_path(collection, doc_id))
        if len(vecs) != len(recs):
            raise RuntimeError(f"Vector count mismatch for document {doc_id}")
        parts.append(vecs)
    vectors = np.concatenate(parts) if parts else np.zeros((0, dim), dtype=np.float32)
    if len(vectors):
        dim = vectors.shape[1]
    dense = faiss.IndexFlatIP(dim)
    if len(vectors):
        dense.add(vectors)
    bm25 = None
    if chunks:
        bm25 = bm25s.BM25()
        bm25.index(tokenize([c.search_text for c in chunks]), show_progress=False)
    return CollectionIndex(collection, chunks, vectors, dense, bm25)


def _persist(index: CollectionIndex) -> None:
    d = collection_dir(index.collection)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.dense, str(d / "dense.faiss"))
    (d / "ids.json").write_text(json.dumps([c.id for c in index.chunks]))
    if index.bm25 is not None:
        index.bm25.save(str(d / "bm25"))
    elif (d / "bm25").exists():
        shutil.rmtree(d / "bm25")


def rebuild(session: Session, collection: str) -> CollectionIndex:
    with _lock(collection):
        index = _build(collection, _load_chunks(session, collection))
        _persist(index)
        _cache[collection] = index
        return index


def _load_persisted(collection: str, chunks: list[ChunkRec]) -> CollectionIndex | None:
    d = collection_dir(collection)
    try:
        ids = json.loads((d / "ids.json").read_text())
        if ids != [c.id for c in chunks]:
            return None
        dense = faiss.read_index(str(d / "dense.faiss"))
        bm25 = bm25s.BM25.load(str(d / "bm25")) if chunks else None
        vectors = dense.reconstruct_n(0, dense.ntotal) if dense.ntotal else np.zeros((0, dense.d), "float32")
        return CollectionIndex(collection, chunks, vectors, dense, bm25)
    except (OSError, ValueError, RuntimeError):
        return None


def get_index(session: Session, collection: str) -> CollectionIndex:
    if collection in _cache:
        return _cache[collection]
    chunks = _load_chunks(session, collection)
    with _lock(collection):
        if collection in _cache:
            return _cache[collection]
        index = _load_persisted(collection, chunks)
        if index is None:
            index = _build(collection, chunks)
            _persist(index)
        _cache[collection] = index
        return index


def drop_collection(collection: str) -> None:
    with _lock(collection):
        _cache.pop(collection, None)
        shutil.rmtree(collection_dir(collection), ignore_errors=True)


def forget(collection: str | None = None) -> None:
    """Clear the in-memory cache (tests)."""
    if collection is None:
        _cache.clear()
    else:
        _cache.pop(collection, None)
