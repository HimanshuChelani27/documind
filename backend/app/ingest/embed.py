"""Embedding and reranking models (fastembed, ONNX runtime), loaded once per process.

fastembed runs the same model families as sentence-transformers through ONNX, without torch,
which keeps the whole backend under ~450 MB of RAM.
"""

import threading
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import get_settings

_lock = threading.Lock()


def _kwargs() -> dict:
    s = get_settings()
    kw: dict = {}
    if s.model_threads:
        kw["threads"] = s.model_threads
    if s.model_cache_dir:
        kw["cache_dir"] = str(s.model_cache_dir)
    return kw


@lru_cache
def get_embedder() -> TextEmbedding:
    with _lock:
        return TextEmbedding(get_settings().embed_model, **_kwargs())


@lru_cache
def get_reranker() -> TextCrossEncoder:
    with _lock:
        return TextCrossEncoder(get_settings().rerank_model, **_kwargs())


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _normalise(np.stack(list(get_embedder().passage_embed(texts, batch_size=32))))


def embed_queries(texts: list[str]) -> np.ndarray:
    # BGE models prepend a retrieval instruction to queries; query_embed handles that.
    return _normalise(np.stack(list(get_embedder().query_embed(texts))))


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    return [float(s) for s in get_reranker().rerank(query, passages, batch_size=32)]


def warm_up() -> None:
    """Load (and download, on first run) both models so the first request isn't slow."""
    embed_queries(["warm up"])
    rerank_scores("warm up", ["warm up"])
