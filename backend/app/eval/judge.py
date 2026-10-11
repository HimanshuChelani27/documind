"""Answer quality. With an LLM key: LLM-as-judge faithfulness and answer relevance. Without one:
clearly labelled embedding-based proxies, so the keyless demo still shows something meaningful."""

import json
import re

import numpy as np

from app.answer import llm
from app.answer.citations import CITE
from app.answer.extractive import split_sentences
from app.ingest.embed import embed_passages, embed_queries
from app.retrieval.strategies import Hit

SUPPORT_THRESHOLD = 0.8

JUDGE_SYSTEM = """You grade answers produced by a document question-answering system. Reply with JSON only."""

JUDGE_PROMPT = """Context passages:
{context}

Question: {question}
Answer: {answer}

1. Split the answer into its factual claims. Count how many there are and how many are fully supported by the context passages.
2. Rate how directly and completely the answer addresses the question, from 1 (not at all) to 5 (fully).
A reply saying the information could not be found has 0 claims.

Return exactly: {{"claims": <int>, "supported": <int>, "relevance": <1-5>}}"""


def _strip_citations(text: str) -> str:
    return CITE.sub("", text).strip()


def proxy_metrics(question: str, answer: str, hits: list[Hit]) -> dict[str, float]:
    """Keyless proxies: share of answer sentences closely matched by a retrieved sentence
    (cosine ≥ 0.8), and question–answer embedding similarity."""
    sentences = split_sentences(_strip_citations(answer))
    source_sents = [s for h in hits for s in split_sentences(h.text)]
    if not sentences or not source_sents:
        return {"supported_share": 0.0, "qa_similarity": 0.0}
    a = embed_passages(sentences)
    src = embed_passages(source_sents)
    supported = float(np.mean((a @ src.T).max(axis=1) >= SUPPORT_THRESHOLD))
    q = embed_queries([question])[0]
    qa = float(embed_passages([_strip_citations(answer)])[0] @ q)
    return {"supported_share": round(supported, 4), "qa_similarity": round(qa, 4)}


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("Judge returned no JSON")
    return json.loads(m.group(0))


async def judge_metrics(
    provider: llm.Provider, question: str, answer: str, hits: list[Hit]
) -> dict[str, float]:
    context = "\n\n".join(f"[{n}] {h.text}" for n, h in enumerate(hits, start=1))
    reply = await llm.complete(
        provider,
        JUDGE_SYSTEM,
        JUDGE_PROMPT.format(context=context, question=question, answer=_strip_citations(answer)),
        max_tokens=200,
    )
    data = _parse_json(reply)
    claims = max(0, int(data.get("claims", 0)))
    supported = min(claims, max(0, int(data.get("supported", 0))))
    relevance = min(5, max(1, int(data.get("relevance", 1))))
    return {
        "faithfulness": round(supported / claims, 4) if claims else 1.0,
        "answer_relevance": round((relevance - 1) / 4, 4),
    }
