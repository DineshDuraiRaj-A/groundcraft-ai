"""
Lightweight TF-IDF retrieval.

Deliberately avoids sentence-transformers / embedding APIs so that:
  1. Retrieval works with zero external calls and zero cost.
  2. Cold starts on free hosting tiers (Render/Railway) stay fast — no large
     model download on boot.
  3. The only paid/rate-limited call in the whole app is the final LLM
     generation step, which is exactly what the "token & cost" lesson wants
     to highlight.

This is intentionally simple — good enough to demonstrate what "semantic-ish"
retrieval feels like for teaching purposes. Swap in real embeddings later if
you want higher retrieval quality (see README "v2 roadmap").
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class TfidfIndex:
    """A tiny in-memory TF-IDF index over a list of chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._doc_tokens: list[list[str]] = [_tokenize(c.text) for c in chunks]
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        self._n_docs = max(1, len(chunks))
        self._doc_vectors: list[Counter[str]] = [
            self._tfidf_vector(tokens) for tokens in self._doc_tokens
        ]

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        # +1 smoothing avoids div-by-zero and zeroing out terms seen everywhere
        return math.log((1 + self._n_docs) / (1 + df)) + 1

    def _tfidf_vector(self, tokens: list[str]) -> Counter[str]:
        tf = Counter(tokens)
        total = max(1, len(tokens))
        return Counter({term: (count / total) * self._idf(term) for term, count in tf.items()})

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, question: str, top_k: int = 3) -> list[ScoredChunk]:
        q_tokens = _tokenize(question)
        q_vec = self._tfidf_vector(q_tokens)
        scored = [
            ScoredChunk(chunk=chunk, score=self._cosine(q_vec, doc_vec))
            for chunk, doc_vec in zip(self.chunks, self._doc_vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


def classify_confidence(top_score: float | None) -> str:
    """
    Feature 9: confidence/uncertainty signal.

    Turns the top retrieval similarity score into a plain-language
    confidence level. Thresholds are heuristic (tuned for this TF-IDF
    index, not universal) — the point is to demonstrate that a system
    *can* self-flag uncertainty instead of always sounding equally sure.
    """
    if top_score is None:
        return "none"
    if top_score >= 0.30:
        return "high"
    if top_score >= 0.12:
        return "medium"
    return "low"
