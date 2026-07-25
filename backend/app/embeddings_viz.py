"""
Feature 4: Embeddings similarity, visualized.

Honest note on the approach: true semantic similarity ("dog" is close to
"puppy") can't come from word-overlap methods like TF-IDF, because those
two words share no characters or context — you need either a real trained
embedding model or hand-crafted feature vectors. Calling a live embedding
API would break this app's "zero external calls except the final LLM
generation" design (see README), so instead we use a small, transparent,
hand-built feature table for a curated demo vocabulary: each term gets a
short vector across a few conceptual axes (animal-ness, youth, finance,
nature, technology...), and we PCA-reduce that to 2D with pure numpy.

This is clearly a teaching illustration, not a production embedding
model — that's the honest v2 upgrade path (swap this module for real
embeddings from an API once you're OK adding that external call; see
README roadmap).
"""
from __future__ import annotations
import numpy as np

# axes: [animal, youth, finance, nature, technology, danger]
_FEATURE_TABLE: dict[str, list[float]] = {
    "dog":                  [0.95, 0.10, 0.00, 0.20, 0.00, 0.05],
    "puppy":                [0.95, 0.90, 0.00, 0.15, 0.00, 0.02],
    "cat":                  [0.90, 0.10, 0.00, 0.20, 0.00, 0.05],
    "kitten":               [0.90, 0.90, 0.00, 0.15, 0.00, 0.02],
    "stock market":         [0.00, 0.05, 0.95, 0.00, 0.10, 0.30],
    "investment portfolio": [0.00, 0.05, 0.95, 0.00, 0.05, 0.25],
    "finance and banking":  [0.00, 0.05, 0.90, 0.00, 0.15, 0.20],
    "ocean wave":           [0.05, 0.10, 0.00, 0.95, 0.00, 0.35],
    "beach sand":           [0.02, 0.10, 0.00, 0.90, 0.00, 0.05],
    "mountain trail":       [0.05, 0.10, 0.00, 0.85, 0.00, 0.20],
    "laptop computer":      [0.00, 0.00, 0.05, 0.00, 0.95, 0.05],
    "smartphone app":       [0.00, 0.05, 0.10, 0.00, 0.90, 0.05],
    "wolf":                 [0.85, 0.15, 0.00, 0.40, 0.00, 0.65],
    "shark":                [0.60, 0.10, 0.00, 0.60, 0.00, 0.80],
}

DEFAULT_TERMS = [
    "dog", "puppy", "cat", "kitten",
    "stock market", "investment portfolio", "finance and banking",
    "ocean wave", "beach sand",
]


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Pure-numpy PCA via SVD, projected to the top 2 components."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    return centered @ components.T


def available_terms() -> list[str]:
    return sorted(_FEATURE_TABLE.keys())


def compute_embedding_map(terms: list[str] | None = None) -> dict:
    terms = terms or DEFAULT_TERMS
    unknown = [t for t in terms if t not in _FEATURE_TABLE]
    known = [t for t in terms if t in _FEATURE_TABLE]

    if len(known) < 2:
        raise ValueError(
            f"Need at least 2 recognized terms to plot. Unknown: {unknown}. "
            f"Available terms: {available_terms()}"
        )

    matrix = np.array([_FEATURE_TABLE[t] for t in known])
    coords = _pca_2d(matrix)

    return {
        "points": [
            {"label": known[i], "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
            for i in range(len(known))
        ],
        "unknown_terms": unknown,
    }
