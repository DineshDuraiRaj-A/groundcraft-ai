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
import re

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
    "dog", "puppy", "cat", "kitten", "wolf",
    "stock market", "investment portfolio", "finance and banking",
    "ocean wave", "beach sand", "mountain trail",
    "laptop computer", "smartphone app",
]

# Keyword -> axis weights, used to place words the table has never seen.
# Deliberately small and readable: this is a teaching illustration, and a
# learner should be able to look at this list and understand exactly why
# their word landed where it did.
_AXIS_HINTS: dict[str, list[float]] = {
    # animal
    "animal": [.9,0,0,.3,0,.1], "pet": [.9,.2,0,.1,0,0], "bird": [.85,0,0,.4,0,0],
    "fish": [.8,0,0,.5,0,.1], "horse": [.9,0,0,.3,0,0], "lion": [.85,0,0,.4,0,.7],
    "tiger": [.85,0,0,.4,0,.7], "bear": [.85,0,0,.45,0,.6], "rabbit": [.9,.3,0,.3,0,0],
    # youth
    "baby": [.3,.95,0,0,0,0], "child": [.2,.9,0,0,0,0], "young": [.1,.85,0,0,0,0],
    "cub": [.85,.9,0,.2,0,.1], "kid": [.2,.9,0,0,0,0],
    # finance
    "money": [0,0,.95,0,0,.2], "bank": [0,0,.9,0,.1,.1], "invest": [0,0,.95,0,.05,.3],
    "salary": [0,0,.85,0,0,0], "tax": [0,0,.85,0,0,.2], "budget": [0,0,.8,0,0,.1],
    "profit": [0,0,.9,0,0,.2], "loan": [0,0,.9,0,0,.4], "shares": [0,0,.95,0,0,.3],
    # nature
    "tree": [.1,0,0,.9,0,0], "forest": [.2,0,0,.95,0,.2], "river": [.05,0,0,.9,0,.2],
    "mountain": [0,0,0,.9,0,.3], "sea": [.05,0,0,.95,0,.35], "sky": [0,0,0,.85,0,0],
    "rain": [0,0,0,.85,0,.2], "flower": [.05,.2,0,.9,0,0], "beach": [0,0,0,.9,0,.1],
    # technology
    "computer": [0,0,.05,0,.95,0], "software": [0,0,.1,0,.95,0], "code": [0,0,.05,0,.95,0],
    "internet": [0,0,.1,0,.9,.1], "robot": [.2,0,0,0,.9,.2], "phone": [0,0,.1,0,.9,0],
    "data": [0,0,.2,0,.85,0], "server": [0,0,.1,0,.9,0], "app": [0,0,.1,0,.9,0],
    # danger
    "war": [0,0,.1,0,0,.95], "fire": [0,0,0,.4,0,.9], "storm": [0,0,0,.7,0,.8],
    "poison": [.1,0,0,.3,0,.95], "weapon": [0,0,.1,0,.2,.95], "crash": [0,0,.3,0,.2,.85],
    # extra everyday words, so ordinary guesses land somewhere sensible
    "elephant": [.9,0,0,.4,0,.2], "cow": [.9,0,0,.3,0,0], "sheep": [.9,0,0,.3,0,0],
    "mouse": [.85,0,0,.3,0,0], "snake": [.85,0,0,.4,0,.7], "shark": [.6,0,0,.6,0,.8],
    "savings": [0,0,.9,0,0,0], "account": [0,0,.8,0,.1,0], "cash": [0,0,.95,0,0,.1],
    "price": [0,0,.85,0,0,.1], "cost": [0,0,.85,0,0,.1], "market": [0,0,.9,0,.05,.2],
    "garden": [.1,.1,0,.85,0,0], "island": [.05,0,0,.9,0,.2], "desert": [0,0,0,.85,0,.4],
    "laptop": [0,0,.05,0,.95,0], "ai": [0,0,.1,0,.95,.15], "algorithm": [0,0,.05,0,.95,0],
    "network": [0,0,.1,0,.9,.05], "cloud": [0,0,.1,.3,.8,0],
    "danger": [0,0,0,.1,0,.95], "risk": [0,0,.4,0,.1,.8], "threat": [0,0,.1,0,.1,.9],
}

_STOP = {"my","a","an","the","of","and","in","on","for","to","is","some","this","that"}


def _infer_vector(term: str) -> list[float] | None:
    """
    Place an unknown word by matching it against the keyword hints above.
    Returns None when nothing matches, so the caller can tell the learner
    honestly that the word isn't in the demo vocabulary rather than
    silently dropping it at the origin.
    """
    t = term.lower().strip()
    if not t:
        return None
    tokens = [w for w in re.findall(r"[a-z]+", t) if w not in _STOP]
    matches = []
    for w in tokens:
        if w in _AXIS_HINTS:
            matches.append(_AXIS_HINTS[w]); continue
        # crude singular/plural + substring fallback
        for k, v in _AXIS_HINTS.items():
            if w.rstrip("s") == k.rstrip("s") or (len(w) > 4 and (w in k or k in w)):
                matches.append(v); break
    if not matches:
        return None
    n = len(matches)
    return [sum(m[i] for m in matches) / n for i in range(6)]


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Pure-numpy PCA via SVD, projected to the top 2 components."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    return centered @ components.T


def available_terms() -> list[str]:
    return sorted(_FEATURE_TABLE.keys())


def compute_embedding_map(terms: list[str] | None = None, include_reference: bool = False) -> dict:
    """
    include_reference: when a learner supplies their own words, we also plot
    the built-in vocabulary (greyed out in the UI) so their words have
    recognisable landmarks to sit against. Plotting 3 words alone tells you
    nothing; plotting 3 words among 13 known ones is the whole lesson.
    """
    terms = terms or DEFAULT_TERMS
    user_terms = list(terms)

    vectors: dict[str, list[float]] = {}
    unknown: list[str] = []
    for t in terms:
        key = t.lower().strip()
        if key in _FEATURE_TABLE:
            vectors[t] = _FEATURE_TABLE[key]
        else:
            inferred = _infer_vector(t)
            if inferred:
                vectors[t] = inferred
            else:
                unknown.append(t)

    if include_reference:
        for ref in DEFAULT_TERMS:
            if ref not in vectors:
                vectors[ref] = _FEATURE_TABLE[ref]

    known = list(vectors.keys())

    if len(known) < 2:
        raise ValueError(
            "I could not place any of those words. This demo understands a small "
            "vocabulary — try words about animals, money, nature, technology or danger. "
            f"Unrecognised: {', '.join(unknown) if unknown else 'none'}"
        )

    matrix = np.array([vectors[t] for t in known])
    coords = _pca_2d(matrix)

    return {
        "points": [
            {
                "label": known[i],
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "is_user": known[i] in user_terms,
            }
            for i in range(len(known))
        ],
        "unknown_terms": unknown,
    }
