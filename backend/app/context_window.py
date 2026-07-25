"""
Feature 3: Context window limits.

Simulates what happens when you keep stuffing retrieved chunks into a
fixed-size context window. Pure arithmetic on token estimates — no LLM
call needed, so this is fully testable and demoable without a live API key.

The "wall" the slider is meant to visualize: chunks are added in ranked
order (highest retrieval score first) until the running total would
exceed max_context_tokens. Everything after that point is either
truncated (partially fits) or dropped entirely.
"""
from __future__ import annotations
from dataclasses import dataclass

from .providers import estimate_tokens
from .retrieval import ScoredChunk

# Rough overhead for system prompt + question + formatting boilerplate.
# Real number depends on the exact prompt template; this is a reasonable
# estimate for teaching purposes, not a billing-accurate figure.
PROMPT_OVERHEAD_TOKENS = 60


@dataclass
class ContextSlot:
    chunk_id: str
    score: float
    tokens: int
    status: str  # "included" | "truncated" | "dropped_overflow"
    cumulative_tokens: int


def simulate_context_stuffing(
    scored_chunks: list[ScoredChunk],
    question: str,
    max_context_tokens: int,
) -> dict:
    """
    Returns a dict describing which chunks fit, which got truncated at the
    boundary, and which were dropped entirely — plus the running token
    total, so the frontend can render the "wall" visually.
    """
    budget = max_context_tokens - PROMPT_OVERHEAD_TOKENS - estimate_tokens(question)
    slots: list[ContextSlot] = []
    running = 0
    wall_hit = False

    for sc in scored_chunks:
        chunk_tokens = estimate_tokens(sc.chunk.text)

        if wall_hit:
            slots.append(ContextSlot(sc.chunk.id, sc.score, chunk_tokens, "dropped_overflow", running))
            continue

        if running + chunk_tokens <= budget:
            running += chunk_tokens
            slots.append(ContextSlot(sc.chunk.id, sc.score, chunk_tokens, "included", running))
        else:
            # this chunk is the one that hits the wall — show how much of
            # it would actually fit before truncation
            remaining_room = max(0, budget - running)
            running = budget
            slots.append(ContextSlot(sc.chunk.id, sc.score, chunk_tokens, "truncated", running))
            wall_hit = True

    return {
        "max_context_tokens": max_context_tokens,
        "prompt_overhead_tokens": PROMPT_OVERHEAD_TOKENS + estimate_tokens(question),
        "budget_for_chunks": max(0, budget),
        "used_tokens": running,
        "wall_hit": wall_hit,
        "slots": [
            {
                "chunk_id": s.chunk_id,
                "score": round(s.score, 4),
                "tokens": s.tokens,
                "status": s.status,
                "cumulative_tokens": s.cumulative_tokens,
            }
            for s in slots
        ],
    }
