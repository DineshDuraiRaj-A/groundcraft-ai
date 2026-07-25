"""
Splits raw text into overlapping chunks.

We chunk on whitespace-split "tokens" (words) rather than true LLM tokens to
avoid pulling in a tokenizer dependency — close enough for a teaching demo,
and it keeps the backend dependency-light and fast to cold-start on free
hosting tiers.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    start_word: int
    end_word: int


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    chunk_size and overlap are measured in words.
    Returns a list of Chunk objects covering the whole document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                id=f"c{idx}",
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
            )
        )
        idx += 1
        if end == len(words):
            break
        start += step

    return chunks
