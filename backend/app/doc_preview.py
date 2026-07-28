"""
Document preview: what's in here, and what could I ask about it?

Learners were selecting a document blind. They'd upload their own PDF, then
ask about something it doesn't cover, get "I don't know", and conclude the app
was broken rather than that they'd asked the wrong question.

This produces a short summary and a handful of concrete, answerable questions
so they know what they're working with before they start.

Deliberately NO LLM call:
  - it runs instantly, in the document picker, where a spinner would be awful
  - it costs nothing and can't be rate-limited away
  - the app's whole premise is that a lot of "AI magic" is ordinary text
    processing, so doing this extractively is on-message

The questions are generated from what the document actually contains — its
headings, numbers, dates, durations, money and proper nouns — so anything
suggested is answerable from the text.
"""
from __future__ import annotations

import re
from collections import Counter

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z0-9']+")

_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
    "with", "by", "as", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "you", "your", "he", "she", "his", "her", "from", "not", "no", "will", "would",
    "can", "could", "should", "may", "might", "must", "have", "has", "had", "do",
    "does", "did", "any", "all", "each", "which", "who", "when", "where", "what",
    "how", "why", "than", "then", "there", "here", "also", "such", "into", "within",
    "per", "via", "shall", "must", "upon", "other", "more", "most", "some", "only",
}

# Signals that a sentence is worth asking about, with the question it suggests.
_TOPIC_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d+)\s*(business\s+)?days?\b", re.I), "__DURATION__"),
    (re.compile(r"\b(\d+)\s*(hours?|minutes?)\b", re.I), "__DURATION__"),
    (re.compile(r"\b(check[- ]?in|check[- ]?out)\b", re.I), "What are the check-in and check-out times?"),
    (re.compile(r"\b(refund|reimburse)", re.I), "What is the refund policy?"),
    (re.compile(r"\b(cancel|cancellation)", re.I), "What are the cancellation rules?"),
    (re.compile(r"\b(fee|charge|cost|price|deposit)s?\b", re.I), "What fees or charges apply?"),
    (re.compile(r"\b(eligib|qualify|entitled)", re.I), "Who is eligible?"),
    (re.compile(r"\b(warrant|guarantee)", re.I), "What does the warranty cover?"),
    (re.compile(r"\b(penalt|fine|violation|breach)", re.I), "What happens if the rules are broken?"),
    (re.compile(r"\b(contact|email|phone|call)\b", re.I), "Who do I contact?"),
    (re.compile(r"\b(pet|smoking|noise|guest|visitor)s?\b", re.I), "What are the rules about {match}?"),
    (re.compile(r"\b(parking|wifi|internet|pool|gym)\b", re.I), "What does it say about {match}?"),
    (re.compile(r"\b(deadline|due|expire|valid until)", re.I), "What deadlines apply?"),
    (re.compile(r"\b(security|privacy|data)\b", re.I), "What does it say about {match}?"),
    (re.compile(r"\b(reset|restart|troubleshoot|error)", re.I), "How do I fix problems?"),
]


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT.split(clean) if len(s.split()) >= 4]


def _keywords(text: str, limit: int = 8) -> list[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    counts = Counter(w for w in words if w not in _STOP and len(w) > 3 and not w.isdigit())
    return [w for w, _ in counts.most_common(limit)]


def _headings(text: str) -> list[str]:
    """Lines that look like section headings — numbered, short, or title-case."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) > 70:
            continue
        words = line.split()
        if len(words) > 9:
            continue
        numbered = re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line)
        titleish = line.isupper() or (
            sum(1 for w in words if w[:1].isupper()) >= max(2, len(words) - 1)
        )
        if (numbered or titleish) and not line.endswith((".", ",", ";")):
            out.append(re.sub(r"^\d+(\.\d+)*[.)]?\s*", "", line))
    seen, uniq = set(), []
    for h in out:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    return uniq[:8]


def _summarise(text: str, max_sentences: int = 3) -> str:
    """
    Extractive summary: score sentences by how many of the document's frequent
    keywords they contain, keep the best few, restore original order.
    """
    sents = _sentences(text)
    if not sents:
        return "This document is too short to summarise."
    if len(sents) <= max_sentences:
        return " ".join(sents)

    keys = set(_keywords(text, 12))
    scored = []
    for i, s in enumerate(sents):
        words = {w.lower() for w in _WORD.findall(s)}
        score = len(words & keys)
        if i < 3:
            score += 1.5          # openings usually state the subject
        score = score / (1 + abs(len(s.split()) - 22) / 30)   # prefer mid-length
        scored.append((score, i, s))

    best = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(s for _, _, s in sorted(best, key=lambda x: x[1]))


def _suggest_questions(text: str, headings: list[str], limit: int = 5) -> list[str]:
    """Only suggest questions the text can actually answer."""
    questions: list[str] = []
    seen: set[str] = set()

    def add(q: str):
        k = q.lower()
        if k not in seen and len(questions) < limit:
            seen.add(k)
            questions.append(q)

    for pattern, template in _TOPIC_HINTS:
        m = pattern.search(text)
        if not m:
            continue

        if template == "__DURATION__":
            # Ask about the sentence the duration lives in, not the document
            # title — "How long does acme saas - refund policy take?" was the
            # kind of nonsense the title-as-subject fallback produced.
            sent = next((s for s in _sentences(text) if pattern.search(s)), "")
            verb = next((v for v in ("refund", "delivery", "shipping", "processing",
                                     "approval", "response", "repair", "check-in",
                                     "cancellation", "setup", "activation")
                         if v in sent.lower()), "")
            article = "" if verb.endswith("ing") or verb == "check-in" else "a "
            add(f"How long does {article}{verb} take?" if verb else "What timeframes are mentioned?")
            continue

        # use the FULL match, so plurals survive ("pets", not "pet")
        match_word = m.group(0).strip()
        if match_word.isdigit() or len(match_word) < 3:
            match_word = ""
            for g in (m.groups() or ()):
                if g and not g.strip().isdigit():
                    match_word = g.strip()
                    break
        if "{match}" in template and not match_word:
            continue
        add(template.format(match=match_word.lower()))

    # headings make reliable questions — they name what the document covers
    for h in headings[1:]:
        add(f"What does the section on {h.lower()} say?")

    used_stems: set[str] = set()
    for kw in _keywords(text, 10):
        stem = kw.rstrip("s")                     # reef / reefs are one idea
        if stem in used_stems:
            continue
        used_stems.add(stem)
        add(f"What does the document say about {kw}?")

    return questions[:limit]


def preview(text: str, title: str = "") -> dict:
    words = len(text.split())
    headings = _headings(text)
    return {
        "title": title,
        "summary": _summarise(text),
        "suggested_questions": _suggest_questions(text, headings),
        "sections": headings[:6],
        "topics": _keywords(text, 6),
        "word_count": words,
        "char_count": len(text),
        "reading_time": ("under a minute" if words < 210
                         else f"about {max(1, round(words / 210))} min read"),
        # so the picker can say "this becomes ~N pieces at your current setting"
        "approx_chunks_at_500": max(1, -(-words // 500)),
    }
