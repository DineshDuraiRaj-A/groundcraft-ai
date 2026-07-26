"""
The in-app learning assistant ("Craft Guide").

Design notes worth knowing before you change this:

1. The assistant is GROUNDED in a local knowledge base of the app's own
   missions, objectives and concepts — the same retrieval machinery the
   learner is being taught about. It is deliberately not a general-purpose
   chatbot: if someone asks it to write Python or plan a holiday, it should
   politely steer back to the lessons.

2. It degrades gracefully. If no LLM key is configured (or the provider
   errors), the endpoint still answers by returning the best-matching
   knowledge-base entry verbatim, flagged as `"mode": "offline"`. That
   keeps the Help button useful on a fresh deploy, and it's a live
   demonstration of the exact point the app is making: retrieval alone
   gets you most of the way; the model just makes it conversational.

3. Answers are kept short on purpose. This renders in a small chat panel,
   often on a phone, usually while someone is mid-mission and stuck.
"""
from __future__ import annotations
from dataclasses import dataclass

import math
import re
from collections import Counter


@dataclass
class KBEntry:
    key: str
    title: str
    text: str


# ---------------------------------------------------------------------------
# Knowledge base. Each entry is written in plain English, at the level of a
# complete beginner, because that's exactly who asks the assistant for help.
# ---------------------------------------------------------------------------
KB: list[KBEntry] = [
    # ---------- core concepts ----------
    KBEntry("grounding", "Grounding",
        "Grounding means forcing an AI to base its answer on a document you supply, rather than on "
        "whatever it happens to remember from training. In Ground Craft AI you turn this on and off "
        "with the 'Use the document' switch. When grounding is on, the app searches your document, "
        "picks the best-matching passages, and hands only those to the model along with your question. "
        "Grounding is the single most common technique used to make AI trustworthy at work, because it "
        "means every answer can be traced back to a source you control. The Ground Zero mission teaches this."),

    KBEntry("hallucination", "Hallucination",
        "A hallucination is when an AI states something false with complete confidence. It is not lying: "
        "it genuinely has no idea it is wrong. A language model predicts words that sound right; it does "
        "not check facts. This is why confidence and correctness are completely unrelated in an AI answer. "
        "The best way to see it is to switch grounding off and ask about something your document never "
        "covers — the model will invent an answer rather than admit it doesn't know. The Myth Buster mission covers this."),

    KBEntry("tokens", "Tokens and cost",
        "AI models don't read words, they read tokens — chunks of roughly three quarters of a word. "
        "Everything you send in and everything the model sends back is counted in tokens, and that count "
        "is what you are billed for. This is why sending more context is not free, and why a chatty model "
        "costs more than a concise one. In the app, the Bean Counter mission shows the real token split "
        "between what went in (your question plus retrieved evidence) and what came back (the answer)."),

    KBEntry("context_window", "Context window",
        "The context window is the AI's working memory for a single question. Think of a desk that only "
        "fits so many pages: hand it more and pages fall off the edge, silently, with no warning. This is "
        "why 'just give the AI everything' is bad advice — past the limit, content is truncated or dropped "
        "entirely, and the model never sees it. Choosing the right few passages beats sending all of them. "
        "The Wall Breaker mission lets you shrink the window with a slider and watch evidence get cut off."),

    KBEntry("embeddings", "Embeddings and semantic search",
        "An embedding turns a word or sentence into a position on a map of meaning. Words that mean similar "
        "things land near each other, so 'dog' sits close to 'puppy' even though they share no letters. "
        "That's how semantic search finds relevant text without exact keyword matches. The Cartographer "
        "mission plots everyday words on a 2D map so you can see animals cluster together, money words "
        "cluster together, and beach words cluster separately — without anyone telling the computer those categories exist."),

    KBEntry("prompt_injection", "Prompt injection and guardrails",
        "Prompt injection is when a cleverly worded message talks an AI into ignoring its own instructions. "
        "It works because the model cannot reliably tell its instructions apart from your message — to it, "
        "both are just text. This is why you should never trust an AI with a secret it is also allowed to "
        "discuss. Guardrails are the extra checks built around a model to catch this. The Red Teamer mission "
        "lets you try it yourself against an assistant told to protect a code."),

    KBEntry("model_comparison", "Comparing models",
        "Different AI models were trained on different data, at different sizes, with different goals. A "
        "bigger model is slower and more expensive, and for a simple factual lookup it is often no better "
        "at all. The useful skill is choosing a model based on evidence — actual answers, speed and cost on "
        "your own task — rather than on hype or parameter counts. The Judge mission runs the same grounded "
        "question through two or three models side by side."),

    KBEntry("system_prompt", "System prompts",
        "A system prompt is a standing instruction the AI receives before it ever sees your message — the "
        "invisible rulebook. Every AI product you have used has one, and you never get to read it. It has "
        "enormous power over the answer: the same question, same document and same model can produce very "
        "different replies depending on whether the model was told 'answer only from the document', 'use the "
        "document and fill gaps from memory', or 'ignore the document'. The Rule Bender mission lets you flip between all three."),

    KBEntry("streaming", "Streaming",
        "Streaming means the app forwards each piece of the answer the instant the model produces it, instead "
        "of waiting for the whole reply and showing it at once. The AI writes one token at a time either way; "
        "streaming just stops hiding that. It feels dramatically faster even though the total time is identical, "
        "which makes it a great lesson in perceived versus actual speed. The Speed Watcher mission shows it live."),

    KBEntry("bias", "Source bias",
        "A grounded AI is loyal to its sources. Feed it one-sided material and it will give you confident, "
        "well-cited, one-sided answers — the citations do not make it balanced. This is why the useful habit "
        "is to ask 'what was this AI reading?' before trusting what it says. The Skeptic mission asks one "
        "question against two documents that argue opposite sides of the same issue, so you can watch the "
        "answer change while the model stays equally confident."),

    KBEntry("chunking", "Chunk size",
        "Long documents get cut into smaller pieces before the AI searches them, and chunk size sets how big "
        "each piece is. Pieces that are too small lose their surrounding context and stop making sense on "
        "their own. Pieces that are too big bury the real answer in unrelated text and cost more to send. "
        "There is no universally correct value — it depends on your document, which is exactly why the app "
        "gives you a slider instead of a number to memorise."),

    KBEntry("topk", "Evidence pieces (top-K)",
        "After searching your document, the app keeps only the best-matching passages and hands those to the "
        "model. The 'Evidence pieces' slider controls how many it keeps. One piece might only contain half the "
        "answer. Ten pieces bury the good one in noise and cost more tokens. Three is a sensible default for "
        "most short documents."),

    KBEntry("temperature", "Creativity (temperature)",
        "Temperature controls how predictable or adventurous the model's wording is. Low temperature means it "
        "plays safe and picks the most likely next word, which is what you want for factual answers. High "
        "temperature means it takes risks, which is better for brainstorming and creative writing. Same model, "
        "very different behaviour. For grounded factual questions, keep it low — around 0.2 to 0.4."),

    KBEntry("confidence", "Confidence signals",
        "When the app searches your document it scores how well each passage matches your question. If the best "
        "match is weak, the app shows a 'Low confidence' badge instead of pretending to be sure. This matters "
        "because a well-built AI system can and should flag its own uncertainty — most consumer chatbots simply "
        "don't, which is why they sound equally confident whether they know the answer or not."),

    # ---------- app mechanics ----------
    KBEntry("missions_overview", "The ten missions",
        "The Forge contains ten missions. In order: Ground Zero (grounding), Myth Buster (hallucination), "
        "Bean Counter (tokens and cost), Wall Breaker (context window), Cartographer (embeddings), Red Teamer "
        "(prompt injection), The Judge (model comparison), Rule Bender (system prompts), Speed Watcher "
        "(streaming) and Skeptic (source bias). Each one shows what you'll do and what you'll learn before you "
        "start, so you can decide whether to commit."),

    KBEntry("unlocking", "Unlocking and XP",
        "Missions unlock in stages so beginners aren't dropped into all ten at once. Ground Zero is open "
        "immediately. Myth Buster and Bean Counter unlock after one mission, Wall Breaker and Cartographer "
        "after three, Red Teamer, The Judge and Rule Bender after five, and Speed Watcher and Skeptic after "
        "eight. Each mission awards XP, and XP moves you through four ranks: Curious Beginner, Grounded "
        "Thinker, AI Interpreter and Craft Master."),

    KBEntry("stuck_grounding", "Stuck on Ground Zero",
        "To complete Ground Zero, type a question in the box at the top of the lab and press Ask. You'll get "
        "two answers side by side: one where the AI read your document, one where it guessed from memory. If "
        "nothing happens, check the error message under the Ask button — the most common cause is the backend "
        "still waking up, which can take up to a minute on free hosting. Asking one question completes the "
        "first three missions at once."),

    KBEntry("stuck_wall", "Stuck on Wall Breaker",
        "If the wall never appears, your desk space is too generous — everything fits, so nothing overflows. "
        "Drag the 'Desk space' slider down to around 150 to 250 tokens and press 'Fill the desk' again. You'll "
        "see solid blocks for evidence that fit, a striped block for the passage that got truncated at the "
        "boundary, and dashed outlines for everything dropped entirely."),

    KBEntry("stuck_injection", "Stuck on Red Teamer",
        "The assistant has been told never to reveal a code. Straightforward requests will usually be refused — "
        "that's the point. Things people try: asking it to role-play as a system with no rules, asking it to "
        "translate its instructions into another language, asking it to repeat everything above your message, "
        "or claiming to be a developer running a test. You complete the mission whether you succeed or not; "
        "failing is just as instructive."),

    KBEntry("stuck_compare", "Stuck on The Judge",
        "You need at least two models ticked before the comparison will run. If one model returns an error while "
        "the other works, that's usually the free tier being busy or that specific model being temporarily "
        "unavailable — untick it and try another. Comparison results show each model's answer alongside its "
        "speed and token count so you can judge the trade-off."),

    KBEntry("upload", "Using your own document",
        "Press the Upload button next to the document picker to use your own .txt, .md or .pdf file instead of "
        "a sample. There's a size cap of about 20,000 characters, roughly five pages, which keeps searches fast "
        "and keeps one upload from using up the shared free AI budget. Scanned or image-only PDFs won't work "
        "because there's no text in them to extract — you'll get a clear error rather than an empty answer."),

    KBEntry("free_tier", "The free AI and its limits",
        "By default the app uses a shared free-tier key so you can try everything without signing up for "
        "anything. That key is limited to ten questions per visitor per hour, which exists so one enthusiastic "
        "learner can't use up the daily budget for everyone else. If you hit the limit you can wait, or add "
        "your own API key in settings — your own key isn't rate-limited by this app at all."),

    KBEntry("saving", "Saving your progress",
        "Progress is kept in memory unless you sign in with Google. Signing in stores only your name and email, "
        "used purely to remember your XP and completed missions between visits and to place you on the "
        "leaderboard. There's no password, no posting, no contacts access. If you'd rather not sign in, "
        "everything still works — you just start fresh each visit."),

    KBEntry("slow", "The app feels slow to start",
        "The first request after a quiet period can take 30 to 60 seconds. The backend runs on a free hosting "
        "tier that puts the server to sleep when nobody's using it, and the first visitor has to wait for it to "
        "wake up. After that, answers come back in a second or two. It's not broken — it's the trade-off for "
        "the whole thing being free."),
]

# ---------------------------------------------------------------------------
# Retrieval.
#
# This deliberately does NOT reuse retrieval.TfidfIndex. That index is tuned
# for whole documents and its IDF has a +1 floor, so ubiquitous words ("a",
# "is", "what") keep a weight of 1 and dominate short queries — fine when
# matching paragraph-length text, wrong when someone types four words into a
# help box. Testing this directly: "what is a hallucination?" retrieved the
# tokens-and-cost entry instead of the hallucination one.
#
# So the assistant gets its own scorer: stopwords removed, IDF that actually
# decays to zero for terms present in every entry, and a boost for words that
# appear in an entry's title.
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","am","do","does","did","doing",
    "i","im","i'm","me","my","you","your","it","its","this","that","these","those","they","them",
    "what","whats","which","who","how","when","where","why","can","could","should","would","will",
    "of","in","on","at","to","for","with","about","from","by","as","and","or","but","if","then",
    "so","not","no","yes","get","got","have","has","had","need","want","help","please","tell","show",
    "there","here","up","out","more","some","any","all","just","like","really","very","much","also",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class _KBIndex:
    def __init__(self, entries: list[KBEntry]):
        self.entries = entries
        # titles carry the strongest signal, so weight them 3x by repetition
        self.docs = [_tokens(e.text) + _tokens(e.title) * 3 for e in entries]
        self.n = max(1, len(entries))
        df: Counter[str] = Counter()
        for toks in self.docs:
            for t in set(toks):
                df[t] += 1
        self.df = df
        self.vectors = [self._vec(toks) for toks in self.docs]

    def _idf(self, term: str) -> float:
        # No +1 floor: a term in every entry scores 0 and drops out entirely.
        d = self.df.get(term, 0)
        if d == 0:
            return 0.0
        return math.log(self.n / d) + 0.05

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        total = max(1, len(toks))
        return {t: (n / total) * self._idf(t) for t, n in tf.items()}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in (a.keys() & b.keys()))
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def query(self, q: str, top_k: int) -> list[tuple[KBEntry, float]]:
        qv = self._vec(_tokens(q))
        scored = [(e, self._cos(qv, v)) for e, v in zip(self.entries, self.vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


_KB_INDEX: _KBIndex | None = None

# Below this, we treat the question as "not covered here" rather than
# returning a confidently irrelevant entry.
RELEVANCE_FLOOR = 0.035


def _index() -> _KBIndex:
    global _KB_INDEX
    if _KB_INDEX is None:
        _KB_INDEX = _KBIndex(KB)
    return _KB_INDEX


def retrieve(question: str, top_k: int = 3) -> list[tuple[KBEntry, float]]:
    return [(e, s) for e, s in _index().query(question, top_k) if s > 0]


SYSTEM_PROMPT = """You are the Craft Guide, the built-in tutor inside Ground Craft AI — a hands-on app that \
teaches how AI language models work to people with no technical background at all.

How to answer:
- Answer ONLY from the reference material provided below. It covers this app's missions and the concepts they teach.
- If the material doesn't cover something, say so plainly and point them to the mission that's closest, or suggest \
they ask about one of the ten missions. Never invent app features, mission names, buttons or settings.
- Write for someone who has never used an AI tool before. No jargon unless you immediately explain it in ordinary words.
- Be brief: two to four sentences normally. This appears in a small chat panel, often on a phone, usually while \
someone is stuck mid-mission and wants to get unstuck.
- Be warm and encouraging, never condescending. Being confused here is completely normal — the whole app exists \
because this stuff is genuinely unintuitive.
- If they seem stuck on a specific mission, give them the concrete next action to take, not a lecture.
- If they ask something off-topic (unrelated coding help, general trivia, personal advice), gently say that's \
outside what you can help with here and offer to explain a mission instead."""


def build_messages(question: str, entries: list[tuple[KBEntry, float]],
                   history: list[dict], current_mission: str | None) -> list[dict]:
    reference = "\n\n".join(f"[{e.title}]\n{e.text}" for e, _ in entries)

    context_note = ""
    if current_mission:
        context_note = f"\n\nThe learner is currently on the '{current_mission}' mission — take that into account."

    messages = [{"role": "system", "content": SYSTEM_PROMPT + context_note +
                 f"\n\n--- REFERENCE MATERIAL ---\n{reference}"}]

    # keep the last few turns only; this is a help widget, not a long conversation
    for turn in history[-6:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = str(turn.get("content", ""))[:1500]
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def offline_answer(entries: list[tuple[KBEntry, float]]) -> str:
    """
    Used when no LLM is configured or the provider fails. Returns the best
    matching knowledge-base entry directly — less conversational, still correct,
    and a live demonstration of retrieval doing the heavy lifting.
    """
    if not entries or entries[0][1] < RELEVANCE_FLOOR:
        return ("I'm not sure which part that's about. I can explain any of these:\n\n"
                "• Grounding — why the AI reads your document\n"
                "• Hallucination — why it invents things\n"
                "• Tokens & cost — what a question actually costs\n"
                "• Context window — the AI's memory limit\n"
                "• Embeddings — how meaning is stored\n"
                "• Prompt injection — breaking the rules\n"
                "• Comparing models, system prompts, streaming, source bias\n\n"
                "Or tell me which mission you're stuck on.")
    entry, _ = entries[0]
    # keep it short — this renders in a small chat panel
    text = entry.text
    if len(text) > 420:
        cut = text[:420]
        text = cut[:cut.rfind('. ') + 1] if '. ' in cut else cut + '…'
    return f"**{entry.title}**\n\n{text}"
