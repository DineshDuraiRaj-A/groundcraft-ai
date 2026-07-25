# Ground Craft AI

**Learn how AI really thinks.** An interactive, no-code lab that teaches LLM grounding — chunking, retrieval, temperature, and hallucination — by letting you tune real parameters against a real model and watch the answer change live.

Live parameters, real answers, zero jargon. Built to make AI grounding visible to everyone.

---

## What's in this repo

```
tracestack/
├── backend/           FastAPI app — chunking, retrieval, LLM calls, rate limiting
│   ├── app/
│   │   ├── main.py         API routes
│   │   ├── chunking.py     word-based chunking with overlap
│   │   ├── retrieval.py    lightweight TF-IDF retrieval (no embedding API needed)
│   │   ├── providers.py    LLM provider abstraction (free tier + own key)
│   │   ├── documents.py    sample docs + upload handling (.txt/.md/.pdf)
│   │   ├── ratelimit.py    per-session rate limiter for the shared free-tier key
│   │   ├── auth.py         Google Sign-In verification (stores name + email only)
│   │   ├── tutor.py        Craft Guide assistant — grounded KB + its own retrieval
│   │   ├── context_window.py   context-overflow simulation (mission 4)
│   │   ├── embeddings_viz.py   2D meaning-map via numpy PCA (mission 5)
│   │   ├── storage.py      feedback/visitor persistence (Supabase, in-memory fallback)
│   │   └── samples/        3 sample documents used by the Lab
│   ├── requirements.txt
│   ├── Procfile        for Render/Railway
│   └── render.yaml     one-click Render deploy config
├── frontend/
│   └── index.html     single-file frontend — vanilla HTML/CSS/JS, no build step
└── README.md
```

## Why it's built this way

- **No embeddings API for retrieval.** Retrieval uses a small in-process TF-IDF index instead of a real embedding model. This keeps cold starts fast on free hosting tiers and means the *only* external, rate-limited call in the whole app is the final LLM generation — exactly the step the token/cost lesson wants to highlight. Swapping in real embeddings later is a drop-in change to `retrieval.py` (see Roadmap).
- **No build step on the frontend.** Single HTML file, vanilla JS, CSS custom properties for theming. Loads fast, caches well, and there's nothing to compile before deploying.
- **Two LLM paths, one interface.** Visitors either use a shared free-tier key (OpenRouter `:free` models, rate-limited per session) or paste their own key. Both go through the same `providers.py` interface.

---

## Running it locally

**Backend**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENROUTER_API_KEY
uvicorn app.main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

The frontend defaults to `http://localhost:8000` for the API. To point it elsewhere (e.g. your deployed backend), set `window.TRACESTACK_API_BASE` before the script runs, or add a `<script>window.TRACESTACK_API_BASE = "https://your-api.onrender.com";</script>` tag before the closing `</body>`.

> **Note on this session's testing:** the sandbox this was built in can't reach `openrouter.ai` (network is restricted to package registries), so the free-tier LLM call itself is untested against a live model here. Everything else — chunking, TF-IDF retrieval, upload validation, rate limiting, the feedback/stats API, and the full request/response contract — was tested end-to-end with FastAPI's TestClient and a live local server. Once you add a real `OPENROUTER_API_KEY` and deploy (or run locally with internet access), the generation step should work as-is; if the response shape from a specific provider differs, check `providers.py`'s `_call_openai_compatible`.

---

## Getting a free OpenRouter key (for the "Try it free" path)

1. Sign up at openrouter.ai — no credit card required
2. Generate a key in account settings
3. Put it in `backend/.env` as `OPENROUTER_API_KEY`
4. Free-tier models (IDs ending in `:free`) are rate-limited by OpenRouter itself (roughly 20 req/min, 50/day — check current limits, they change) **on top of** this app's own 10-requests-per-session-per-hour limit, which exists specifically to stop one visitor from draining your shared key

## Bringing your own key (for visitors)

Click **⚙️ Settings** in the app, choose "Use my own key," and paste an OpenRouter/OpenAI-compatible key. It's sent only with that visitor's own requests and is never persisted server-side.

---

## Deployment

- **Frontend** → GitHub Pages or Vercel (static hosting, zero config)
- **Backend** → Render or Railway free tier
  - Set `OPENROUTER_API_KEY` and `PUBLIC_APP_URL` as environment variables
  - Free tiers sleep after inactivity — first request after idle will be slow. Mention this near the "Try it free" button so it doesn't read as broken.
  - Tighten `allow_origins` in `main.py`'s CORS middleware to your actual frontend URL before going live (currently `"*"` for local development)

---

## Feedback & visitor persistence (Supabase)

`app/storage.py` uses Supabase when `SUPABASE_URL` and `SUPABASE_KEY` are set in your environment, and **automatically falls back to in-memory storage** when they're not — so the app runs immediately without requiring a Supabase account first.

To enable it:
1. Create a free project at supabase.com
2. In the SQL editor, create the `feedback` and `visits` tables — the exact schema is documented at the top of `app/storage.py`
3. Enable Row Level Security on both tables and add a policy allowing `anon` to `INSERT` and `SELECT` — don't use the service role key here, and don't skip RLS because it's "just a demo"
4. Add `SUPABASE_URL` and `SUPABASE_KEY` (the anon/public key) to your `.env` or hosting provider's environment variables

`GET /api/stats` includes a `"storage"` field (`"supabase"` or `"in-memory"`) so you can confirm which one is active without checking logs.

## Extended features (backend built, frontend wiring pending)

These endpoints exist and are exercised in this session's testing, but the frontend UI for them isn't wired yet — that's tomorrow's work. Two of them need zero LLM call and are fully tested end-to-end; the other four call a model and were verified by mocking the provider (confirms the request/response contract and error handling), not against a live model — sanity-check those once you have a real `OPENROUTER_API_KEY` and network access.

| # | Feature | Endpoint | LLM call? | Test status |
|---|---|---|---|---|
| 3 | Context window limits | `POST /api/context-preview` | No | ✅ Fully tested — pure token arithmetic |
| 4 | Embedding similarity map | `POST /api/embedding-map` | No | ✅ Fully tested — hand-built feature table + numpy PCA |
| 5 | Prompt injection sandbox | `POST /api/injection-test` | Yes | ⚠️ Contract tested with a mocked provider |
| 6 | Model comparison | `POST /api/compare` | Yes (parallel, 2–3 models) | ⚠️ Contract tested with a mocked provider |
| 7 | System prompt strictness | `strictness` field on `POST /api/query` | Yes | ⚠️ Prompt-building logic tested; live effect unverified |
| 8 | Streaming toggle | `POST /api/query/stream` (SSE) | Yes | ⚠️ SSE parsing tested with a mocked stream |
| 9 | Confidence signal | `confidence_level` field on `POST /api/query` response | No | ✅ Fully tested — thresholded off retrieval score |
| 10 | Bias/perspective probe | `POST /api/bias-probe` | Yes (parallel, 2 sources) | ⚠️ Contract tested with a mocked provider |

Notes worth knowing before wiring the frontend:

- **Embedding map (#4)** doesn't call a real embedding API — see the docstring in `embeddings_viz.py` for why (TF-IDF can't capture "dog" ≈ "puppy" since they share no words) and what the honest v2 upgrade path is.
- **Context preview (#3)** ranks *every* chunk by retrieval score, then fills a token budget greedily — the frontend should call this whenever the "max context" slider moves, not just on submit, since it's cheap (no LLM call).
- **Bias probe (#10)** ships with one preset pair (`remote_work`) — two short sample docs on the same topic, framed differently. Add more pairs in `documents.PERSPECTIVE_PAIRS` following the same shape.
- **Streaming (#8)** returns Server-Sent Events (`text/event-stream`); the frontend will need an `EventSource`-style reader rather than a normal `fetch().json()` call.
- All four LLM-calling endpoints route through the same session rate limiter as `/api/query` when `provider: "free"` — verified they correctly return `429` after the per-session cap.

## The Craft Guide (in-app assistant)

A help assistant grounded in a 24-entry knowledge base covering every mission, concept and common sticking point. `POST /api/assistant`.

Two things worth knowing before you change it:

- **It has its own retriever**, not the shared `TfidfIndex`. That index has a `+1` IDF floor, which is fine for paragraph-length document matching but wrong for four-word help queries — ubiquitous words like "a"/"is"/"what" keep a weight of 1 and dominate. Tested directly: *"what is a hallucination?"* retrieved the tokens-and-cost entry. `tutor.py` uses stopword filtering, IDF that decays to zero, and a 3x title boost. Routing is now 9/10 on the test set, and off-topic queries score exactly 0.
- **It degrades to retrieval-only.** With no LLM key, or on provider error, or once the hourly budget is spent, it returns the best-matching KB entry directly with `"mode": "offline"` instead of failing. The help button therefore works on a fresh deploy with zero configuration — and it doubles as a live demonstration of the app's own lesson.

The assistant has a separate rate-limit bucket (20/hour) from missions (10/hour), so being stuck doesn't cost you mission budget.

## Known limitations / v1 scope cuts

- Rate limiting is in-memory and per-process — fine for one instance, not for multiple. Swap for Redis if you scale past one backend process.
- PDF extraction won't work on scanned/image-only PDFs (no OCR) — the upload endpoint returns a clear error in that case rather than silently returning empty text.
- The Hallucination test and Token & cost tabs populate from the same query you run on the Grounding tab — there's no separate "run" button per tab, by design, so all three views stay in sync with one question.

## v2 roadmap (deliberately deferred)

- Embedding similarity map (2D visualization of semantic closeness)
- Model comparison (same question, multiple models side by side)
- Prompt injection sandbox
- Real embeddings option alongside TF-IDF for better retrieval quality
