"""
Feedback + visitor-count storage.

Uses Supabase when SUPABASE_URL and SUPABASE_KEY are set. Falls back to an
in-memory store automatically when they're not — so the app still runs
locally or in a quick demo without requiring a Supabase account first.

Expected Supabase schema (create these via the SQL editor):

    create table feedback (
        id bigint generated always as identity primary key,
        name text not null default 'Anonymous',
        emoji text not null default '🙂',
        text text not null,
        created_at timestamptz not null default now()
    );

    create table visits (
        id bigint generated always as identity primary key,
        created_at timestamptz not null default now()
    );

Row Level Security: for a public demo, enable RLS and add a policy allowing
anon INSERT and SELECT on both tables — don't use the service role key on
the frontend, and don't skip RLS just because it's "just a demo."
"""
from __future__ import annotations
import os
import time
from typing import Optional

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

# ---- in-memory fallback ----
_memory_feedback: list[dict] = []
_memory_visit_count = 4812


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def add_feedback(name: Optional[str], emoji: str, text: str) -> dict:
    entry = {"name": name or "Anonymous", "emoji": emoji, "text": text}

    if not SUPABASE_ENABLED:
        entry["ts"] = time.time()
        _memory_feedback.append(entry)
        return entry

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{SUPABASE_URL}/rest/v1/feedback", headers=_headers(), json=entry)
    resp.raise_for_status()
    row = resp.json()[0]
    return {"name": row["name"], "emoji": row["emoji"], "text": row["text"], "ts": row["created_at"]}


async def list_feedback(limit: int = 50) -> list[dict]:
    if not SUPABASE_ENABLED:
        return list(reversed(_memory_feedback[-limit:]))

    params = {"select": "name,emoji,text,created_at", "order": "created_at.desc", "limit": str(limit)}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{SUPABASE_URL}/rest/v1/feedback", headers=_headers(), params=params)
    resp.raise_for_status()
    return [{"name": r["name"], "emoji": r["emoji"], "text": r["text"], "ts": r["created_at"]} for r in resp.json()]


async def record_visit() -> int:
    global _memory_visit_count

    if not SUPABASE_ENABLED:
        _memory_visit_count += 1
        return _memory_visit_count

    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(f"{SUPABASE_URL}/rest/v1/visits", headers=_headers(), json={})
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/visits",
            headers=_headers(),
            params={"select": "id"},
        )
    resp.raise_for_status()
    return len(resp.json())


async def get_visit_count() -> int:
    if not SUPABASE_ENABLED:
        return _memory_visit_count

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/visits",
            headers=_headers(),
            params={"select": "id"},
        )
    resp.raise_for_status()
    return len(resp.json())


# ===================== progress & leaderboard =====================
# Supabase schema for these (create alongside `feedback` and `visits`):
#
#     create table progress (
#         user_sub text primary key,
#         name text not null,
#         email text not null,
#         xp int not null default 0,
#         completed jsonb not null default '[]'::jsonb,
#         updated_at timestamptz not null default now()
#     );
#
# RLS note: unlike feedback/visits, this table holds email addresses.
# Do NOT expose it to the anon key with a blanket SELECT policy — the
# leaderboard endpoint below deliberately returns name + xp only, never
# email, and the backend is the only thing that reads the full row.

_memory_progress: dict[str, dict] = {}


async def save_progress(user_sub: str, name: str, email: str, xp: int, completed: list[str]) -> dict:
    record = {"user_sub": user_sub, "name": name, "email": email, "xp": xp, "completed": completed}

    if not SUPABASE_ENABLED:
        _memory_progress[user_sub] = record
        return record

    headers = _headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{SUPABASE_URL}/rest/v1/progress", headers=headers, json=record)
    resp.raise_for_status()
    return resp.json()[0]


async def load_progress(user_sub: str) -> dict | None:
    if not SUPABASE_ENABLED:
        return _memory_progress.get(user_sub)

    params = {"select": "user_sub,name,xp,completed", "user_sub": f"eq.{user_sub}", "limit": "1"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{SUPABASE_URL}/rest/v1/progress", headers=_headers(), params=params)
    resp.raise_for_status()
    rows = resp.json()
    return rows[0] if rows else None


async def leaderboard(limit: int = 20) -> list[dict]:
    """Returns name + xp only — never email, even though we store it."""
    if not SUPABASE_ENABLED:
        rows = sorted(_memory_progress.values(), key=lambda r: r["xp"], reverse=True)[:limit]
        return [{"name": r["name"], "xp": r["xp"], "completed_count": len(r["completed"])} for r in rows]

    params = {"select": "name,xp,completed", "order": "xp.desc", "limit": str(limit)}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{SUPABASE_URL}/rest/v1/progress", headers=_headers(), params=params)
    resp.raise_for_status()
    return [
        {"name": r["name"], "xp": r["xp"], "completed_count": len(r.get("completed") or [])}
        for r in resp.json()
    ]
