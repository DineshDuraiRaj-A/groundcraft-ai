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
import time
from typing import Optional

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

# Set once a Supabase call fails (bad key, missing tables, project paused).
# The app then serves from memory instead of returning 500s. This matters
# because /api/status is polled by the lab boot gate: a half-configured
# database used to take the ENTIRE app down, not just persistence.
_supabase_broken: dict = {"failed": False, "reason": ""}


def supabase_live() -> bool:
    return SUPABASE_ENABLED and not _supabase_broken["failed"]


def _note_supabase_failure(exc: Exception) -> None:
    if not _supabase_broken["failed"]:
        _supabase_broken["failed"] = True
        _supabase_broken["reason"] = str(exc)[:200]


def storage_mode() -> str:
    if not supabase_live():
        return "in-memory"
    return "supabase" if supabase_live() else "in-memory (supabase unreachable)"


def storage_note() -> str:
    return _supabase_broken["reason"] if _supabase_broken["failed"] else ""

# ---- in-memory fallback ----
_memory_feedback: list[dict] = []
# Real count, not a vanity seed. It used to start at 4812, which made the
# "visitors today" figure meaningless — and because cross-site cookies were
# broken, every single request minted a new session and incremented it again.
_memory_visit_count = 0
_counted_sessions: set[str] = set()


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def add_feedback(name: Optional[str], emoji: str, text: str) -> dict:
    entry = {"name": name or "Anonymous", "emoji": emoji, "text": text}

    if not supabase_live():
        entry["ts"] = time.time()
        _memory_feedback.append(entry)
        return entry

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{SUPABASE_URL}/rest/v1/feedback", headers=_headers(), json=entry)
        resp.raise_for_status()
        row = resp.json()[0]
        return {"name": row["name"], "emoji": row["emoji"], "text": row["text"], "ts": row["created_at"]}
    except Exception as e:
        _note_supabase_failure(e)
        entry["ts"] = time.time()
        _memory_feedback.append(entry)
        return entry


async def list_feedback(limit: int = 50) -> list[dict]:
    if not supabase_live():
        return list(reversed(_memory_feedback[-limit:]))

    params = {"select": "name,emoji,text,created_at", "order": "created_at.desc", "limit": str(limit)}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/feedback", headers=_headers(), params=params)
        resp.raise_for_status()
        return [{"name": r["name"], "emoji": r["emoji"], "text": r["text"], "ts": r["created_at"]} for r in resp.json()]
    except Exception as e:
        _note_supabase_failure(e)
        return list(reversed(_memory_feedback[-limit:]))


async def record_visit(session_id: str | None = None) -> int:
    """Count each visitor once, not once per request."""
    global _memory_visit_count

    if session_id:
        if session_id in _counted_sessions:
            return await get_visit_count()
        _counted_sessions.add(session_id)

    if not supabase_live():
        _memory_visit_count += 1
        return _memory_visit_count

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{SUPABASE_URL}/rest/v1/visits", headers=_headers(), json={})
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/visits",
                                    headers=_headers(), params={"select": "id"})
        resp.raise_for_status()
        return len(resp.json())
    except Exception as e:
        _note_supabase_failure(e)
        _memory_visit_count += 1
        return _memory_visit_count


async def get_visit_count() -> int:
    if not supabase_live():
        return _memory_visit_count

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/visits",
                                    headers=_headers(), params={"select": "id"})
        resp.raise_for_status()
        return len(resp.json())
    except Exception as e:
        _note_supabase_failure(e)
        return _memory_visit_count


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

    if not supabase_live():
        _memory_progress[user_sub] = record
        return record

    headers = _headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{SUPABASE_URL}/rest/v1/progress", headers=headers, json=record)
        resp.raise_for_status()
        return resp.json()[0]
    except Exception as e:
        _note_supabase_failure(e)
        _memory_progress[user_sub] = record
        return record


async def load_progress(user_sub: str) -> dict | None:
    if not supabase_live():
        return _memory_progress.get(user_sub)

    params = {"select": "user_sub,name,xp,completed", "user_sub": f"eq.{user_sub}", "limit": "1"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/progress", headers=_headers(), params=params)
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None
    except Exception as e:
        _note_supabase_failure(e)
        return _memory_progress.get(user_sub)


async def leaderboard(limit: int = 20) -> list[dict]:
    """Returns name + xp only — never email, even though we store it."""
    if not supabase_live():
        rows = sorted(_memory_progress.values(), key=lambda r: r["xp"], reverse=True)[:limit]
        return [{"name": r["name"], "xp": r["xp"], "completed_count": len(r["completed"])} for r in rows]

    params = {"select": "name,xp,completed", "order": "xp.desc", "limit": str(limit)}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/progress", headers=_headers(), params=params)
        resp.raise_for_status()
        return [
            {"name": r["name"], "xp": r["xp"], "completed_count": len(r.get("completed") or [])}
            for r in resp.json()
        ]
    except Exception as e:
        _note_supabase_failure(e)
        rows = sorted(_memory_progress.values(), key=lambda r: r["xp"], reverse=True)[:limit]
        return [{"name": r["name"], "xp": r["xp"], "completed_count": len(r["completed"])} for r in rows]
