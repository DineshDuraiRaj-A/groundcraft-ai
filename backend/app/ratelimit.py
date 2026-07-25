"""
In-memory per-session rate limiter.

This protects the shared OPENROUTER_API_KEY (used on the free-tier path)
from being drained by one heavy visitor. It's intentionally simple —
in-memory, single-process — which is fine for a weekend build on a single
free-tier instance. If you outgrow one instance, swap this for Redis
(same interface: check_and_increment(session_id) -> bool) without touching
callers.
"""
from __future__ import annotations
import time
from collections import defaultdict

# (bucket, session_id) -> list of unix timestamps of free-tier requests
_requests: dict[tuple[str, str], list[float]] = defaultdict(list)

MAX_REQUESTS_PER_SESSION = 10
WINDOW_SECONDS = 60 * 60  # 1 hour rolling window

# Separate budgets per bucket. The assistant gets its own, more generous
# allowance: someone stuck on a mission shouldn't have to spend their
# mission budget just to ask for help.
BUCKET_LIMITS = {
    "missions": MAX_REQUESTS_PER_SESSION,
    "assistant": 20,
}


def check_and_increment(session_id: str, bucket: str = "missions") -> tuple[bool, int]:
    """
    Returns (allowed, remaining). Only call this on the free-tier path —
    user-supplied-key requests don't touch your quota, so they're exempt.
    """
    limit = BUCKET_LIMITS.get(bucket, MAX_REQUESTS_PER_SESSION)
    now = time.time()
    timestamps = _requests[(bucket, session_id)]
    # drop anything outside the rolling window
    timestamps[:] = [t for t in timestamps if now - t < WINDOW_SECONDS]

    if len(timestamps) >= limit:
        return False, 0

    timestamps.append(now)
    return True, limit - len(timestamps)
