"""
Per-session rate limiting for the shared provider keys.

Two things this got wrong in production, both fixed here:

1. It counted API CALLS, not user actions. One "Ask" in the UI fires two
   requests (grounded + ungrounded, so the learner can compare them), and
   the comparison mission fires one per model. A 10-unit budget therefore
   allowed only FIVE questions an hour, and users hit
   "You've hit the free-tier session limit" almost immediately.
   Actions are now deduplicated by an action token, so a paired comparison
   costs one unit, not two.

2. The budget was sized for a single OpenRouter key (50 requests/day). With
   Groq (~14,400/day), Mistral and Gemini also configured, that cap was
   roughly two orders of magnitude too conservative. The budget now scales
   with the number of providers actually configured.

Still in-memory and single-process, which is fine on one instance. The
interface is unchanged, so swapping in Redis later touches nothing else.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

# (bucket, session_id) -> [unix timestamps]
_requests: dict[tuple[str, str], list[float]] = defaultdict(list)

# action_token -> (bucket, session_id) already charged for
_charged_actions: dict[str, float] = {}

WINDOW_SECONDS = 60 * 60          # rolling hour
_ACTION_TTL = 120                 # a single user action's fan-out window

# Base budgets assuming ONE modest provider. Scaled up below per extra key.
BASE_LIMITS = {
    "missions": 25,
    "assistant": 30,
}

_PROVIDER_ENVS = ("OPENROUTER_API_KEY", "GROQ_API_KEY",
                  "MISTRAL_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY")


def _configured_providers() -> int:
    return sum(1 for e in _PROVIDER_ENVS if os.environ.get(e))


def limit_for(bucket: str) -> int:
    """
    Budget scales with configured providers: more keys means more real
    headroom, so there is no reason to keep learners on a starvation diet.
    Override with RATE_LIMIT_MISSIONS / RATE_LIMIT_ASSISTANT.
    """
    override = os.environ.get(f"RATE_LIMIT_{bucket.upper()}")
    if override and override.isdigit():
        return int(override)
    base = BASE_LIMITS.get(bucket, 25)
    return base * max(1, _configured_providers())


def _prune_actions(now: float) -> None:
    for token in [t for t, ts in _charged_actions.items() if now - ts > _ACTION_TTL]:
        _charged_actions.pop(token, None)


def check_and_increment(session_id: str, bucket: str = "missions",
                        action: str | None = None) -> tuple[bool, int]:
    """
    Returns (allowed, remaining).

    `action` is an opaque token identifying one user action. Every request
    sharing a token is charged once — so the grounded/ungrounded pair, or a
    three-model comparison, costs a single unit.

    Only call this on the shared-key path; requests using a visitor's own
    API key never touch our quota.
    """
    now = time.time()
    limit = limit_for(bucket)
    key = (bucket, session_id)

    timestamps = _requests[key]
    timestamps[:] = [t for t in timestamps if now - t < WINDOW_SECONDS]
    remaining = max(0, limit - len(timestamps))

    if action:
        _prune_actions(now)
        token = f"{bucket}:{session_id}:{action}"
        if token in _charged_actions:
            return True, remaining          # already paid for this action
        if len(timestamps) >= limit:
            return False, 0
        _charged_actions[token] = now
        timestamps.append(now)
        return True, limit - len(timestamps)

    if len(timestamps) >= limit:
        return False, 0
    timestamps.append(now)
    return True, limit - len(timestamps)


def status(session_id: str) -> dict:
    """Used by /api/status so the UI can warn before the wall, not after."""
    now = time.time()
    out = {}
    for bucket in BASE_LIMITS:
        ts = [t for t in _requests[(bucket, session_id)] if now - t < WINDOW_SECONDS]
        limit = limit_for(bucket)
        out[bucket] = {"used": len(ts), "limit": limit, "remaining": max(0, limit - len(ts))}
    return out
