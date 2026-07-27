"""
Lightweight Google Sign-In.

Deliberately minimal: we verify the Google ID token server-side and keep
ONLY the user's name, email, and Google subject id (used as the stable
primary key). No profile picture, no refresh token, no OAuth scopes
beyond basic identity — the sole purpose is persisting a learner's
progress across sessions and devices.

Flow:
  1. Frontend uses Google Identity Services to get an ID token (JWT).
  2. Frontend POSTs it to /api/auth/google.
  3. We verify the signature + audience against Google's public keys.
  4. We issue our own opaque session token, stored in an httpOnly cookie.

If GOOGLE_CLIENT_ID isn't configured, sign-in is disabled gracefully and
the app runs in guest mode (progress lives in memory for that session
only) — so the app is never broken by missing config.
"""
from __future__ import annotations
import os
import secrets
from dataclasses import dataclass
from typing import Optional

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
AUTH_ENABLED = bool(GOOGLE_CLIENT_ID)


@dataclass
class User:
    sub: str      # Google's stable unique id for this account
    email: str
    name: str


class AuthError(Exception):
    pass


# session_token -> User. In-memory: fine for a single instance, and
# sessions are cheap to re-establish (the user just signs in again).
# Swap for Redis if you scale past one backend process.
_sessions: dict[str, User] = {}


async def verify_google_token(id_token_str: str) -> User:
    """Verify a Google ID token and extract the minimal profile we keep."""
    if not AUTH_ENABLED:
        raise AuthError("Google sign-in is not configured on this server (missing GOOGLE_CLIENT_ID).")

    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token
    except ImportError as e:
        raise AuthError(
            "Google sign-in needs the 'google-auth' package, which isn't installed on this "
            "server. Add these two lines to backend/requirements.txt and redeploy:\n"
            "    google-auth==2.35.0\n"
            "    requests==2.32.3\n"
            "(If they're already there, the deploy is running an older commit — check the "
            "build log on Render for 'Installing google-auth'.)"
        ) from e

    try:
        # This validates signature, expiry, issuer, and audience.
        claims = google_id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError as e:
        raise AuthError(f"Invalid Google token: {e}") from e

    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise AuthError("Unexpected token issuer.")

    email = claims.get("email")
    if not email or not claims.get("email_verified"):
        raise AuthError("Google account email is missing or unverified.")

    return User(
        sub=claims["sub"],
        email=email,
        name=claims.get("name") or email.split("@")[0],
    )


def create_session(user: User) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = user
    return token


def get_user(session_token: Optional[str]) -> Optional[User]:
    if not session_token:
        return None
    return _sessions.get(session_token)


def destroy_session(session_token: Optional[str]) -> None:
    if session_token:
        _sessions.pop(session_token, None)
