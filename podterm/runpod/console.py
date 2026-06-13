"""Clerk console-session auth for the hapi machine-logs endpoint.

The hapi.runpod.net logs endpoint is the console's internal backend and only
accepts a short-lived (~60s) Clerk *console* session JWT — the RunPod API key
gets 403. We mint a fresh JWT on demand from the user's `__client` cookie
(RUNPOD_CONSOLE_CLIENT_COOKIE, set in .env), the same way the console does.
All of this is best-effort: any failure returns None and the boot panel just
never appears. See memory hapi-logs-auth / chrome-cookie-read-blocked.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request

_CLERK_BASE = "https://clerk.runpod.io/v1"
_CLERK_QS = "__clerk_api_version=2025-11-10&_clerk_js_version=5.125.13"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"

# Cached across calls: the derived session id, and the last minted JWT + its
# expiry (epoch seconds). Tokens live ~60s; re-mint a few seconds early.
_console_sid: str | None = None
_console_jwt: tuple[str, float] | None = None  # (jwt, exp_epoch)


def _console_cookie() -> str | None:
    return os.environ.get("RUNPOD_CONSOLE_CLIENT_COOKIE") or None


def _clerk_request(path: str, cookie: str) -> dict:
    req = urllib.request.Request(
        f"{_CLERK_BASE}/{path}?{_CLERK_QS}",
        data=b"",  # POST; GET endpoints tolerate an empty body here
        headers={
            "Origin": "https://console.runpod.io",
            "Referer": "https://console.runpod.io/",
            "User-Agent": _BROWSER_UA,
            "Cookie": f"__client={cookie}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _console_session_id(cookie: str) -> str | None:
    global _console_sid
    if _console_sid:
        return _console_sid
    try:
        # GET /client lists the active session(s); empty-body POST is rejected,
        # so issue it as a real GET via a tweaked request.
        req = urllib.request.Request(
            f"{_CLERK_BASE}/client?{_CLERK_QS}",
            headers={"Origin": "https://console.runpod.io", "User-Agent": _BROWSER_UA,
                     "Cookie": f"__client={cookie}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        r = data.get("response") or data
        _console_sid = r.get("last_active_session_id") or (
            (r.get("sessions") or [{}])[0].get("id"))
    except Exception:
        _console_sid = None
    return _console_sid


def console_jwt_fresh() -> str | None:
    """Return a valid console JWT, minting (and caching) one if needed."""
    global _console_jwt
    if _console_jwt and _console_jwt[1] - time.time() > 8:
        return _console_jwt[0]
    cookie = _console_cookie()
    if not cookie:
        return None
    sid = _console_session_id(cookie)
    if not sid:
        return None
    try:
        data = _clerk_request(f"client/sessions/{sid}/tokens", cookie)
        jwt = data.get("jwt")
    except Exception:
        return None
    if not jwt:
        return None
    # Trust the JWT's own exp claim rather than guessing a fixed TTL.
    exp = time.time() + 55
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp", exp)
    except Exception:
        pass
    _console_jwt = (jwt, float(exp))
    return jwt
