# website/mobile_auth.py
"""Bearer-token authentication for the mobile API (table `mobile_api_tokens`, migration 0008).

* A token is 32 random bytes (URL-safe base64) from `secrets`, shown to the app once at sign-in.
* Only SHA-256(token) is stored, so a database leak does not reveal usable tokens.
* Tokens expire after MOBILE_TOKEN_TTL_DAYS (default 30), are revoked at sign-out, and stop working
  as soon as the account is deactivated (the user row is re-read on every request).
* The mobile API reads ONLY the Authorization header, never the browser session cookie. That is why
  its blueprint can be exempt from CSRF: a cross-site page cannot make a browser attach this header.
"""
import functools
import hashlib
import os
import secrets
from datetime import timedelta

from flask import g, jsonify, request

from . import get_user_by_id
from .assistance_service import transaction, utcnow

TOKEN_BYTES = 32
LAST_USED_RESOLUTION_S = 60      # last_used_at is refreshed at most once a minute per token


def token_ttl():
    try:
        days = int(os.environ.get("MOBILE_TOKEN_TTL_DAYS", "30"))
    except ValueError:
        days = 30
    return timedelta(days=max(1, min(days, 365)))


def _hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_token(user_id, device_name=None):
    """Create a token for ``user_id``; returns (token, expires_at). The plain token is never stored."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    now = utcnow()
    expires = now + token_ttl()
    device = (device_name.strip()[:80] or None) if isinstance(device_name, str) else None
    with transaction() as cursor:
        cursor.execute("INSERT INTO mobile_api_tokens (user_id, token_hash, device_name, created_at, expires_at) "
                       "VALUES (%s,%s,%s,%s,%s)", (user_id, _hash(token), device, now, expires))
    return token, expires


def _bearer():
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    value = value.strip()
    if scheme.lower() != "bearer" or not value or len(value) > 200:
        return None
    return value


def resolve(token):
    """(user, token_row_id) for a valid, unexpired, unrevoked token of an active user, else (None, None)."""
    now = utcnow()
    with transaction() as cursor:
        cursor.execute("SELECT id, user_id, last_used_at FROM mobile_api_tokens WHERE token_hash=%s "
                       "AND revoked_at IS NULL AND expires_at > %s", (_hash(token), now))
        row = cursor.fetchone()
        if row is None:
            return None, None
        if row["last_used_at"] is None or (now - row["last_used_at"]).total_seconds() >= LAST_USED_RESOLUTION_S:
            cursor.execute("UPDATE mobile_api_tokens SET last_used_at=%s WHERE id=%s", (now, row["id"]))
    user = get_user_by_id(row["user_id"])
    if user is None or not user.is_active:
        return None, None
    return user, row["id"]


def revoke(token_id):
    with transaction() as cursor:
        cursor.execute("UPDATE mobile_api_tokens SET revoked_at=%s WHERE id=%s AND revoked_at IS NULL",
                       (utcnow(), token_id))


def revoke_all(user_id):
    with transaction() as cursor:
        cursor.execute("UPDATE mobile_api_tokens SET revoked_at=%s WHERE user_id=%s AND revoked_at IS NULL",
                       (utcnow(), user_id))


def unauthorized(message="Sign in again to continue."):
    response = jsonify({"error": message, "code": "UNAUTHORIZED"})
    response.status_code = 401
    response.headers["WWW-Authenticate"] = 'Bearer realm="safedrive-mobile"'
    return response


def token_required(*roles):
    """Require a valid bearer token (and, if ``roles`` are given, one of those roles).

    The user is available as ``g.mobile_user`` and the token row id as ``g.mobile_token_id``.
    """
    def decorator(view):
        @functools.wraps(view)
        def wrapper(*args, **kwargs):
            token = _bearer()
            user, token_id = resolve(token) if token else (None, None)
            if user is None:
                return unauthorized()
            if roles and user.role not in roles:
                return jsonify({"error": "You do not have permission for this action.", "code": "FORBIDDEN"}), 403
            g.mobile_user, g.mobile_token_id = user, token_id
            return view(*args, **kwargs)
        return wrapper
    return decorator
