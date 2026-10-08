# website/legal.py
"""Versioned Terms & Conditions / Privacy Policy and their acceptance records.

Each acceptance is appended to `legal_acceptances` (never overwritten), and the
user's row keeps the versions currently accepted. When a document's version is
raised here, users who accepted an older version are asked to accept again.
"""
import os

from flask import has_request_context, request

from . import audit

TERMS_VERSION = "1.0"
PRIVACY_VERSION = "1.0"
EFFECTIVE_DATE = "8 October 2026"


def contact_email():
    """Official contact address, configurable; defaults to the project's credited contact."""
    return os.environ.get("SAFEDRIVE_CONTACT_EMAIL", "").strip() or "fabricendayisaba16@gmail.com"


def needs_acceptance(user):
    return (getattr(user, "terms_version", None) != TERMS_VERSION
            or getattr(user, "privacy_version", None) != PRIVACY_VERSION)


def record_acceptance(cursor, user_id, now):
    """Record acceptance of the CURRENT Terms and Privacy versions inside the caller's transaction."""
    ip = request.remote_addr if has_request_context() else None
    for document, version in (("TERMS", TERMS_VERSION), ("PRIVACY", PRIVACY_VERSION)):
        cursor.execute("INSERT INTO legal_acceptances (user_id, document, version, accepted_at, ip_address) "
                       "VALUES (%s,%s,%s,%s,%s)", (user_id, document, version, now, ip))
    cursor.execute("UPDATE users SET terms_version=%s, terms_accepted_at=%s, privacy_version=%s, privacy_accepted_at=%s "
                   "WHERE id=%s", (TERMS_VERSION, now, PRIVACY_VERSION, now, user_id))
    audit.record(audit.TERMS_ACCEPTED, actor_id=user_id, target_type="user", target_id=user_id,
                 details={"terms": TERMS_VERSION, "privacy": PRIVACY_VERSION}, cursor=cursor)
