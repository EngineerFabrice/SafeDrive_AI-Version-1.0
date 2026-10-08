# website/email_otp.py
"""Email verification with one-time codes (table `email_otps`).

* 6-digit code from `secrets`, valid OTP_TTL_S (10 min), single use.
* Only HMAC-SHA256(secret key, user id, code) is stored; codes are never logged or sent to the page.
* At most OTP_MAX_ATTEMPTS wrong guesses per code; a new code invalidates the previous one.
* Resend cooldown OTP_RESEND_S (60 s) and at most OTP_MAX_PER_HOUR codes per user per hour,
  so guessing is bounded to a few dozen tries per hour against a 1-in-a-million code.
"""
import hashlib
import hmac
import secrets
from datetime import timedelta

from flask import current_app

from . import audit
from . import mailer
from .assistance_service import AssistanceError, transaction, utcnow

PURPOSE = "EMAIL_VERIFY"
OTP_TTL_S = 600
OTP_RESEND_S = 60
OTP_MAX_ATTEMPTS = 5
OTP_MAX_PER_HOUR = 5
GENERIC_FAILURE = "That verification code is incorrect or has expired. Please request a new code."


def mask_email(email):
    local, _, domain = (email or "").partition("@")
    shown = local[:4] if len(local) > 4 else local[:1]
    return f"{shown}***@{domain}" if domain else "***"


def _hash(user_id, code):
    key = current_app.config["SECRET_KEY"].encode()
    return hmac.new(key, f"{int(user_id)}:{code}".encode(), hashlib.sha256).hexdigest()


def _latest(cursor, user_id, lock=False):
    cursor.execute("SELECT * FROM email_otps WHERE user_id=%s AND purpose=%s ORDER BY id DESC LIMIT 1"
                   + (" FOR UPDATE" if lock else ""), (user_id, PURPOSE))
    return cursor.fetchone()


def resend_wait_s(user_id):
    """Seconds until another code may be requested (0 = now)."""
    with transaction() as cursor:
        last = _latest(cursor, user_id)
    if not last:
        return 0
    return max(0, int(OTP_RESEND_S - (utcnow() - last["created_at"]).total_seconds()))


COOLDOWN_MESSAGE = "Please wait {wait} seconds before requesting another code."
HOURLY_MESSAGE = "Too many codes were requested. Please try again in an hour."
TOO_MANY_ATTEMPTS = "Too many incorrect attempts. Please request a new code."
DELIVERY_FAILED = "We could not send the verification email. Please try RESEND CODE in a moment."


def issue(user_id):
    """Create, store (hashed) and email a new code; raises AssistanceError on cooldown / limits / delivery.

    The code row is committed BEFORE sending, so the cooldown and hourly cap also count attempts
    whose delivery fails; an undelivered code is invalidated immediately (it can never be used).
    """
    with transaction() as cursor:
        cursor.execute("SELECT id, email, is_active, email_verified_at FROM users WHERE id=%s FOR UPDATE", (user_id,))
        user = cursor.fetchone()
        if not user or not user["is_active"]:
            raise AssistanceError(403, "INACTIVE", "This account cannot be verified. Please contact support.")
        if user["email_verified_at"]:
            raise AssistanceError(409, "ALREADY_VERIFIED", "Your email address is already verified.")
        now = utcnow()
        last = _latest(cursor, user_id, lock=True)
        if last and (now - last["created_at"]).total_seconds() < OTP_RESEND_S:
            wait = int(OTP_RESEND_S - (now - last["created_at"]).total_seconds()) + 1
            raise AssistanceError(429, "COOLDOWN", COOLDOWN_MESSAGE.format(wait=wait))
        cursor.execute("SELECT COUNT(*) AS n FROM email_otps WHERE user_id=%s AND purpose=%s AND created_at>=%s",
                       (user_id, PURPOSE, now - timedelta(hours=1)))
        if cursor.fetchone()["n"] >= OTP_MAX_PER_HOUR:
            raise AssistanceError(429, "TOO_MANY", HOURLY_MESSAGE)
        cursor.execute("UPDATE email_otps SET invalidated_at=%s WHERE user_id=%s AND purpose=%s AND consumed_at IS NULL "
                       "AND invalidated_at IS NULL", (now, user_id, PURPOSE))
        code = f"{secrets.randbelow(10 ** 6):06d}"
        cursor.execute("INSERT INTO email_otps (user_id, purpose, code_hash, sent_to, created_at, expires_at) "
                       "VALUES (%s,%s,%s,%s,%s,%s)", (user_id, PURPOSE, _hash(user_id, code), user["email"], now,
                                                      now + timedelta(seconds=OTP_TTL_S)))
        otp_id = cursor.lastrowid
        audit.record(audit.EMAIL_OTP_SENT, actor_id=user_id, target_type="user", target_id=user_id, cursor=cursor)
    try:
        mailer.send(user["email"], OTP_SUBJECT, *otp_email(code))
    except mailer.MailError:
        with transaction() as cursor:
            cursor.execute("UPDATE email_otps SET invalidated_at=%s WHERE id=%s", (utcnow(), otp_id))
            audit.record(audit.EMAIL_DELIVERY_FAILED, actor_id=user_id, target_type="user", target_id=user_id,
                         cursor=cursor)
        raise AssistanceError(503, "DELIVERY_FAILED", DELIVERY_FAILED)


OTP_SUBJECT = "SafeDrive AI — Email Verification"


def otp_email(code):
    """(plain text, HTML) of the verification email: the code, its lifetime and a warning; nothing personal."""
    minutes = OTP_TTL_S // 60
    text = ("SafeDrive AI\n"
            "Email Verification\n\n"
            f"Your 6-digit verification code is: {code}\n\n"
            f"The code expires in {minutes} minutes and can be used once.\n"
            "Do not share this code with anyone. SafeDrive staff will never ask you for it.\n\n"
            "If you did not create a SafeDrive AI account, you can ignore this email.\n")
    html = ('<!doctype html><html><body style="margin:0;background:#f4f7fb;font-family:Arial,Helvetica,sans-serif;'
            'color:#1d2b3d;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" '
            'style="padding:32px 12px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="max-width:480px;background:#ffffff;border:1px solid #e3e9f1;border-radius:14px;"><tr><td style="padding:28px;">'
            '<div style="font-size:13px;font-weight:bold;letter-spacing:.12em;color:#0d2f57;">SAFEDRIVE AI</div>'
            '<h1 style="font-size:22px;color:#0d2f57;margin:6px 0 16px;">Email Verification</h1>'
            '<p style="font-size:15px;margin:0 0 12px;">Your 6-digit verification code:</p>'
            f'<div style="font-size:32px;font-weight:bold;letter-spacing:8px;color:#0d2f57;background:#f6f9fd;'
            f'border:1px solid #dfe7f1;border-radius:10px;padding:14px;text-align:center;">{code}</div>'
            f'<p style="font-size:14px;margin:16px 0 6px;">The code expires in <b>{minutes} minutes</b> and can be used once.</p>'
            '<p style="font-size:14px;margin:0 0 16px;"><b>Do not share this code with anyone.</b> SafeDrive staff will '
            'never ask you for it.</p><p style="font-size:12px;color:#6b7a8f;margin:0;">If you did not create a SafeDrive AI '
            'account, you can ignore this email.</p></td></tr></table></td></tr></table></body></html>')
    return text, html


def last_delivery_ok(user_id):
    """True if the latest code was handed to the mail transport, False if its delivery failed, None if none yet.

    The latest code can only be invalidated by a failed delivery (a newer code would be the latest one).
    """
    with transaction() as cursor:
        last = _latest(cursor, user_id)
    if last is None:
        return None
    return last["invalidated_at"] is None or last["consumed_at"] is not None


# ---------------------------------------------------------------- pending verification after registration
# The browser keeps the state of a just-registered (not yet signed-in) verification in its session cookie.
# Flask session cookies are signed but readable, so the state is ENCRYPTED (Fernet, key derived from the
# secret key) with fixed-width fields: a new account and a duplicate-email "decoy" produce cookies of the
# same shape and length, and both follow the same cooldown / hourly / attempt rules and messages.
def _fernet():
    import base64
    from cryptography.fernet import Fernet
    key = hashlib.sha256(b"safedrive-pending-verification:" + current_app.config["SECRET_KEY"].encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def flow_new(user_id, email):
    """State for a pending verification; user_id None marks a duplicate-email decoy."""
    return {"u": int(user_id or 0), "e": email, "s": [], "a": 0, "f": 0}


def flow_dump(state):
    import json
    body = {"u": f"{state['u']:010d}", "e": state["e"], "s": [f"{int(t):010d}" for t in state["s"][-OTP_MAX_PER_HOUR:]],
            "a": f"{min(state['a'], 99):02d}", "f": "1" if state.get("f") else "0"}
    return _fernet().encrypt(json.dumps(body, separators=(",", ":")).encode()).decode()


def flow_load(token):
    import json
    from cryptography.fernet import InvalidToken
    try:
        body = json.loads(_fernet().decrypt(str(token or "").encode(), ttl=24 * 3600))
        return {"u": int(body["u"]), "e": body["e"], "s": [int(t) for t in body["s"]], "a": int(body["a"]),
                "f": int(body.get("f", "0"))}
    except (InvalidToken, ValueError, KeyError, TypeError):
        return None


def flow_wait(state, now):
    """Resend countdown for a pending verification (same computation for real and decoy flows)."""
    return max(0, int(OTP_RESEND_S - (now - state["s"][-1]))) if state["s"] else 0


def flow_check_send(state, now):
    """Cooldown / hourly cap with the same messages as issue()."""
    if state["s"] and now - state["s"][-1] < OTP_RESEND_S:
        raise AssistanceError(429, "COOLDOWN", COOLDOWN_MESSAGE.format(wait=int(OTP_RESEND_S - (now - state["s"][-1])) + 1))
    if len([t for t in state["s"] if now - t < 3600]) >= OTP_MAX_PER_HOUR:
        raise AssistanceError(429, "TOO_MANY", HOURLY_MESSAGE)


def flow_record_send(state, now, delivered=True):
    """Record a send attempt (it counts for cooldown / hourly cap even if delivery failed)."""
    state["s"].append(int(now))
    state["a"] = 0
    state["f"] = 0 if delivered else 1


def flow_delivered(state):
    """Same answer as last_delivery_ok(), for a pending (real or decoy) verification."""
    return None if not state["s"] else not state.get("f")


def decoy_send(email):
    """Duplicate-email registration: tell the real owner instead of sending a code (never a code)."""
    try:
        mailer.send(email, "SafeDrive AI: someone tried to register with your email",
                    "Someone (possibly you) tried to create a new SafeDrive AI account with this email address.\n"
                    "You already have an account: please sign in instead. If this was not you, you can ignore this "
                    "email; your account has not been changed.\n")
    except mailer.MailError:
        raise AssistanceError(503, "DELIVERY_FAILED", DELIVERY_FAILED)


def decoy_verify(state):
    """A decoy code can never be correct; failures mirror verify() (generic, then too-many-attempts)."""
    state["a"] += 1
    if state["a"] >= OTP_MAX_ATTEMPTS:
        raise AssistanceError(429, "TOO_MANY_ATTEMPTS", TOO_MANY_ATTEMPTS)
    raise AssistanceError(400, "INVALID_CODE", GENERIC_FAILURE)


def verify(user_id, code):
    """Check a code; on success the email is verified (once). Raises AssistanceError with a generic message."""
    code = "".join(ch for ch in str(code or "") if ch.isdigit())
    with transaction() as cursor:
        cursor.execute("SELECT id, is_active, email_verified_at FROM users WHERE id=%s FOR UPDATE", (user_id,))
        user = cursor.fetchone()
        if not user or not user["is_active"]:
            raise AssistanceError(403, "INACTIVE", "This account cannot be verified. Please contact support.")
        if user["email_verified_at"]:
            raise AssistanceError(409, "ALREADY_VERIFIED", "Your email address is already verified.")
        otp = _latest(cursor, user_id, lock=True)
        now = utcnow()
        if (otp is None or otp["consumed_at"] or otp["invalidated_at"] or otp["expires_at"] <= now):
            raise AssistanceError(400, "INVALID_CODE", GENERIC_FAILURE)
        if otp["attempts"] >= OTP_MAX_ATTEMPTS:
            raise AssistanceError(429, "TOO_MANY_ATTEMPTS", TOO_MANY_ATTEMPTS)
        failure = None
        if len(code) != 6 or not hmac.compare_digest(otp["code_hash"], _hash(user_id, code)):
            # the failed attempt is committed (the error is raised after the transaction ends)
            cursor.execute("UPDATE email_otps SET attempts=attempts+1 WHERE id=%s", (otp["id"],))
            audit.record(audit.EMAIL_OTP_FAILED, actor_id=user_id, target_type="user", target_id=user_id, cursor=cursor)
            failure = (AssistanceError(429, "TOO_MANY_ATTEMPTS", TOO_MANY_ATTEMPTS)
                       if otp["attempts"] + 1 >= OTP_MAX_ATTEMPTS else AssistanceError(400, "INVALID_CODE", GENERIC_FAILURE))
        else:
            cursor.execute("UPDATE email_otps SET consumed_at=%s WHERE id=%s", (now, otp["id"]))
            cursor.execute("UPDATE users SET email_verified_at=%s WHERE id=%s", (now, user_id))
            audit.record(audit.EMAIL_VERIFIED, actor_id=user_id, target_type="user", target_id=user_id, cursor=cursor)
    if failure:
        raise failure
