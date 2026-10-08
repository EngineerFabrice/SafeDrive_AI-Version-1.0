# website/mailer.py
"""Outgoing email, configured only through the environment (no provider or account is hard-coded).

    MAIL_HOST, MAIL_PORT (587), MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, MAIL_USE_TLS (1), MAIL_USE_SSL (0)
    MAIL_BACKEND   smtp | file | memory   (optional override)

Which transport is used:
* SMTP when it is configured: MAIL_HOST plus MAIL_USERNAME and MAIL_PASSWORD (e.g. Gmail:
  smtp.gmail.com, port 587, STARTTLS, a Google App Password), or MAIL_BACKEND=smtp for a relay
  that needs no login.
* Otherwise "file" in development (instance/dev_outbox/, git-ignored, nothing is sent) and
  "memory" in testing (tests never reach a real mail server).
* Production without SMTP refuses to send, so delivery is never faked.

Message contents (which can contain one-time codes), recipients and credentials are never logged;
SMTP errors are logged by type only and never shown to users.

Development check (never available when SAFEDRIVE_ENV=production):

    python -m website.mailer smtp-test --to you@example.com
"""
import argparse
import logging
import os
import smtplib
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage

from .config import env_flag

log = logging.getLogger(__name__)
OUTBOX = []                                   # "memory" backend (tests)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_OUTBOX_DIR = os.path.join(_ROOT, "instance", "dev_outbox")
SMTP_TIMEOUT_S = 15


class MailError(RuntimeError):
    pass


def _env():
    return os.environ.get("SAFEDRIVE_ENV", "development").strip().lower()


def smtp_configured():
    """True when SMTP has a host and credentials (or MAIL_BACKEND=smtp explicitly allows a login-less relay)."""
    host = os.environ.get("MAIL_HOST", "").strip()
    if not host:
        return False
    if os.environ.get("MAIL_BACKEND", "").strip().lower() == "smtp":
        return True
    return bool(os.environ.get("MAIL_USERNAME", "").strip() and os.environ.get("MAIL_PASSWORD", ""))


def backend():
    configured = os.environ.get("MAIL_BACKEND", "").strip().lower()
    if configured in ("file", "memory"):
        return None if _env() == "production" else configured       # never fake delivery in production
    if _env() == "testing" and configured != "smtp":
        return "memory"              # automated tests never reach a real mail server, even if .env has SMTP set
    if configured == "smtp" or smtp_configured():
        return "smtp" if os.environ.get("MAIL_HOST", "").strip() else None
    return {"development": "file", "testing": "memory"}.get(_env())


def sender():
    return os.environ.get("MAIL_FROM", "").strip() or os.environ.get("MAIL_USERNAME", "").strip() \
        or "SafeDrive AI <no-reply@safedrive.local>"


def send(to, subject, body, html=None):
    """Send one email (plain text, optional HTML alternative); raises MailError when it cannot be delivered.

    Returning normally means the message was handed to the configured transport.
    """
    kind = backend()
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender(), to, subject
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")
    if kind == "memory":
        OUTBOX.append({"to": to, "subject": subject, "body": body, "html": html})
        return
    if kind == "file":
        os.makedirs(DEV_OUTBOX_DIR, exist_ok=True)
        path = os.path.join(DEV_OUTBOX_DIR, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".eml")
        with open(path, "wb") as fh:
            fh.write(bytes(msg))
        log.info("Development email written to instance/dev_outbox/ (not sent)")
        return
    if kind != "smtp":
        raise MailError("Email delivery is not configured.")
    _send_smtp(msg)
    log.info("Email handed to the SMTP server")


def _send_smtp(msg):
    host = os.environ.get("MAIL_HOST", "").strip()
    user, password = os.environ.get("MAIL_USERNAME", "").strip(), os.environ.get("MAIL_PASSWORD", "")
    try:
        port = int(os.environ.get("MAIL_PORT", "587"))
    except ValueError:
        log.warning("Email delivery failed: MAIL_PORT is not a number")
        raise MailError("Email delivery is not configured correctly.")
    stage = "connect"
    try:
        context = ssl.create_default_context()
        if env_flag("MAIL_USE_SSL"):
            server = smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_S, context=context)
        else:
            server = smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_S)
        with server:                                   # closed on every path, including a failed STARTTLS
            if not env_flag("MAIL_USE_SSL") and env_flag("MAIL_USE_TLS", True):
                stage = "starttls"
                server.starttls(context=context)       # Gmail on 587 requires STARTTLS before login
            if user:
                # Note: when the server rejects the credentials (535) and then closes the connection,
                # smtplib's retry with the next AUTH method surfaces as SMTPServerDisconnected.
                stage = "auth"
                server.login(user, password)
            stage = "send"
            server.send_message(msg)
    except (OSError, smtplib.SMTPException) as exc:
        code = getattr(exc, "smtp_code", None)
        log.warning("Email delivery failed at %s: %s%s", stage.upper(), type(exc).__name__,
                    f" (SMTP {code})" if code else "")
        if stage == "auth" and "@" not in user:
            log.warning("MAIL_USERNAME is not an email address; Gmail SMTP requires the full Gmail address")
        error = MailError("Email could not be delivered.")
        error.stage = stage                            # safe to show to a developer: no credentials, no message
        raise error from exc


# ---------------------------------------------------------------- development check
def _mask(value):
    local, _, domain = (value or "").partition("@")
    return f"{local[:2]}***@{domain}" if domain else ("set" if value else "not set")


def smtp_test(to, out=sys.stdout):
    """Send one plain test email through SMTP (no code, no secrets). Returns a process exit code."""
    if _env() == "production":
        print("Refused: the SMTP test is a development tool (SAFEDRIVE_ENV=production).", file=out)
        return 2
    print(f"MAIL_HOST:     {os.environ.get('MAIL_HOST', '').strip() or 'not set'}", file=out)
    print(f"MAIL_PORT:     {os.environ.get('MAIL_PORT', '587')}", file=out)
    print(f"MAIL_USE_TLS:  {env_flag('MAIL_USE_TLS', True)}   MAIL_USE_SSL: {env_flag('MAIL_USE_SSL')}", file=out)
    print(f"MAIL_USERNAME: {_mask(os.environ.get('MAIL_USERNAME', '').strip())}", file=out)
    print(f"MAIL_PASSWORD: {'set' if os.environ.get('MAIL_PASSWORD') else 'not set'}", file=out)   # never the value
    print(f"MAIL_FROM:     {_mask(sender())}", file=out)
    username = os.environ.get("MAIL_USERNAME", "").strip()
    if username and "@" not in username:
        print("WARNING: MAIL_USERNAME is not an email address. Gmail SMTP requires the full Gmail address "
              "(the account the App Password belongs to), not the App Password's name.", file=out)
    if backend() != "smtp":
        print(f"SMTP is not configured; the app would use the '{backend()}' transport. Nothing was sent.", file=out)
        return 2
    try:
        send(to, "SafeDrive AI: SMTP test",
             "This is a test message from SafeDrive AI to confirm that email delivery works.\n"
             "No action is needed.\n")
    except MailError as exc:
        stage = getattr(exc, "stage", "unknown")
        print(f"FAILED at {stage.upper()}: the SMTP server did not accept the message "
              "(see the log line above for the error type).", file=out)
        if stage == "auth":
            print("Hint: the server rejected MAIL_USERNAME / MAIL_PASSWORD. For Gmail, MAIL_USERNAME must be the exact "
                  "Gmail address the App Password was created in, 2-Step Verification must be on, and MAIL_PASSWORD "
                  "must be that account's current 16-letter App Password (not the normal password).", file=out)
        return 1
    print(f"OK: the message was accepted by {os.environ.get('MAIL_HOST', '').strip()} for {_mask(to)}. "
          "Check that inbox (and its Spam folder).", file=out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SafeDrive AI email tools (development)")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("smtp-test", help="send one test email through the configured SMTP server")
    t.add_argument("--to", required=True, help="recipient address for the test message")
    args = parser.parse_args()
    sys.exit(smtp_test(args.to))
