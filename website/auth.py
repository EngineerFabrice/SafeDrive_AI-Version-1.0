# website/auth.py
"""Role-based authorization and input validation helpers."""
import functools
import re

from flask import abort, url_for
from flask_login import current_user

from . import login_manager

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_BYTES = 72          # bcrypt only uses the first 72 bytes; reject longer input
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
_USERNAME = re.compile(r"^[\w][\w .'-]{1,58}[\w.]$", re.UNICODE)


def roles_required(*roles):
    """Allow only authenticated users whose role is in ``roles``; others get 401/403."""
    def decorator(view):
        @functools.wraps(view)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if current_user.role not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapper
    return decorator


def dashboard_url(user):
    """Landing page for a user's role."""
    endpoint = {
        "admin": "routes.admin_dashboard",
        "manager": "routes.manager_dashboard",
        "umusare": "routes.umusare_dashboard",
    }.get(user.role, "routes.driver_dashboard")
    return url_for(endpoint)


def validate_registration(username, email, password, confirm):
    """Return a list of human-readable problems (empty when valid)."""
    errors = []
    if not _USERNAME.match(username or ""):
        errors.append("Name must be 3–60 characters (letters, numbers, spaces, . ' -).")
    if len(email or "") > 255 or not _EMAIL.match(email or ""):
        errors.append("Enter a valid email address.")
    errors.extend(validate_password(password))
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors


def validate_password(password):
    password = password or ""
    if len(password) < PASSWORD_MIN_LENGTH:
        return [f"Password must be at least {PASSWORD_MIN_LENGTH} characters."]
    if len(password.encode("utf-8")) > PASSWORD_MAX_BYTES:
        return [f"Password must be at most {PASSWORD_MAX_BYTES} bytes."]
    if password.isdigit() or password.isalpha():
        return ["Password must mix letters with numbers or symbols."]
    return []


_PHONE_DIGITS = re.compile(r"^\+?[0-9]{9,15}$")


def normalize_phone(raw):
    """Return an E.164-style number (+2507XXXXXXXX for Rwandan 07... numbers) or raise ValueError."""
    value = re.sub(r"[\s().-]", "", raw or "")
    if re.match(r"^07[0-9]{8}$", value):            # Rwandan national format
        value = "+250" + value[1:]
    elif re.match(r"^2507[0-9]{8}$", value):
        value = "+" + value
    if not _PHONE_DIGITS.match(value):
        raise ValueError("Enter a valid phone number, e.g. +250 78X XXX XXX.")
    return value if value.startswith("+") else "+" + value


def safe_local_path(target):
    """Return ``target`` only if it is a same-site path such as "/driver-dashboard?x=1", else None.

    Rejects absolute URLs ("https://x"), protocol-relative ("//x"), any backslash (browsers treat
    slash-backslash like "//"), percent-encoded versions of those, control characters / whitespace,
    and anything urlsplit() sees a scheme or host in.
    """
    from urllib.parse import unquote, urlsplit
    if not isinstance(target, str) or not target or len(target) > 2048:
        return None
    if not target.startswith("/") or target.startswith("//"):
        return None
    decoded = unquote(target)
    if "\\" in target or "\\" in decoded or decoded.startswith("//"):
        return None
    if any(ord(ch) < 33 or ord(ch) == 127 for ch in target):
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return None
    return target
