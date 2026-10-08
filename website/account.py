# website/account.py
"""Email verification (OTP), Terms & Conditions / Privacy pages and Terms re-acceptance.

The account being verified is either the signed-in user or the account just registered in
this browser session (session["verify"]); it is never taken from the form. All POSTs are
CSRF-protected by the global CSRFProtect.
"""
import time

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_user

from . import ROLES, get_user_by_id, legal
from . import email_otp
from .assistance_service import AssistanceError, transaction, utcnow
from .auth import dashboard_url, roles_required, safe_local_path

account = Blueprint("account", __name__)


def _pending():
    """(state, masked email) of the verification started by registration in this browser, or (None, None)."""
    pending = session.get("verify") or {}
    state = email_otp.flow_load(pending.get("t")) if pending.get("t") else None
    return (state, pending.get("masked")) if state else (None, None)


def _save(state, masked):
    session["verify"] = {"t": email_otp.flow_dump(state), "masked": masked}


@account.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    state, masked = (None, None) if current_user.is_authenticated else _pending()
    if not current_user.is_authenticated and state is None:
        flash("Sign in to verify your email address.", "info")
        return redirect(url_for("routes.login"))
    if current_user.is_authenticated:
        uid, masked = int(current_user.id), email_otp.mask_email(current_user.email)
        if current_user.email_verified:
            flash("Your email address is already verified.", "info")
            return redirect(dashboard_url(current_user))
    else:
        uid = state["u"] or None                          # None: duplicate-email decoy (never succeeds)
    if request.method == "POST":
        try:
            if uid is None:
                email_otp.decoy_verify(state)
            else:
                email_otp.verify(uid, request.form.get("code"))
        except AssistanceError as exc:
            if state is not None:
                if uid is not None:
                    state["a"] += 1                       # mirror the server count in both flows
                _save(state, masked)
            flash(exc.message, "danger" if exc.code != "ALREADY_VERIFIED" else "info")
            return redirect(url_for("account.verify_email"))
        _after_verified(uid)
        if not current_user.is_authenticated:
            user = get_user_by_id(uid)
            if user and user.is_active:
                login_user(user)
        session.pop("verify", None)
        flash("Email verified successfully.", "success")
        user = get_user_by_id(uid)
        return redirect(dashboard_url(user) if user else url_for("routes.login"))
    # Both pending flows (new account and decoy) compute the countdown the same way.
    wait = email_otp.flow_wait(state, time.time()) if state is not None else email_otp.resend_wait_s(uid)
    # Only claim "we sent a code" when the last code really was handed to the mail transport.
    delivered = email_otp.flow_delivered(state) if state is not None else email_otp.last_delivery_ok(uid)
    return render_template("verify-email.html", masked=masked, resend_wait=wait, delivered=delivered,
                           signed_in=current_user.is_authenticated, ttl_min=email_otp.OTP_TTL_S // 60)


@account.route("/verify-email/resend", methods=["POST"])
def resend_code():
    state, masked = (None, None) if current_user.is_authenticated else _pending()
    if current_user.is_authenticated:
        if current_user.email_verified:
            return redirect(url_for("account.verify_email"))
        uid, masked = int(current_user.id), email_otp.mask_email(current_user.email)
    elif state is None:
        return redirect(url_for("routes.login"))
    else:
        uid = state["u"] or None
    now = time.time()
    try:
        if state is not None:
            email_otp.flow_check_send(state, now)          # same limits for real and decoy pending flows
        try:
            if uid is None:
                email_otp.decoy_send(state["e"])
            else:
                email_otp.issue(uid)
        except AssistanceError as exc:
            if state is not None and exc.code == "DELIVERY_FAILED":
                email_otp.flow_record_send(state, now, delivered=False)   # a failed delivery still counts
                _save(state, masked)
            raise
        if state is not None:
            email_otp.flow_record_send(state, now)
            _save(state, masked)
    except AssistanceError as exc:
        flash(exc.message, "warning" if exc.http_status in (429, 503) else "danger")
    else:
        flash(f"A new verification code was sent to {masked}.", "success")
    return redirect(url_for("account.verify_email"))


def _after_verified(user_id):
    """Email verified: drivers / Umusare now wait for cooperative verification; tell their manager."""
    from .cooperative_service import request_verification
    with transaction() as cursor:
        cursor.execute("SELECT u.username, u.role, m.cooperative_id FROM users u LEFT JOIN cooperative_memberships m "
                       "ON m.user_id = u.id WHERE u.id=%s", (user_id,))
        row = cursor.fetchone()
        if row and row["role"] in ("driver", "umusare") and row["cooperative_id"]:
            request_verification(cursor, user_id, row["cooperative_id"], row["role"], row["username"])


# ---------------------------------------------------------------- legal pages
@account.route("/terms")
def terms():
    return render_template("terms.html", version=legal.TERMS_VERSION, effective=legal.EFFECTIVE_DATE,
                           contact=legal.contact_email())


@account.route("/privacy")
def privacy():
    return render_template("privacy.html", version=legal.PRIVACY_VERSION, effective=legal.EFFECTIVE_DATE,
                           contact=legal.contact_email())


@account.route("/terms/accept", methods=["POST"])
@roles_required(*ROLES)
def accept_terms():
    if request.form.get("accept_terms") != "1":
        flash("Tick the box to confirm you have read and accept the Terms & Conditions and Privacy Policy.", "warning")
    else:
        with transaction() as cursor:
            legal.record_acceptance(cursor, int(current_user.id), utcnow())
        flash("Thank you. Your acceptance of the current Terms and Privacy Policy has been recorded.", "success")
    target = request.form.get("next") or ""
    return redirect(safe_local_path(target) or dashboard_url(current_user))


@account.app_context_processor
def inject_legal():
    """Lets every page show the email-verification and Terms re-acceptance banners."""
    needs_terms = current_user.is_authenticated and legal.needs_acceptance(current_user)
    return {"legal_versions": {"terms": legal.TERMS_VERSION, "privacy": legal.PRIVACY_VERSION},
            "needs_terms_acceptance": needs_terms}
