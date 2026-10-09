# website/mobile_api.py
"""JSON API for the SafeDrive AI mobile app (Flutter, see mobile/README.md).

    /api/mobile/v1/...   Authorization: Bearer <token>   (from POST /auth/login or /auth/verify-email)

Every endpoint delegates to the same service functions as the web app (assistance_service,
email_otp, cooperative_service, badges, monitoring), so roles, ownership, cooperative scoping and
privacy rules are enforced in one place. The acting user always comes from the token, never from
the request body. The browser session cookie is never accepted here, which is why this blueprint
is exempt from CSRF (see mobile_auth).
"""
import time
from datetime import date, datetime
from decimal import Decimal

from flask import Blueprint, g, jsonify, request

from . import ROLE_ADMIN, ROLE_DRIVER, ROLE_MANAGER, ROLE_UMUSARE, ROLES, get_user_by_id
from . import assistance_service as svc
from . import audit, legal, mobile_auth, mobile_monitoring
from .assistance import verified_trigger
from .assistance_service import AssistanceError, transaction, utcnow
from .auth import normalize_phone
from .mobile_auth import token_required

API_VERSION = 1
mobile_api = Blueprint("mobile_api", __name__, url_prefix="/api/mobile/v1")

MONITORING_ROLES = (ROLE_DRIVER, ROLE_ADMIN)
CREDIT = {"author": "Fabrice NDAYISABA", "contact": "fabricendayisaba16@gmail.com"}


# ---------------------------------------------------------------- helpers
def _clean(value):
    """JSON-safe copy: datetimes as ISO-8601 UTC strings, Decimals as numbers, tuples as lists."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds") + ("Z" if value.tzinfo is None else "")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return None
    return value


def ok(payload=None, status=200):
    return jsonify(_clean(payload if payload is not None else {})), status


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _str(data, key, limit=255):
    value = data.get(key)
    return value.strip()[:limit] if isinstance(value, str) else ""


def _uid():
    return int(g.mobile_user.id)


@mobile_api.errorhandler(AssistanceError)
def _service_error(err):
    return jsonify({"error": err.message, "code": err.code}), err.http_status


@mobile_api.errorhandler(mobile_monitoring.FrameRejected)
def _frame_error(err):
    return jsonify({"error": err.message, "code": err.code}), err.http_status


@mobile_api.errorhandler(413)
def _too_large(err):
    return jsonify({"error": "The request is too large.", "code": "TOO_LARGE"}), 413


@mobile_api.after_request
def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def _user_payload(user):
    """Everything the app needs to choose and render the signed-in user's role dashboard."""
    from .badges import for_user
    from .cooperative_service import own_verification
    uid = int(user.id)
    with transaction() as cursor:
        cursor.execute(
            "SELECT u.phone, m.member_role, m.status AS membership_status, c.id AS cooperative_id, "
            "c.name AS cooperative_name, c.code AS cooperative_code, g.name AS group_name "
            "FROM users u LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
            "LEFT JOIN cooperatives c ON c.id = m.cooperative_id "
            "LEFT JOIN cooperative_groups g ON g.id = m.group_id WHERE u.id=%s", (uid,))
        row = cursor.fetchone() or {}
        driver = umusare = None
        if user.role == ROLE_DRIVER:
            cursor.execute("SELECT vehicle_plate_number, vehicle_make, vehicle_model, vehicle_type, "
                           "verification_status FROM driver_profiles WHERE user_id=%s", (uid,))
            driver = cursor.fetchone()
        elif user.role == ROLE_UMUSARE:
            cursor.execute("SELECT verification_status, availability FROM umusare_profiles WHERE user_id=%s", (uid,))
            umusare = cursor.fetchone()
    membership = None
    if row.get("cooperative_id"):
        membership = {k: row.get(k) for k in ("member_role", "membership_status", "cooperative_id",
                                               "cooperative_name", "cooperative_code", "group_name")}
    return {"id": uid, "username": user.username, "email": user.email, "role": user.role, "phone": row.get("phone"),
            "email_verified": user.email_verified, "needs_terms_acceptance": legal.needs_acceptance(user),
            "membership": membership, "badge": for_user(uid),
            "verification": own_verification(uid) if user.role in (ROLE_DRIVER, ROLE_UMUSARE) else None,
            "driver_profile": driver, "umusare_profile": umusare}


def _signed_in(user, status=200):
    token, expires = mobile_auth.issue_token(int(user.id), _str(_body(), "device_name", 80))
    return ok({"token": token, "token_type": "Bearer", "expires_at": expires, "user": _user_payload(user)}, status)


# ---------------------------------------------------------------- public
@mobile_api.route("/meta")
def meta():
    """Public facts the app shows before sign-in. No configuration values or secrets."""
    from .monitoring import SYSTEM_NOTICE
    return ok({"api_version": API_VERSION, "app": "SafeDrive AI", "roles": list(ROLES),
               "assessment_labels": ["SOBER", "UNCERTAIN", "POTENTIALLY_NOT_SOBER"],
               "system_notice": SYSTEM_NOTICE, "terms_version": legal.TERMS_VERSION,
               "privacy_version": legal.PRIVACY_VERSION, "contact": legal.contact_email(), "credit": CREDIT})


@mobile_api.route("/registration-options")
def registration_options():
    """Approved cooperatives, their active groups and vehicle types (the same lists as the web form)."""
    from .routes import registration_options as options
    from .vehicle import VEHICLE_TYPES
    cooperatives, groups = options()
    return ok({"cooperatives": [{k: c[k] for k in ("id", "name", "code", "district")} for c in cooperatives],
               "groups": [{"id": gr["id"], "name": gr["name"], "cooperative_id": gr["cooperative_id"]} for gr in groups],
               "vehicle_types": [{"value": k, "label": v} for k, v in VEHICLE_TYPES.items()],
               "roles": ["driver", "umusare"]})


@mobile_api.route("/auth/login", methods=["POST"])
def login():
    from .routes import check_credentials
    b = _body()
    user = check_credentials(b.get("email"), b.get("password"))
    if user is None:
        return jsonify({"error": "Incorrect email or password.", "code": "INVALID_CREDENTIALS"}), 401
    return _signed_in(user)


@mobile_api.route("/auth/register", methods=["POST"])
def register():
    """Same validation, rate limit and duplicate-email protection as the web form.

    Returns an opaque, encrypted ``pending_token`` (the web keeps the same value in its session cookie);
    a duplicate email gets a response of the same shape, and its token can never verify.
    """
    from . import email_otp
    from .routes import REGISTRATION_KEYS, register_account, registration_options as options
    b = _body()
    form = {k: _str(b, k) for k in REGISTRATION_KEYS}
    form["cooperative_id"] = str(b.get("cooperative_id") or "").strip()[:20]
    form["group_id"] = str(b.get("group_id") or "").strip()[:20]
    cooperatives, groups = options()
    errors, result = register_account(form, b.get("password") if isinstance(b.get("password"), str) else "",
                                      b.get("confirm_password") if isinstance(b.get("confirm_password"), str) else "",
                                      b.get("accept_terms") is True, cooperatives, groups)
    if errors:
        return jsonify({"error": errors[0], "errors": errors, "code": "VALIDATION"}), 400
    state = result["state"]
    return ok({"pending_token": email_otp.flow_dump(state), "masked_email": result["masked"],
               "message": result["warning"] or result["message"], "delivered": result["warning"] is None,
               "resend_wait_s": email_otp.flow_wait(state, time.time()), "code_ttl_min": email_otp.OTP_TTL_S // 60},
              201)


def _optional_user():
    token = mobile_auth._bearer()
    return mobile_auth.resolve(token)[0] if token else None


@mobile_api.route("/auth/verify-email", methods=["POST"])
def verify_email():
    """Verify with the emailed code: either a pending registration (``pending_token``) or the signed-in user.

    A successful pending verification signs the new account in (returns a token), like the web flow.
    """
    from . import email_otp
    from .account import _after_verified
    b = _body()
    user = _optional_user()
    if user is not None:
        email_otp.verify(int(user.id), b.get("code"))
        _after_verified(int(user.id))
        return ok({"verified": True, "user": _user_payload(get_user_by_id(user.id))})
    state = email_otp.flow_load(b.get("pending_token")) if isinstance(b.get("pending_token"), str) else None
    if state is None:
        return jsonify({"error": "This verification has expired. Please sign in to request a new code.",
                        "code": "PENDING_EXPIRED"}), 400
    uid = state["u"] or None
    try:
        if uid is None:
            email_otp.decoy_verify(state)
        email_otp.verify(uid, b.get("code"))
    except AssistanceError as exc:
        if uid is not None:
            state["a"] += 1                         # mirror the server count in both flows
        return jsonify({"error": exc.message, "code": exc.code,
                        "pending_token": email_otp.flow_dump(state)}), exc.http_status
    _after_verified(uid)
    verified = get_user_by_id(uid)
    if verified is None or not verified.is_active:
        return ok({"verified": True})
    audit.record(audit.LOGIN_SUCCEEDED, actor_id=uid, target_type="user", target_id=uid)
    return _signed_in(verified)


@mobile_api.route("/auth/resend-code", methods=["POST"])
def resend_code():
    from . import email_otp
    b = _body()
    user = _optional_user()
    if user is not None:
        if user.email_verified:
            return jsonify({"error": "Your email address is already verified.", "code": "ALREADY_VERIFIED"}), 409
        email_otp.issue(int(user.id))
        return ok({"message": f"A new verification code was sent to {email_otp.mask_email(user.email)}."})
    state = email_otp.flow_load(b.get("pending_token")) if isinstance(b.get("pending_token"), str) else None
    if state is None:
        return jsonify({"error": "This verification has expired. Please sign in to request a new code.",
                        "code": "PENDING_EXPIRED"}), 400
    uid, now = state["u"] or None, time.time()
    try:
        email_otp.flow_check_send(state, now)            # same limits for real and decoy pending flows
        try:
            if uid is None:
                email_otp.decoy_send(state["e"])
            else:
                email_otp.issue(uid)
        except AssistanceError as exc:
            if exc.code == "DELIVERY_FAILED":
                email_otp.flow_record_send(state, now, delivered=False)
            raise
        email_otp.flow_record_send(state, now)
    except AssistanceError as exc:
        return jsonify({"error": exc.message, "code": exc.code,
                        "pending_token": email_otp.flow_dump(state)}), exc.http_status
    return ok({"message": f"A new verification code was sent to {email_otp.mask_email(state['e'])}.",
               "pending_token": email_otp.flow_dump(state), "resend_wait_s": email_otp.flow_wait(state, time.time())})


# ---------------------------------------------------------------- signed in
@mobile_api.route("/auth/logout", methods=["POST"])
@token_required()
def logout():
    mobile_auth.revoke(g.mobile_token_id)
    audit.record(audit.LOGOUT, actor_id=_uid(), target_type="user", target_id=_uid())
    return ok({"signed_out": True})


@mobile_api.route("/me")
@token_required()
def me():
    return ok({"user": _user_payload(g.mobile_user)})


@mobile_api.route("/me/accept-terms", methods=["POST"])
@token_required()
def accept_terms():
    if _body().get("accept_terms") is not True:
        return jsonify({"error": "Confirm that you have read and accept the Terms & Conditions and Privacy Policy.",
                        "code": "VALIDATION"}), 400
    with transaction() as cursor:
        legal.record_acceptance(cursor, _uid(), utcnow())
    return ok({"user": _user_payload(get_user_by_id(_uid()))})


@mobile_api.route("/me/phone", methods=["POST"])
@token_required(ROLE_DRIVER, ROLE_UMUSARE)
def my_phone():
    try:
        phone = normalize_phone(_body().get("phone") if isinstance(_body().get("phone"), str) else "")
    except ValueError as exc:
        raise AssistanceError(400, "INVALID_PHONE", str(exc))
    with transaction() as cursor:
        cursor.execute("UPDATE users SET phone=%s WHERE id=%s", (phone, _uid()))
        audit.record(audit.PHONE_UPDATED, actor_id=_uid(), target_type="user", target_id=_uid(), cursor=cursor)
    return ok({"phone": phone})


@mobile_api.route("/notifications")
@token_required()
def notifications():
    from . import notifications as notes
    return ok({"notifications": notes.recent(_uid())})


@mobile_api.route("/notifications/read", methods=["POST"])
@token_required()
def notifications_read():
    from . import notifications as notes
    notes.mark_all_read(_uid())
    return ok({"ok": True})


# ---------------------------------------------------------------- assistance: driver
@mobile_api.route("/assistance/requests", methods=["POST"])
@token_required(ROLE_DRIVER)
def create_request():
    """The location is shared only here, when the driver explicitly asks for help."""
    b = _body()
    request_id = svc.create_request(_uid(), b.get("lat"), b.get("lon"), b.get("accuracy"),
                                    verified_trigger(_uid(), b.get("trigger")))
    return ok({"request": svc.driver_view(_uid(), request_id)}, 201)


@mobile_api.route("/assistance/requests/current")
@token_required(ROLE_DRIVER)
def current_request():
    return ok({"request": svc.driver_view(_uid())})


@mobile_api.route("/assistance/requests/<int:request_id>/cancel", methods=["POST"])
@token_required(ROLE_DRIVER)
def cancel(request_id):
    svc.cancel_request(_uid(), request_id)
    return ok({"request": svc.driver_view(_uid(), request_id)})


@mobile_api.route("/assistance/requests/<int:request_id>/payment-sent", methods=["POST"])
@token_required(ROLE_DRIVER)
def payment_sent(request_id):
    svc.mark_payment_sent(_uid(), request_id)
    return ok({"request": svc.driver_view(_uid(), request_id)})


@mobile_api.route("/assistance/requests/<int:request_id>/rate", methods=["POST"])
@token_required(ROLE_DRIVER)
def rate(request_id):
    b = _body()
    svc.rate_assistance(_uid(), request_id, b.get("rating"), b.get("comment"))
    return ok({"request": svc.driver_view(_uid(), request_id)})


@mobile_api.route("/assistance/requests/<int:request_id>/report", methods=["POST"])
@token_required(ROLE_DRIVER)
def report(request_id):
    svc.report_problem(_uid(), request_id, _body().get("text"))
    return ok({"request": svc.driver_view(_uid(), request_id)})


# ---------------------------------------------------------------- assistance: Umusare
@mobile_api.route("/assistance/umusare/status")
@token_required(ROLE_UMUSARE)
def umusare_status():
    return ok(svc.umusare_view(_uid()))


@mobile_api.route("/assistance/umusare/availability", methods=["POST"])
@token_required(ROLE_UMUSARE)
def availability():
    """Going AVAILABLE shares the current position for matching; going OFFLINE deletes it."""
    b = _body()
    state = svc.set_availability(_uid(), b.get("available") is True, b.get("lat"), b.get("lon"), b.get("accuracy"))
    return ok({"availability": state, "status": svc.umusare_view(_uid())})


@mobile_api.route("/assistance/requests/<int:request_id>/accept", methods=["POST"])
@token_required(ROLE_UMUSARE)
def accept(request_id):
    svc.accept_request(_uid(), request_id)
    return ok(svc.umusare_view(_uid()))


@mobile_api.route("/assistance/requests/<int:request_id>/decline", methods=["POST"])
@token_required(ROLE_UMUSARE)
def decline(request_id):
    svc.decline_request(_uid(), request_id)
    return ok(svc.umusare_view(_uid()))


@mobile_api.route("/assistance/requests/<int:request_id>/payment-received", methods=["POST"])
@token_required(ROLE_UMUSARE)
def payment_received(request_id):
    svc.confirm_payment_received(_uid(), request_id)
    return ok(svc.umusare_view(_uid()))


@mobile_api.route("/assistance/requests/<int:request_id>/payment-problem", methods=["POST"])
@token_required(ROLE_UMUSARE)
def payment_problem(request_id):
    svc.report_payment_problem(_uid(), request_id, _body().get("note"))
    return ok(svc.umusare_view(_uid()))


# ---------------------------------------------------------------- assistance: both participants
def _participant_view():
    return {"request": svc.driver_view(_uid())} if g.mobile_user.role == ROLE_DRIVER else svc.umusare_view(_uid())


@mobile_api.route("/assistance/requests/<int:request_id>/connect", methods=["POST"])
@token_required(ROLE_DRIVER, ROLE_UMUSARE)
def connect(request_id):
    svc.mark_connected(_uid(), request_id)
    return ok(_participant_view())


@mobile_api.route("/assistance/requests/<int:request_id>/complete", methods=["POST"])
@token_required(ROLE_DRIVER, ROLE_UMUSARE)
def complete(request_id):
    """Distance and fare are calculated by the server; anything in the body is ignored."""
    result = svc.complete_request(_uid(), request_id)
    return ok({"result": result, **_participant_view()})


@mobile_api.route("/assistance/requests/<int:request_id>/location", methods=["POST"])
@token_required(ROLE_DRIVER, ROLE_UMUSARE)
def live_location(request_id):
    """Live position during an accepted assistance only (the service refuses it in any other state)."""
    b = _body()
    svc.update_live_location(_uid(), request_id, b.get("lat"), b.get("lon"), b.get("accuracy"))
    return ok({"ok": True})


# ---------------------------------------------------------------- monitoring
@mobile_api.route("/monitoring/model")
@token_required(*MONITORING_ROLES)
def monitoring_model():
    from .monitoring import SYSTEM_NOTICE
    return ok({**mobile_monitoring.model_info(), "notice": SYSTEM_NOTICE})


@mobile_api.route("/monitoring/phone/start", methods=["POST"])
@token_required(*MONITORING_ROLES)
def phone_start():
    session = mobile_monitoring.start(_uid())
    audit.record(audit.MONITORING_STARTED, actor_id=_uid(), target_type="monitoring_session",
                 details={"source": "phone"})
    return ok(session)


@mobile_api.route("/monitoring/phone/stop", methods=["POST"])
@token_required(*MONITORING_ROLES)
def phone_stop():
    if mobile_monitoring.stop(_uid()):
        audit.record(audit.MONITORING_STOPPED, actor_id=_uid(), target_type="monitoring_session",
                     details={"source": "phone"})
    return ok({"active": False})


@mobile_api.route("/monitoring/phone/status")
@token_required(*MONITORING_ROLES)
def phone_status():
    return ok(mobile_monitoring.status(_uid()))


@mobile_api.route("/monitoring/phone/frame", methods=["POST"])
@token_required(*MONITORING_ROLES)
def phone_frame():
    """One JPEG frame from the phone camera: multipart field ``frame`` or a raw image/jpeg body."""
    upload = request.files.get("frame")
    data = upload.read(mobile_monitoring.MAX_FRAME_BYTES + 1) if upload else request.get_data(cache=False)
    return ok(mobile_monitoring.analyze(_uid(), data))


@mobile_api.route("/monitoring/vehicle/status")
@token_required(*MONITORING_ROLES)
def vehicle_status():
    """The server-attached (in-vehicle) camera engine used by the web dashboard. Only its owner sees results."""
    from . import monitoring
    engine = monitoring.get_engine()
    owner = monitoring.get_recorder().active_owner()
    running = engine.is_running
    mine = running and (owner == _uid() or g.mobile_user.role == ROLE_ADMIN)
    snap = engine.snapshot().to_dict() if mine else {}
    return ok({"running": running, "owned_by_me": running and owner == _uid(),
               "status": snap.get("status"), "camera_status": snap.get("camera_status"),
               "driver_detected": snap.get("driver_detected"), "face_detected": snap.get("face_detected"),
               "fps": snap.get("fps"), "assessment": snap.get("assessment"), "impairment": snap.get("impairment"),
               "face_quality": snap.get("face_quality"), "message": snap.get("message")})


@mobile_api.route("/monitoring/vehicle/start", methods=["POST"])
@token_required(*MONITORING_ROLES)
def vehicle_start():
    from . import monitoring
    body, status = monitoring.start_for(_uid())
    return ok(body, status)


@mobile_api.route("/monitoring/vehicle/stop", methods=["POST"])
@token_required(*MONITORING_ROLES)
def vehicle_stop():
    from . import monitoring
    body, status = monitoring.stop_for(_uid(), g.mobile_user.role == ROLE_ADMIN)
    return ok(body, status)


@mobile_api.route("/monitoring/history")
@token_required(*MONITORING_ROLES)
def monitoring_history():
    """Own server-camera sessions (technical statistics only); administrators see all."""
    from . import monitoring
    from .history import HistoryUnavailable, session_to_dict
    limit = max(1, min(request.args.get("limit", 20, type=int), 100))
    recorder = monitoring.get_recorder()
    owner = None if g.mobile_user.role == ROLE_ADMIN else _uid()
    try:
        sessions = [session_to_dict(r) for r in recorder.store.list_sessions(limit, started_by=owner)]
    except HistoryUnavailable:
        return jsonify({"sessions": [], "error": "Monitoring history is temporarily unavailable."}), 503
    return ok({"sessions": sessions})


# ---------------------------------------------------------------- manager / admin
@mobile_api.route("/manager/console")
@token_required(ROLE_MANAGER)
def manager_console():
    """The manager's own cooperative only (scope comes from the database, never from the request)."""
    from .cooperative_service import manager_console as console
    data = console(_uid())
    if data is None:
        return ok({"console": None, "message": "You are not assigned to a cooperative yet."})
    for key in ("drivers", "umusare"):
        data[key] = [{k: m.get(k) for k in ("id", "code", "username", "email", "phone", "role", "is_active",
                                            "verification_status", "verification_note", "info_requested",
                                            "email_verified", "ready", "operational", "group_name",
                                            "vehicle_plate_number", "membership_status", "availability",
                                            "created_at")} for m in data[key]]
    return ok({"console": data})


@mobile_api.route("/manager/members/<int:member_id>/review", methods=["POST"])
@token_required(ROLE_MANAGER, ROLE_ADMIN)
def review_member(member_id):
    """VERIFY / REJECT / REQUEST_INFO / SUSPEND; the service checks the member is in the actor's scope."""
    from .cooperative_service import review_member as review
    b = _body()
    status = review(_uid(), member_id, _str(b, "action", 20).upper(), b.get("note") if isinstance(b.get("note"), str)
                    else None)
    return ok({"verification_status": status})


@mobile_api.route("/admin/overview")
@token_required(ROLE_ADMIN)
def admin_overview():
    """Read-only platform overview; user and cooperative management stays in the web console."""
    with transaction() as cursor:
        cursor.execute("SELECT role, COUNT(*) AS n FROM users GROUP BY role")
        counts = {r["role"]: r["n"] for r in cursor.fetchall()}
        cursor.execute("SELECT status, COUNT(*) AS n FROM cooperatives GROUP BY status")
        coops = {r["status"]: r["n"] for r in cursor.fetchall()}
    return ok({"users": {role: counts.get(role, 0) for role in ROLES}, "cooperatives": coops,
               "assistance": svc.admin_overview(limit=25)})
