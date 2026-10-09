# website/assistance.py
"""HTTP API for the Driver -> Umusare assistance workflow (logic lives in assistance_service).

Every endpoint requires login and the right role; ownership is checked by the
service on every call. POSTs are CSRF-protected (X-CSRFToken header).
"""
import os

from flask import Blueprint, jsonify, request
from flask_login import current_user

from . import ROLE_DRIVER, ROLE_UMUSARE, audit
from . import assistance_service as svc
from .auth import normalize_phone, roles_required

assistance = Blueprint("assistance", __name__, url_prefix="/assistance")


def maps_config():
    """Browser-side map settings. The key is optional; without it the UI uses plain Google Maps links."""
    from .config import map_tiles
    return {"google_maps_api_key": os.environ.get("GOOGLE_MAPS_API_KEY", "").strip(), "tiles": map_tiles()}


@assistance.app_context_processor
def inject_maps():
    return {"maps": maps_config()}


@assistance.errorhandler(svc.AssistanceError)
def assistance_error(err):
    return jsonify({"error": err.message, "code": err.code}), err.http_status


def _uid():
    return int(current_user.get_id())


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _verified_trigger(claimed):
    return verified_trigger(_uid(), claimed)


def verified_trigger(user_id, claimed):
    """AI_TRIGGERED only if this driver's own monitoring really is POTENTIALLY_NOT_SOBER right now.

    Checked against the driver's live (server camera) session, or their phone-camera session from the
    mobile app when its latest assessment is recent. The client's claim is never trusted on its own;
    anything else is a manual (DRIVER_INITIATED) request. "SAFETY_ALERT" is an older name for the claim.
    """
    if claimed not in (svc.AI_TRIGGERED, "SAFETY_ALERT"):
        return svc.DRIVER_INITIATED
    try:
        from . import monitoring
        engine, recorder = monitoring._engine, monitoring._recorder
        if engine is not None and recorder is not None and recorder.active_owner() == user_id:
            a = engine.snapshot().assessment
            if a and a.get("assessment") == "POTENTIALLY_NOT_SOBER":
                return svc.AI_TRIGGERED
    except Exception:
        pass
    try:
        from . import mobile_monitoring
        if mobile_monitoring.fresh_assessment(user_id) == "POTENTIALLY_NOT_SOBER":
            return svc.AI_TRIGGERED
    except Exception:
        pass
    return svc.DRIVER_INITIATED


# ---------------------------------------------------------------- driver
@assistance.route("/requests", methods=["POST"])
@roles_required(ROLE_DRIVER)
def create():
    b = _body()
    request_id = svc.create_request(_uid(), b.get("lat"), b.get("lon"), b.get("accuracy"),
                                    _verified_trigger(b.get("trigger")))
    return jsonify(svc.driver_view(_uid(), request_id)), 201


@assistance.route("/requests/current")
@roles_required(ROLE_DRIVER)
def current():
    return jsonify({"request": svc.driver_view(_uid())})


@assistance.route("/requests/<int:request_id>/cancel", methods=["POST"])
@roles_required(ROLE_DRIVER)
def cancel(request_id):
    svc.cancel_request(_uid(), request_id)
    return jsonify(svc.driver_view(_uid(), request_id))


# ---------------------------------------------------------------- Umusare
@assistance.route("/umusare/status")
@roles_required(ROLE_UMUSARE)
def umusare_status():
    return jsonify(svc.umusare_view(_uid()))


@assistance.route("/umusare/availability", methods=["POST"])
@roles_required(ROLE_UMUSARE)
def availability():
    b = _body()
    state = svc.set_availability(_uid(), bool(b.get("available")), b.get("lat"), b.get("lon"), b.get("accuracy"))
    return jsonify({"availability": state})


@assistance.route("/requests/<int:request_id>/accept", methods=["POST"])
@roles_required(ROLE_UMUSARE)
def accept(request_id):
    svc.accept_request(_uid(), request_id)
    return jsonify(svc.umusare_view(_uid()))


@assistance.route("/requests/<int:request_id>/decline", methods=["POST"])
@roles_required(ROLE_UMUSARE)
def decline(request_id):
    svc.decline_request(_uid(), request_id)
    return jsonify(svc.umusare_view(_uid()))


# ---------------------------------------------------------------- both participants
@assistance.route("/requests/<int:request_id>/connect", methods=["POST"])
@roles_required(ROLE_DRIVER, ROLE_UMUSARE)
def connect(request_id):
    svc.mark_connected(_uid(), request_id)
    return jsonify({"status": svc.DRIVER_CONNECTED})


@assistance.route("/requests/<int:request_id>/complete", methods=["POST"])
@roles_required(ROLE_DRIVER, ROLE_UMUSARE)
def complete(request_id):
    """COMPLETE JOURNEY. Any distance/fare in the body is ignored: the server calculates both."""
    result = svc.complete_request(_uid(), request_id)
    return jsonify({"status": svc.COMPLETED, "payment_status": svc.P_PENDING, **result})


# ---------------------------------------------------------------- payment (no amounts accepted from the browser)
@assistance.route("/requests/<int:request_id>/payment-sent", methods=["POST"])
@roles_required(ROLE_DRIVER)
def payment_sent(request_id):
    svc.mark_payment_sent(_uid(), request_id)
    return jsonify(svc.driver_view(_uid(), request_id))


@assistance.route("/requests/<int:request_id>/payment-received", methods=["POST"])
@roles_required(ROLE_UMUSARE)
def payment_received(request_id):
    svc.confirm_payment_received(_uid(), request_id)
    return jsonify(svc.umusare_view(_uid()))


@assistance.route("/requests/<int:request_id>/payment-problem", methods=["POST"])
@roles_required(ROLE_UMUSARE)
def payment_problem(request_id):
    svc.report_payment_problem(_uid(), request_id, _body().get("note"))
    return jsonify(svc.umusare_view(_uid()))


@assistance.route("/requests/<int:request_id>/rate", methods=["POST"])
@roles_required(ROLE_DRIVER)
def rate(request_id):
    b = _body()
    svc.rate_assistance(_uid(), request_id, b.get("rating"), b.get("comment"))
    return jsonify(svc.driver_view(_uid(), request_id))


@assistance.route("/requests/<int:request_id>/report", methods=["POST"])
@roles_required(ROLE_DRIVER)
def report(request_id):
    svc.report_problem(_uid(), request_id, _body().get("text"))
    return jsonify(svc.driver_view(_uid(), request_id))


# ---------------------------------------------------------------- own contact number
@assistance.route("/profile/phone", methods=["POST"])
@roles_required(ROLE_DRIVER, ROLE_UMUSARE)
def my_phone():
    """A user may only change their own number. Completed journeys keep the number stored at completion."""
    try:
        phone = normalize_phone(_body().get("phone"))
    except ValueError as exc:
        raise svc.AssistanceError(400, "INVALID_PHONE", str(exc))
    with svc.transaction() as cursor:
        cursor.execute("UPDATE users SET phone=%s WHERE id=%s", (phone, _uid()))
        audit.record(audit.PHONE_UPDATED, actor_id=_uid(), target_type="user", target_id=_uid(), cursor=cursor)
    return jsonify({"phone": phone})


@assistance.route("/requests/<int:request_id>/location", methods=["POST"])
@roles_required(ROLE_DRIVER, ROLE_UMUSARE)
def location(request_id):
    b = _body()
    svc.update_live_location(_uid(), request_id, b.get("lat"), b.get("lon"), b.get("accuracy"))
    return jsonify({"ok": True})
