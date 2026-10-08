# website/driver_views.py
"""Driver-only endpoints: own vehicle information and privacy-safe nearby drivers.

The driver is always the signed-in user; no driver id is accepted from the browser.
"""
from flask import Blueprint, flash, jsonify, redirect, request, url_for
from flask_login import current_user

from . import ROLE_DRIVER
from . import badges, nearby_service, vehicle
from .assistance_service import AssistanceError, transaction
from .auth import roles_required
from .cooperative_service import own_verification

driver = Blueprint("driver", __name__, url_prefix="/driver")


@driver.errorhandler(AssistanceError)
def driver_error(err):
    return jsonify({"error": err.message, "code": err.code}), err.http_status


def _uid():
    return int(current_user.get_id())


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@driver.route("/vehicle", methods=["POST"])
@roles_required(ROLE_DRIVER)
def update_vehicle():
    try:
        v, reverify = vehicle.update_own_vehicle(_uid(), request.form)
    except AssistanceError as exc:
        flash(exc.message, "danger")
        return redirect(url_for("routes.driver_dashboard") + "#vehicle")
    if reverify:
        flash(f"Vehicle updated ({v['vehicle_plate_number']}). Because the plate changed, your cooperative manager "
              "must verify your account again.", "warning")
    else:
        flash(f"Vehicle information updated: {v['vehicle_plate_number']}.", "success")
    return redirect(url_for("routes.driver_dashboard") + "#vehicle")


@driver.route("/presence", methods=["POST"])
@roles_required(ROLE_DRIVER)
def presence():
    """Propose my current position; the server keeps only the ~1 km cell, and only if plausible."""
    b = _body()
    return jsonify(nearby_service.update_presence(_uid(), b.get("lat"), b.get("lon"), b.get("accuracy")))


@driver.route("/nearby", methods=["POST"])
@roles_required(ROLE_DRIVER)
def nearby():
    """Nearby verified drivers around my SERVER-KNOWN position. Any coordinates in the request are ignored."""
    return jsonify(nearby_service.nearby(_uid()))


@driver.route("/nearby/visibility", methods=["POST"])
@roles_required(ROLE_DRIVER)
def visibility():
    return jsonify({"visible": nearby_service.set_visibility(_uid(), bool(_body().get("visible")))})


def dashboard_context(user_id):
    """Everything the driver dashboard shows about the driver themself (all from the database)."""
    with transaction() as cursor:
        cursor.execute(
            "SELECT u.username, u.email, u.phone, u.email_verified_at, g.name AS group_name, g.status AS group_status, "
            "dp.vehicle_plate_number, dp.vehicle_make, dp.vehicle_model, dp.vehicle_type, dp.nearby_visibility "
            "FROM users u LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
            "LEFT JOIN cooperative_groups g ON g.id = m.group_id "
            "LEFT JOIN driver_profiles dp ON dp.user_id = u.id WHERE u.id=%s", (user_id,))
        me = cursor.fetchone() or {}
    initials = "".join(p[0] for p in (me.get("username") or "?").split()[:2]).upper()
    return {"me": {**me, "code": f"DRV-{int(user_id):05d}", "initials": initials,
                   "vehicle_type_label": vehicle.VEHICLE_TYPES.get(me.get("vehicle_type") or "")},
            "badge": badges.for_user(user_id), "vstate": own_verification(user_id),
            "vehicle_types": vehicle.VEHICLE_TYPES}
