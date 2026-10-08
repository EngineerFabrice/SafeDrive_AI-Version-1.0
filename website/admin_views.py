# website/admin_views.py
"""Admin management pages: drivers, Umusare, assistance monitoring, pricing (all admin-only)."""
from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from . import ROLE_ADMIN
from . import admin_service as svc
from .assistance_service import AssistanceError, transaction
from .auth import roles_required
from .pricing import PricingError, format_money, get_pricing, update_pricing, validate

admin_mgmt = Blueprint("admin_mgmt", __name__, url_prefix="/admin")


def _uid():
    return int(current_user.get_id())


def _back(default):
    target = request.form.get("next") or ""
    return redirect(target if target.startswith("/admin/") else url_for(default))


def _run(action, success, default):
    try:
        action()
        flash(success, "success")
    except AssistanceError as exc:
        flash(exc.message, "danger")
    return _back(default)


@admin_mgmt.route("/drivers")
@roles_required(ROLE_ADMIN)
def drivers():
    return render_template("admin-drivers.html", drivers=svc.drivers_overview())


@admin_mgmt.route("/drivers/<int:user_id>")
@roles_required(ROLE_ADMIN)
def driver_detail(user_id):
    driver, history = svc.driver_detail(user_id)
    if driver is None:
        abort(404)
    return render_template("admin-driver-detail.html", driver=driver, history=history)


@admin_mgmt.route("/umusare")
@roles_required(ROLE_ADMIN)
def umusare():
    return render_template("admin-umusare.html", umusare=svc.umusare_overview(), states=svc.VERIFICATION_STATES)


@admin_mgmt.route("/assistance")
@roles_required(ROLE_ADMIN)
def assistance():
    filters = {k: request.args.get(k, "") for k in ("status", "type", "payment", "cooperative")}
    rows, cooperatives = svc.assistance_monitor(filters)
    return render_template("admin-assistance.html", rows=rows, cooperatives=cooperatives, filters=filters)


@admin_mgmt.route("/pricing", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN)
def pricing():
    if request.method == "POST":
        try:
            new = validate(request.form)
            with transaction() as cursor:
                update_pricing(cursor, _uid(), new)
            flash(f"Pricing updated: {format_money(new['price_per_km'], new['currency'])} per km. "
                  "Completed journeys keep the price they were charged.", "success")
        except PricingError as exc:
            flash(str(exc), "danger")
        return redirect(url_for("admin_mgmt.pricing"))
    with transaction() as cursor:
        current = get_pricing(cursor)
    return render_template("admin-pricing.html", pricing=current,
                           example=format_money(current["base_fee"] + current["price_per_km"] * 46 / 10,
                                                current["currency"]))


# ---------------------------------------------------------------- actions (POST + CSRF, audited)
@admin_mgmt.route("/users/<int:user_id>/active", methods=["POST"])
@roles_required(ROLE_ADMIN)
def set_active(user_id):
    active = request.form.get("active") == "1"
    return _run(lambda: svc.set_account_active(_uid(), user_id, active),
                "Account activated." if active else "Account deactivated.", "admin_mgmt.drivers")


@admin_mgmt.route("/umusare/<int:user_id>/verification", methods=["POST"])
@roles_required(ROLE_ADMIN)
def set_verification(user_id):
    status = request.form.get("status", "")
    return _run(lambda: svc.set_umusare_verification(_uid(), user_id, status),
                f"Verification set to {status}.", "admin_mgmt.umusare")


@admin_mgmt.route("/users/<int:user_id>/phone", methods=["POST"])
@roles_required(ROLE_ADMIN)
def set_phone(user_id):
    return _run(lambda: svc.set_phone(_uid(), user_id, request.form.get("phone")),
                "Phone number updated.", "admin_mgmt.drivers")
