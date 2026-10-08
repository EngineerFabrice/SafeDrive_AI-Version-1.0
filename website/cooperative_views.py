# website/cooperative_views.py
"""Manager member management and admin cooperative / verification pages.

Scope is enforced in cooperative_service on every call: a manager only ever reaches
their own cooperative's drivers and Umusare (others answer 404); admin is global.
"""
from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from . import ROLE_ADMIN, ROLE_MANAGER
from . import cooperative_service as svc
from .assistance_service import AssistanceError
from .auth import roles_required
from .chat_service import contact_token

coop = Blueprint("coop", __name__)


def _uid():
    return int(current_user.get_id())


def _back(default):
    target = request.form.get("next") or ""
    ok = target.startswith(("/manager", "/admin/")) and not target.startswith("//")
    return redirect(target if ok else default)


def _run(action, success, default):
    """Run a POST action. Out-of-scope / forbidden targets answer 404 / 403; other problems are flashed."""
    try:
        result = action()
        flash(success(result) if callable(success) else success, "success")
    except AssistanceError as exc:
        if exc.http_status in (403, 404):
            abort(exc.http_status)
        flash(exc.message, "danger")
    return _back(default)


# ---------------------------------------------------------------- manager (and admin) member pages
@coop.route("/manager/members/<int:user_id>")
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def member(user_id):
    try:
        m = svc.member_detail(_uid(), user_id)
    except AssistanceError as exc:
        abort(exc.http_status if exc.http_status in (403, 404) else 400)
    return render_template("manager-member.html", m=m, chat_token=contact_token(_uid(), m["id"]) if m["is_active"] else None)


@coop.route("/manager/members/<int:user_id>/review", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def review(user_id):
    action = request.form.get("action", "")
    labels = {svc.VERIFY: "Member verified.", svc.REJECT: "Verification rejected. The member has been informed.",
              svc.REQUEST_INFO: "More information requested from the member.", svc.SUSPEND: "Member suspended."}
    return _run(lambda: svc.review_member(_uid(), user_id, action, request.form.get("note")),
                labels.get(action, "Updated."), url_for("coop.member", user_id=user_id))


@coop.route("/manager/members/<int:user_id>/active", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def member_active(user_id):
    active = request.form.get("active") == "1"
    return _run(lambda: svc.set_member_active(_uid(), user_id, active),
                "Account activated." if active else "Account deactivated.", url_for("coop.member", user_id=user_id))


@coop.route("/manager/profile/phone", methods=["POST"])
@roles_required(ROLE_MANAGER)
def manager_phone():
    return _run(lambda: svc.set_own_phone(_uid(), request.form.get("phone")),
                lambda phone: f"Contact phone updated to {phone}.", url_for("routes.manager_dashboard"))


# ---------------------------------------------------------------- admin: cooperatives, managers, verification
@coop.route("/admin/cooperatives")
@roles_required(ROLE_ADMIN)
def cooperatives():
    coops, managers = svc.cooperatives_overview()
    return render_template("admin-cooperatives.html", coops=coops, managers=managers)


@coop.route("/admin/cooperatives/<int:coop_id>")
@roles_required(ROLE_ADMIN)
def cooperative(coop_id):
    data = svc.cooperative_detail(coop_id)
    if data is None:
        abort(404)
    token = contact_token(_uid(), data["manager"]["user_id"]) if data["manager"] else None
    return render_template("admin-cooperative-detail.html", d=data, manager_token=token)


@coop.route("/admin/cooperatives/<int:coop_id>/edit", methods=["POST"])
@roles_required(ROLE_ADMIN)
def cooperative_edit(coop_id):
    f = request.form
    return _run(lambda: svc.update_cooperative(_uid(), coop_id, f.get("name"), f.get("district"), f.get("status")),
                "Cooperative updated.", url_for("coop.cooperative", coop_id=coop_id))


@coop.route("/admin/cooperatives/<int:coop_id>/manager", methods=["POST"])
@roles_required(ROLE_ADMIN)
def cooperative_manager(coop_id):
    return _run(lambda: svc.assign_manager(_uid(), coop_id, request.form.get("manager_id")),
                lambda name: f"{name} is now the manager of this cooperative.",
                url_for("coop.cooperative", coop_id=coop_id))


@coop.route("/admin/managers/<int:user_id>/active", methods=["POST"])
@roles_required(ROLE_ADMIN)
def manager_active(user_id):
    active = request.form.get("active") == "1"
    return _run(lambda: svc.set_manager_active(_uid(), user_id, active),
                "Manager activated." if active else "Manager deactivated.", url_for("coop.cooperatives"))


@coop.route("/admin/managers/<int:user_id>/phone", methods=["POST"])
@roles_required(ROLE_ADMIN)
def manager_phone_admin(user_id):
    return _run(lambda: svc.set_manager_phone(_uid(), user_id, request.form.get("phone")),
                "Manager phone updated.", url_for("coop.cooperatives"))


@coop.route("/admin/verification")
@roles_required(ROLE_ADMIN)
def verification():
    status = request.args.get("status", "PENDING")
    coop_raw = request.args.get("cooperative", "")
    rows, coops, counts = svc.verification_queue(status, int(coop_raw) if coop_raw.isdigit() else None)
    return render_template("admin-verification.html", rows=rows, coops=coops, counts=counts,
                           status=status, cooperative=coop_raw, states=svc.STATES)


# ---------------------------------------------------------------- groups (manager: own cooperative; admin: all)
@coop.route("/manager/groups", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def group_create():
    from . import group_service as groups
    coop_id = request.form.get("cooperative_id")             # used only for admin; managers are always scoped
    default = url_for("coop.cooperative", coop_id=int(coop_id)) if current_user.is_admin() and str(coop_id).isdigit() \
        else url_for("routes.manager_dashboard") + "#groups"
    return _run(lambda: groups.create_group(_uid(), request.form.get("name"), coop_id),
                "Group created.", default)


@coop.route("/manager/groups/<int:group_id>")
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def group(group_id):
    from . import group_service as groups
    try:
        d = groups.group_detail(_uid(), group_id)
    except AssistanceError as exc:
        abort(exc.http_status if exc.http_status in (403, 404) else 400)
    return render_template("manager-group.html", d=d, g=d["group"])


@coop.route("/manager/groups/<int:group_id>/edit", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def group_edit(group_id):
    from . import group_service as groups
    return _run(lambda: groups.update_group(_uid(), group_id, request.form.get("name"), request.form.get("status")),
                "Group updated.", url_for("coop.group", group_id=group_id))


@coop.route("/manager/groups/<int:group_id>/members", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def group_add(group_id):
    from . import group_service as groups
    return _run(lambda: groups.add_member(_uid(), group_id, request.form.get("user_id")),
                "Member added to the group.", url_for("coop.group", group_id=group_id))


@coop.route("/manager/groups/<int:group_id>/members/<int:user_id>/remove", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def group_remove(group_id, user_id):
    from . import group_service as groups
    return _run(lambda: groups.remove_member(_uid(), group_id, user_id),
                "Member removed from the group.", url_for("coop.group", group_id=group_id))


@coop.route("/manager/members/<int:user_id>/group", methods=["POST"])
@roles_required(ROLE_MANAGER, ROLE_ADMIN)
def member_group(user_id):
    """Put a member into one of their cooperative's groups (scope checked by group_service)."""
    from . import group_service as groups
    return _run(lambda: groups.add_member(_uid(), request.form.get("group_id"), user_id),
                "Group updated.", url_for("coop.member", user_id=user_id))
