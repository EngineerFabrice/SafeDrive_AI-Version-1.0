# website/cooperative_service.py
"""Cooperative management: manager scope, member verification, admin cooperative actions.

Scope rules (always derived from the database, never from the request):

* Admin   - global: every cooperative and every driver / Umusare.
* Manager - only the cooperative of their own APPROVED 'manager' membership. A member of
            another cooperative is reported as "not found" (404), so existence is not revealed.
* Driver / Umusare - only themselves and their own cooperative's manager contact.

Verification (driver_profiles / umusare_profiles.verification_status):

    PENDING  -> VERIFIED | REJECTED | SUSPENDED   (REQUEST_INFO keeps PENDING and records a note)
    VERIFIED -> REJECTED | SUSPENDED
    REJECTED -> VERIFIED | PENDING (REQUEST_INFO)
    SUSPENDED -> VERIFIED | REJECTED

Verification never touches an AI sobriety assessment, the matching algorithm, fares or
payments. An Umusare who stops being VERIFIED is taken offline (refused while assisting).
"""
import re

from . import ROLE_ADMIN, ROLE_DRIVER, ROLE_MANAGER, ROLE_UMUSARE
from . import audit
from . import notifications as notes
from .admin_service import _take_offline, driver_code, monitoring_status
from .assistance_service import ACTIVE, AssistanceError, assistance_code, transaction, umusare_code, utcnow
from .auth import normalize_phone

STATES = ("PENDING", "VERIFIED", "REJECTED", "SUSPENDED")
VERIFY, REJECT, REQUEST_INFO, SUSPEND = "VERIFY", "REJECT", "REQUEST_INFO", "SUSPEND"
ACTIONS = (VERIFY, REJECT, REQUEST_INFO, SUSPEND)
MANAGED_ROLES = (ROLE_DRIVER, ROLE_UMUSARE)
PROFILE_TABLE = {ROLE_DRIVER: "driver_profiles", ROLE_UMUSARE: "umusare_profiles"}
NOTE_MAX = 255
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
ACTION_LABELS = {
    "USER_REGISTERED": "Registered", "USER_VERIFICATION_REQUESTED": "Verification requested",
    "USER_VERIFIED": "Member verified", "USER_VERIFICATION_REJECTED": "Verification rejected",
    "USER_VERIFICATION_INFO_REQUESTED": "More information requested", "USER_SUSPENDED": "Member suspended",
    "ACCOUNT_STATUS_CHANGED": "Account status changed", "ASSISTANCE_REQUESTED": "Assistance requested",
    "ASSISTANCE_COMPLETED": "Assistance completed", "ASSISTANCE_CANCELLED": "Assistance cancelled",
    "ASSISTANCE_ACCEPTED": "Assistance accepted", "MANAGER_ASSIGNED": "Manager assigned",
    "MANAGER_CHANGED": "Manager changed", "COOPERATIVE_UPDATED": "Cooperative updated",
    "COOPERATIVE_CREATED": "Cooperative created", "ROLE_CHANGED": "Role changed",
    "UMUSARE_VERIFICATION_CHANGED": "Umusare verification changed", "PHONE_UPDATED": "Phone updated",
}
DECISION = {VERIFY: "VERIFIED", REJECT: "REJECTED", REQUEST_INFO: "PENDING", SUSPEND: "SUSPENDED"}
ALLOWED_FROM = {VERIFY: {"PENDING", "REJECTED", "SUSPENDED"}, REJECT: {"PENDING", "VERIFIED", "SUSPENDED"},
                REQUEST_INFO: {"PENDING", "REJECTED"}, SUSPEND: {"PENDING", "VERIFIED"}}


def clean_note(raw, required=False):
    note = _CONTROL.sub(" ", (raw or "")).strip()
    note = re.sub(r"\s{2,}", " ", note)
    if required and not note:
        raise AssistanceError(400, "NOTE_REQUIRED", "Please give a short reason the member can act on.")
    if len(note) > NOTE_MAX:
        raise AssistanceError(400, "NOTE_TOO_LONG", f"Keep the note under {NOTE_MAX} characters.")
    return note or None


def manager_code(user_id):
    return f"MGR-{int(user_id):05d}"


def member_code(role, user_id):
    return umusare_code(user_id) if role == ROLE_UMUSARE else driver_code(user_id)


# ---------------------------------------------------------------- scope
def actor_role(cursor, actor_id):
    """The actor's current role, read from the database (never from the request)."""
    cursor.execute("SELECT role, is_active FROM users WHERE id=%s", (actor_id,))
    row = cursor.fetchone()
    if not row or not row["is_active"]:
        raise AssistanceError(403, "FORBIDDEN", "You do not have permission for this action.")
    return row["role"]


def manager_cooperative(cursor, manager_id):
    """The cooperative a manager may manage, or None (no APPROVED manager membership)."""
    cursor.execute(
        "SELECT c.id, c.name, c.code, c.district, c.status, c.manager_user_id FROM cooperative_memberships m "
        "JOIN cooperatives c ON c.id = m.cooperative_id JOIN users u ON u.id = m.user_id "
        "WHERE m.user_id=%s AND m.member_role='manager' AND m.status='APPROVED' AND u.role='manager'",
        (manager_id,))
    return cursor.fetchone()


def cooperative_manager(cursor, cooperative_id):
    """Contact card of a cooperative's manager: the assigned manager if still valid, else the
    earliest approved manager member; None when the cooperative has no manager."""
    cursor.execute(
        "SELECT u.id, u.username, u.email, u.phone, u.is_active, (u.id = c.manager_user_id) AS assigned "
        "FROM cooperatives c JOIN cooperative_memberships m ON m.cooperative_id = c.id "
        "AND m.member_role='manager' AND m.status='APPROVED' JOIN users u ON u.id = m.user_id AND u.role='manager' "
        "WHERE c.id=%s AND u.is_active=1 ORDER BY assigned DESC, u.id LIMIT 1", (cooperative_id,))
    r = cursor.fetchone()
    if r is None:
        return None
    return {"user_id": r["id"], "name": r["username"], "email": r["email"], "phone": r["phone"],
            "code": manager_code(r["id"]), "active": bool(r["is_active"])}


def _member(cursor, user_id, lock=False):
    cursor.execute(
        "SELECT u.id, u.username, u.email, u.phone, u.role, u.is_active, u.created_at, u.last_login_at, "
        "m.cooperative_id, m.member_role, m.status AS membership_status, m.reviewed_at AS membership_reviewed_at, "
        "c.name AS cooperative, c.code AS cooperative_code FROM users u "
        "LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
        "LEFT JOIN cooperatives c ON c.id = m.cooperative_id WHERE u.id=%s" + (" FOR UPDATE" if lock else ""),
        (user_id,))
    return cursor.fetchone()


def _profile(cursor, role, user_id, lock=False):
    table = PROFILE_TABLE[role]
    extra = (", availability, requests_received, requests_accepted" if role == ROLE_UMUSARE else
             ", license_number, vehicle_plate_number, vehicle_make, vehicle_model, vehicle_type, vehicle_updated_at")
    cursor.execute(f"SELECT verification_status, verified_by, verified_at, verified_cooperative_id, reviewed_by, "
                   f"reviewed_at, verification_note, info_requested_at{extra} FROM {table} WHERE user_id=%s"
                   + (" FOR UPDATE" if lock else ""), (user_id,))
    return cursor.fetchone()


def authorize_member(cursor, actor_id, target_id, lock=False):
    """Return (actor_role, member) if the actor may manage this driver/Umusare, else raise.

    Out-of-scope and non-existent members both raise 404 so a manager cannot probe other cooperatives.
    """
    role = actor_role(cursor, actor_id)
    try:
        target_id = int(target_id)
    except (TypeError, ValueError):
        raise AssistanceError(404, "NOT_FOUND", "Member not found.")
    member = _member(cursor, target_id, lock=lock)
    if member is None or member["role"] not in MANAGED_ROLES:
        raise AssistanceError(404, "NOT_FOUND", "Member not found.")
    if role == ROLE_ADMIN:
        return role, member
    if role != ROLE_MANAGER:
        raise AssistanceError(403, "FORBIDDEN", "You do not have permission for this action.")
    coop = manager_cooperative(cursor, actor_id)
    if (coop is None or member["cooperative_id"] != coop["id"] or member["membership_status"] == "REVOKED"
            or member["member_role"] != member["role"]):
        raise AssistanceError(404, "NOT_FOUND", "Member not found.")
    return role, member


# ---------------------------------------------------------------- verification
def own_verification(user_id):
    """A driver's / Umusare's own verification state and their cooperative manager's contact."""
    with transaction() as cursor:
        member = _member(cursor, user_id)
        if member is None or member["role"] not in MANAGED_ROLES:
            return None
        prof = _profile(cursor, member["role"], user_id) or {}
        manager = None
        if member["cooperative_id"] and member["membership_status"] != "REVOKED":
            manager = cooperative_manager(cursor, member["cooperative_id"])
    status = prof.get("verification_status") or "PENDING"
    return {"status": status, "verified": status == "VERIFIED", "verified_at": prof.get("verified_at"),
            "reviewed_at": prof.get("reviewed_at"), "note": prof.get("verification_note"),
            "info_requested": status == "PENDING" and prof.get("info_requested_at") is not None,
            "cooperative": member["cooperative"], "cooperative_code": member["cooperative_code"],
            "membership_status": member["membership_status"], "manager": manager,
            "member_code": member_code(member["role"], user_id)}


def verification_blockers(cursor, user_id, role):
    """What still prevents cooperative verification (email OTP, and the vehicle plate for drivers)."""
    cursor.execute("SELECT u.email_verified_at, dp.vehicle_plate_number FROM users u "
                   "LEFT JOIN driver_profiles dp ON dp.user_id = u.id WHERE u.id=%s", (user_id,))
    r = cursor.fetchone() or {}
    missing = []
    if not r.get("email_verified_at"):
        missing.append("the member has not verified their email address")
    if role == ROLE_DRIVER and not r.get("vehicle_plate_number"):
        missing.append("the vehicle plate number is missing")
    return missing


def review_member(actor_id, target_id, action, note=None):
    """VERIFY / REJECT / REQUEST_INFO / SUSPEND a driver or Umusare in the actor's scope (audited)."""
    if action not in ACTIONS:
        raise AssistanceError(400, "INVALID", "Unknown verification action.")
    note = clean_note(note, required=action in (REJECT, REQUEST_INFO))
    with transaction() as cursor:
        role, member = authorize_member(cursor, actor_id, target_id, lock=True)
        uid, mrole = member["id"], member["role"]
        table = PROFILE_TABLE[mrole]
        prof = _profile(cursor, mrole, uid, lock=True)
        if prof is None:
            cursor.execute(f"INSERT INTO {table} (user_id) VALUES (%s)", (uid,))
            prof = _profile(cursor, mrole, uid, lock=True)
        current = prof["verification_status"]
        if current not in ALLOWED_FROM[action]:
            raise AssistanceError(409, "INVALID_TRANSITION",
                                  f"A member who is {current} cannot be changed with this action.")
        if action == VERIFY:
            missing = verification_blockers(cursor, uid, mrole)
            if missing:
                raise AssistanceError(409, "INCOMPLETE", "This member cannot be verified yet: " + "; ".join(missing) + ".")
        new, now = DECISION[action], utcnow()
        if mrole == ROLE_UMUSARE and new != "VERIFIED":
            _take_offline(cursor, uid)          # never matchable unless VERIFIED; refused while assisting
        if action == VERIFY:
            cursor.execute(f"UPDATE {table} SET verification_status='VERIFIED', verified_by=%s, verified_at=%s, "
                           f"verified_cooperative_id=%s, reviewed_by=%s, reviewed_at=%s, verification_note=NULL, "
                           f"info_requested_at=NULL WHERE user_id=%s",
                           (actor_id, now, member["cooperative_id"], actor_id, now, uid))
            if member["cooperative_id"] and member["membership_status"] != "APPROVED":
                cursor.execute("UPDATE cooperative_memberships SET status='APPROVED', reviewed_by=%s, reviewed_at=%s "
                               "WHERE user_id=%s", (actor_id, now, uid))
        else:
            cursor.execute(f"UPDATE {table} SET verification_status=%s, reviewed_by=%s, reviewed_at=%s, "
                           f"verification_note=%s, info_requested_at=%s WHERE user_id=%s",
                           (new, actor_id, now, note, now if action == REQUEST_INFO else None, uid))
        event = {VERIFY: audit.USER_VERIFIED, REJECT: audit.USER_VERIFICATION_REJECTED,
                 REQUEST_INFO: audit.USER_VERIFICATION_INFO_REQUESTED, SUSPEND: audit.USER_SUSPENDED}[action]
        audit.record(event, actor_id=actor_id, target_type="user", target_id=uid,
                     cooperative_id=member["cooperative_id"],
                     details={"role": mrole, "from": current, "to": new, "by": role, "note": bool(note)},
                     cursor=cursor)
        title = {VERIFY: "Your account has been verified by your cooperative.",
                 REJECT: "Your verification was not approved. See your dashboard for the reason.",
                 REQUEST_INFO: "Your cooperative manager requested more information.",
                 SUSPEND: "Your verification has been suspended. Contact your cooperative manager."}[action]
        kind = {VERIFY: notes.VERIFICATION_APPROVED, REJECT: notes.VERIFICATION_REJECTED,
                REQUEST_INFO: notes.VERIFICATION_INFO_REQUESTED, SUSPEND: notes.VERIFICATION_SUSPENDED}[action]
        notes.notify(cursor, uid, kind, title, "/dashboard")
    return new


def set_member_active(actor_id, target_id, active):
    """Activate / deactivate a driver or Umusare account in the actor's scope (audited)."""
    with transaction() as cursor:
        role, member = authorize_member(cursor, actor_id, target_id, lock=True)
        if not active and member["role"] == ROLE_UMUSARE:
            _take_offline(cursor, member["id"])
        cursor.execute("UPDATE users SET is_active=%s WHERE id=%s", (int(bool(active)), member["id"]))
        audit.record(audit.ACCOUNT_STATUS_CHANGED, actor_id=actor_id, target_type="user", target_id=member["id"],
                     cooperative_id=member["cooperative_id"],
                     details={"active": bool(active), "role": member["role"], "by": role}, cursor=cursor)


def request_verification(cursor, user_id, cooperative_id, role, name):
    """On registration: audit the request and tell the cooperative's manager (inside the caller's transaction)."""
    audit.record(audit.USER_VERIFICATION_REQUESTED, actor_id=user_id, target_type="user", target_id=user_id,
                 cooperative_id=cooperative_id, details={"role": role}, cursor=cursor)
    manager = cooperative_manager(cursor, cooperative_id)
    if manager:
        label = "Umusare" if role == ROLE_UMUSARE else "driver"
        notes.notify(cursor, manager["user_id"], notes.VERIFICATION_REQUESTED,
                     f"New verification request: {name} ({label})", f"/manager/members/{user_id}")


# ---------------------------------------------------------------- read models
def _operational(role, row, monitoring_on):
    if not row["is_active"]:
        return "DEACTIVATED"
    if role == ROLE_UMUSARE:
        if row["verification_status"] != "VERIFIED":
            return "UNAVAILABLE"
        return {"AVAILABLE": "AVAILABLE", "BUSY": "ASSISTING"}.get(row["availability"], "OFFLINE")
    if row["active_assistance"]:
        return "ASSISTANCE ACTIVE"
    return "ONLINE" if monitoring_on else "OFFLINE"


def _members(cursor, cooperative_id=None):
    """Drivers and Umusare (one cooperative, or all when None). No locations are selected."""
    where = "WHERE u.role IN ('driver','umusare') AND m.member_role = u.role AND m.status <> 'REVOKED'"
    args = ()
    if cooperative_id is not None:
        where += " AND m.cooperative_id = %s"
        args = (cooperative_id,)
    cursor.execute(
        "SELECT u.id, u.username, u.email, u.phone, u.role, u.is_active, u.created_at, u.email_verified_at, "
        "m.cooperative_id, m.status AS membership_status, c.name AS cooperative, m.group_id, g.name AS group_name, "
        "dp.vehicle_plate_number, "
        "COALESCE(IF(u.role='driver', dp.verification_status, up.verification_status), 'PENDING') AS verification_status, "
        "IF(u.role='driver', dp.verified_at, up.verified_at) AS verified_at, "
        "IF(u.role='driver', dp.reviewed_at, up.reviewed_at) AS reviewed_at, "
        "IF(u.role='driver', dp.verification_note, up.verification_note) AS verification_note, "
        "IF(u.role='driver', dp.info_requested_at, up.info_requested_at) AS info_requested_at, "
        "up.availability, "
        "(SELECT r.status FROM assistance_requests r WHERE (r.driver_id = u.id OR r.accepted_umusare_id = u.id) "
        " AND r.status IN ('REQUESTED','MATCHING','ACCEPTED','DRIVER_CONNECTED') LIMIT 1) AS active_assistance, "
        "(SELECT COUNT(*) FROM assistance_requests r WHERE r.accepted_umusare_id = u.id AND r.status='COMPLETED') "
        " AS completed_assists, "
        "(SELECT COUNT(*) FROM assistance_requests r WHERE r.driver_id = u.id AND r.status='COMPLETED') AS completed_requests "
        "FROM users u JOIN cooperative_memberships m ON m.user_id = u.id "
        "JOIN cooperatives c ON c.id = m.cooperative_id LEFT JOIN cooperative_groups g ON g.id = m.group_id "
        "LEFT JOIN driver_profiles dp ON dp.user_id = u.id LEFT JOIN umusare_profiles up ON up.user_id = u.id "
        f"{where} ORDER BY FIELD(COALESCE(IF(u.role='driver', dp.verification_status, up.verification_status), "
        "'PENDING'),'PENDING','REJECTED','SUSPENDED','VERIFIED'), u.username", args)
    rows = cursor.fetchall()
    for r in rows:
        on = r["role"] == ROLE_DRIVER and monitoring_status(r["id"]) is not None
        r["code"] = member_code(r["role"], r["id"])
        r["operational"] = _operational(r["role"], r, on)
        r["info_requested"] = r["verification_status"] == "PENDING" and r["info_requested_at"] is not None
        r["email_verified"] = r["email_verified_at"] is not None
        r["ready"] = r["email_verified"] and (r["role"] != ROLE_DRIVER or bool(r["vehicle_plate_number"]))
    return rows


def _active_assistance(cursor, cooperative_id):
    """Active requests involving the cooperative's members: status and ~1 km area only (read-only)."""
    cursor.execute(
        "SELECT r.id, r.status, r.trigger_source, r.created_at, r.approx_lat, r.approx_lon, "
        "d.username AS driver, u.username AS umusare FROM assistance_requests r "
        "JOIN users d ON d.id = r.driver_id LEFT JOIN users u ON u.id = r.accepted_umusare_id "
        "LEFT JOIN cooperative_memberships um ON um.user_id = r.accepted_umusare_id "
        "WHERE r.status IN ('REQUESTED','MATCHING','ACCEPTED','DRIVER_CONNECTED') "
        "AND (r.driver_cooperative_id = %s OR um.cooperative_id = %s) ORDER BY r.created_at DESC LIMIT 25",
        (cooperative_id, cooperative_id))
    rows = cursor.fetchall()
    for r in rows:
        r["code"] = assistance_code(r["id"])
        r["approx_area"] = f"{float(r.pop('approx_lat')):.2f}, {float(r.pop('approx_lon')):.2f}"
        r["type_label"] = "AI-triggered" if r["trigger_source"] == "AI_TRIGGERED" else "Driver-initiated"
    return rows


def _recent_activity(cursor, cooperative_id=None, actor_id=None, limit=15):
    """Audit entries (action, time, actor name). Details are never shown: they may hold internal data."""
    where, args = [], []
    if cooperative_id is not None:
        where.append("a.cooperative_id = %s")
        args.append(cooperative_id)
    if actor_id is not None:
        where.append("a.actor_user_id = %s")
        args.append(actor_id)
    cursor.execute("SELECT a.action, a.occurred_at, a.target_type, u.username AS actor FROM audit_logs a "
                   "LEFT JOIN users u ON u.id = a.actor_user_id WHERE " + " AND ".join(where)
                   + " AND a.action NOT IN ('CHAT_MESSAGE_SENT','LOGIN_SUCCEEDED','LOGIN_FAILED','LOGOUT') "
                   "ORDER BY a.occurred_at DESC, a.id DESC LIMIT %s", (*args, limit))
    rows = cursor.fetchall()
    for r in rows:
        r["label"] = ACTION_LABELS.get(r["action"], r["action"].replace("_", " ").capitalize())
    return rows


def _stats(members, active):
    drivers = [m for m in members if m["role"] == ROLE_DRIVER]
    umusare = [m for m in members if m["role"] == ROLE_UMUSARE]
    return {
        "drivers": len(drivers), "umusare": len(umusare),
        "drivers_verified": sum(m["verification_status"] == "VERIFIED" for m in drivers),
        "drivers_pending": sum(m["verification_status"] == "PENDING" for m in drivers),
        "umusare_verified": sum(m["verification_status"] == "VERIFIED" for m in umusare),
        "umusare_pending": sum(m["verification_status"] == "PENDING" for m in umusare),
        "umusare_available": sum(m["operational"] in ("AVAILABLE", "ASSISTING") for m in umusare),
        "active_assistance": len(active),
    }


def manager_console(manager_id):
    """Everything the manager dashboard shows, scoped to the manager's own cooperative (None if unassigned)."""
    with transaction() as cursor:
        coop = manager_cooperative(cursor, manager_id)
        if coop is None:
            return None
        cursor.execute("SELECT username, email, phone FROM users WHERE id=%s", (manager_id,))
        me = cursor.fetchone()
        members = _members(cursor, coop["id"])
        active = _active_assistance(cursor, coop["id"])
        activity = _recent_activity(cursor, cooperative_id=coop["id"])
        from .group_service import groups_of
        groups = groups_of(cursor, coop["id"])
    return {"cooperative": coop, "groups": groups, "manager": {**me, "code": manager_code(manager_id),
                                             "assigned": coop["manager_user_id"] == manager_id},
            "drivers": [m for m in members if m["role"] == ROLE_DRIVER],
            "umusare": [m for m in members if m["role"] == ROLE_UMUSARE],
            "stats": _stats(members, active), "active": active, "activity": activity}


def member_detail(actor_id, target_id):
    """One member's profile for a manager (own cooperative) or admin. Only stored data is shown."""
    with transaction() as cursor:
        _, member = authorize_member(cursor, actor_id, target_id)
        prof = _profile(cursor, member["role"], member["id"]) or {}
        verifier = None
        if prof.get("verified_by"):
            cursor.execute("SELECT username FROM users WHERE id=%s", (prof["verified_by"],))
            verifier = (cursor.fetchone() or {}).get("username")
        cursor.execute("SELECT r.id, r.status, r.trigger_source, r.created_at, r.payment_status FROM assistance_requests r "
                       "WHERE r.driver_id=%s OR r.accepted_umusare_id=%s ORDER BY r.created_at DESC LIMIT 10",
                       (member["id"], member["id"]))
        history = cursor.fetchall()
        rows = [m for m in _members(cursor, member["cooperative_id"]) if m["id"] == member["id"]] \
            if member["cooperative_id"] else []
        blockers = verification_blockers(cursor, member["id"], member["role"])
        from .group_service import groups_of
        groups = groups_of(cursor, member["cooperative_id"]) if member["cooperative_id"] else []
        from .badges import _row, evaluate
        badge = evaluate(_row(cursor, member["id"]))
    for h in history:
        h["code"] = assistance_code(h["id"])
        h["active"] = h["status"] in ACTIVE
    status = prof.get("verification_status") or "PENDING"
    return {**member, "code": member_code(member["role"], member["id"]), "verification_status": status,
            "verified_at": prof.get("verified_at"), "verified_by_name": verifier, "reviewed_at": prof.get("reviewed_at"),
            "verification_note": prof.get("verification_note"),
            "info_requested": status == "PENDING" and prof.get("info_requested_at") is not None,
            "license_number": prof.get("license_number"),
            "availability": prof.get("availability"), "requests_received": prof.get("requests_received"),
            "requests_accepted": prof.get("requests_accepted"),
            "operational": rows[0]["operational"] if rows else "—",
            "completed_assists": rows[0]["completed_assists"] if rows else 0,
            "active_assistance": rows[0]["active_assistance"] if rows else None, "history": history,
            "vehicle": {k: prof.get(k) for k in ("vehicle_plate_number", "vehicle_make", "vehicle_model", "vehicle_type",
                                                  "vehicle_updated_at")} if member["role"] == ROLE_DRIVER else None,
            "group_id": rows[0]["group_id"] if rows else None, "group_name": rows[0]["group_name"] if rows else None,
            "groups": groups, "blockers": blockers, "badge": badge,
            "email_verified": rows[0]["email_verified"] if rows else False}


def set_own_phone(user_id, raw):
    try:
        phone = normalize_phone(raw)
    except ValueError as exc:
        raise AssistanceError(400, "INVALID_PHONE", str(exc))
    with transaction() as cursor:
        cursor.execute("UPDATE users SET phone=%s WHERE id=%s", (phone, user_id))
        audit.record(audit.PHONE_UPDATED, actor_id=user_id, target_type="user", target_id=user_id, cursor=cursor)
    return phone


# ---------------------------------------------------------------- admin: cooperatives and managers
def _require_admin(cursor, actor_id):
    if actor_role(cursor, actor_id) != ROLE_ADMIN:
        raise AssistanceError(403, "FORBIDDEN", "Only administrators can do this.")


def cooperatives_overview():
    with transaction() as cursor:
        cursor.execute(
            "SELECT c.id, c.name, c.code, c.district, c.status, c.manager_user_id, c.created_at, "
            "SUM(m.member_role='driver' AND m.status<>'REVOKED') AS drivers, "
            "SUM(m.member_role='umusare' AND m.status<>'REVOKED') AS umusare, "
            "SUM(m.member_role='manager' AND m.status='APPROVED') AS managers, "
            "(SELECT COUNT(*) FROM cooperative_groups g WHERE g.cooperative_id = c.id) AS `groups` "
            "FROM cooperatives c LEFT JOIN cooperative_memberships m ON m.cooperative_id = c.id "
            "GROUP BY c.id ORDER BY c.name")
        coops = cursor.fetchall()
        cursor.execute(
            "SELECT m.cooperative_id, SUM(COALESCE(IF(u.role='driver', dp.verification_status, up.verification_status), "
            "'PENDING')='PENDING') AS pending FROM cooperative_memberships m JOIN users u ON u.id=m.user_id "
            "AND u.role IN ('driver','umusare') AND m.member_role=u.role AND m.status<>'REVOKED' "
            "LEFT JOIN driver_profiles dp ON dp.user_id=u.id LEFT JOIN umusare_profiles up ON up.user_id=u.id "
            "GROUP BY m.cooperative_id")
        pending = {r["cooperative_id"]: int(r["pending"] or 0) for r in cursor.fetchall()}
        for c in coops:
            c["manager"] = cooperative_manager(cursor, c["id"])
            c["pending"] = pending.get(c["id"], 0)
        cursor.execute("SELECT u.id, u.username, u.email, u.is_active, c.name AS cooperative FROM users u "
                       "LEFT JOIN cooperative_memberships m ON m.user_id=u.id AND m.status='APPROVED' "
                       "LEFT JOIN cooperatives c ON c.id=m.cooperative_id WHERE u.role='manager' ORDER BY u.username")
        managers = cursor.fetchall()
    for m in managers:
        m["code"] = manager_code(m["id"])
    return coops, managers


def cooperative_detail(cooperative_id):
    with transaction() as cursor:
        cursor.execute("SELECT id, name, code, district, status, manager_user_id, created_at FROM cooperatives "
                       "WHERE id=%s", (cooperative_id,))
        coop = cursor.fetchone()
        if coop is None:
            return None
        manager = cooperative_manager(cursor, cooperative_id)
        members = _members(cursor, cooperative_id)
        cursor.execute("SELECT u.id, u.username, u.email, u.phone, u.is_active, u.last_login_at, m.status "
                       "FROM cooperative_memberships m JOIN users u ON u.id=m.user_id "
                       "WHERE m.cooperative_id=%s AND m.member_role='manager' ORDER BY u.username", (cooperative_id,))
        managers = cursor.fetchall()
        active = _active_assistance(cursor, cooperative_id)
        activity = _recent_activity(cursor, actor_id=manager["user_id"]) if manager else []
        cursor.execute("SELECT id, username, email FROM users WHERE role='manager' AND is_active=1 ORDER BY username")
        candidates = cursor.fetchall()
        from .group_service import groups_of
        groups = groups_of(cursor, cooperative_id)
    for m in managers:
        m["code"] = manager_code(m["id"])
    return {"cooperative": coop, "manager": manager, "managers": managers, "members": members,
            "stats": _stats(members, active), "active": active, "manager_activity": activity,
            "candidates": candidates, "groups": groups}


def update_cooperative(admin_id, cooperative_id, name, district, status):
    name, district = (name or "").strip(), (district or "").strip() or None
    if not 3 <= len(name) <= 120 or (district and len(district) > 80) or status not in ("APPROVED", "SUSPENDED"):
        raise AssistanceError(400, "INVALID", "Enter a name (3-120 characters), an optional district and a status.")
    with transaction() as cursor:
        _require_admin(cursor, admin_id)
        cursor.execute("SELECT name, status FROM cooperatives WHERE id=%s FOR UPDATE", (cooperative_id,))
        old = cursor.fetchone()
        if old is None:
            raise AssistanceError(404, "NOT_FOUND", "Cooperative not found.")
        if status == "APPROVED" and old["status"] != "APPROVED" and cooperative_manager(cursor, cooperative_id) is None:
            raise AssistanceError(409, "NO_MANAGER", "Assign an active manager before activating this cooperative.")
        cursor.execute("SELECT id FROM cooperatives WHERE name=%s AND id<>%s", (name, cooperative_id))
        if cursor.fetchone():
            raise AssistanceError(409, "DUPLICATE", "Another cooperative already uses this name.")
        cursor.execute("UPDATE cooperatives SET name=%s, district=%s, status=%s WHERE id=%s",
                       (name, district, status, cooperative_id))
        audit.record(audit.COOPERATIVE_UPDATED, actor_id=admin_id, target_type="cooperative", target_id=cooperative_id,
                     cooperative_id=cooperative_id, details={"status": status, "renamed": old["name"] != name},
                     cursor=cursor)


def assign_manager(admin_id, cooperative_id, manager_id):
    """Make an existing manager account the cooperative's manager (MANAGER_ASSIGNED / MANAGER_CHANGED).

    The manager's membership moves to this cooperative (a manager belongs to exactly one). A replaced
    manager's membership is REVOKED, so they lose access to this cooperative's members immediately.
    """
    with transaction() as cursor:
        _require_admin(cursor, admin_id)
        cursor.execute("SELECT id, name, manager_user_id FROM cooperatives WHERE id=%s FOR UPDATE", (cooperative_id,))
        coop = cursor.fetchone()
        if coop is None:
            raise AssistanceError(404, "NOT_FOUND", "Cooperative not found.")
        try:
            manager_id = int(manager_id)
        except (TypeError, ValueError):
            raise AssistanceError(400, "INVALID", "Choose a manager account.")
        cursor.execute("SELECT id, role, is_active, username FROM users WHERE id=%s FOR UPDATE", (manager_id,))
        user = cursor.fetchone()
        if user is None or user["role"] != ROLE_MANAGER or not user["is_active"]:
            raise AssistanceError(400, "NOT_MANAGER", "Only active manager accounts can be assigned. "
                                                      "Give the user the manager role first.")
        previous = coop["manager_user_id"]
        if previous == manager_id:
            raise AssistanceError(409, "UNCHANGED", f"{user['username']} already manages this cooperative.")
        now = utcnow()
        cursor.execute("UPDATE cooperatives SET manager_user_id=NULL WHERE manager_user_id=%s", (manager_id,))
        cursor.execute(
            "INSERT INTO cooperative_memberships (user_id, cooperative_id, member_role, status, reviewed_by, reviewed_at) "
            "VALUES (%s,%s,'manager','APPROVED',%s,%s) ON DUPLICATE KEY UPDATE group_id=NULL, "
            "cooperative_id=VALUES(cooperative_id), "
            "member_role='manager', status='APPROVED', reviewed_by=VALUES(reviewed_by), reviewed_at=VALUES(reviewed_at)",
            (manager_id, cooperative_id, admin_id, now))
        if previous:
            cursor.execute("UPDATE cooperative_memberships SET status='REVOKED', reviewed_by=%s, reviewed_at=%s "
                           "WHERE user_id=%s AND cooperative_id=%s AND member_role='manager'",
                           (admin_id, now, previous, cooperative_id))
        cursor.execute("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s", (manager_id, cooperative_id))
        audit.record(audit.MANAGER_CHANGED if previous else audit.MANAGER_ASSIGNED, actor_id=admin_id,
                     target_type="cooperative", target_id=cooperative_id, cooperative_id=cooperative_id,
                     details={"manager": manager_id, "previous": previous}, cursor=cursor)
        notes.notify(cursor, manager_id, notes.MANAGER_ASSIGNED,
                     f"You are now the manager of {coop['name']}.", "/manager-dashboard")
    return user["username"]


def set_manager_active(admin_id, manager_id, active):
    with transaction() as cursor:
        _require_admin(cursor, admin_id)
        cursor.execute("SELECT id, role FROM users WHERE id=%s FOR UPDATE", (manager_id,))
        user = cursor.fetchone()
        if user is None or user["role"] != ROLE_MANAGER:
            raise AssistanceError(404, "NOT_FOUND", "Manager not found.")
        cursor.execute("UPDATE users SET is_active=%s WHERE id=%s", (int(bool(active)), manager_id))
        audit.record(audit.ACCOUNT_STATUS_CHANGED, actor_id=admin_id, target_type="user", target_id=manager_id,
                     details={"active": bool(active), "role": ROLE_MANAGER}, cursor=cursor)


def set_manager_phone(admin_id, manager_id, raw):
    try:
        phone = normalize_phone(raw)
    except ValueError as exc:
        raise AssistanceError(400, "INVALID_PHONE", str(exc))
    with transaction() as cursor:
        _require_admin(cursor, admin_id)
        cursor.execute("SELECT role FROM users WHERE id=%s FOR UPDATE", (manager_id,))
        user = cursor.fetchone()
        if user is None or user["role"] != ROLE_MANAGER:
            raise AssistanceError(404, "NOT_FOUND", "Manager not found.")
        cursor.execute("UPDATE users SET phone=%s WHERE id=%s", (phone, manager_id))
        audit.record(audit.PHONE_UPDATED, actor_id=admin_id, target_type="user", target_id=manager_id, cursor=cursor)
    return phone


def verification_queue(status="PENDING", cooperative_id=None):
    """Admin: drivers and Umusare across all cooperatives filtered by verification status."""
    with transaction() as cursor:
        rows = _members(cursor, cooperative_id)
        cursor.execute("SELECT id, name FROM cooperatives ORDER BY name")
        coops = cursor.fetchall()
    counts = {s: sum(r["verification_status"] == s for r in rows) for s in STATES}
    if status in STATES:
        rows = [r for r in rows if r["verification_status"] == status]
    return rows, coops, counts

