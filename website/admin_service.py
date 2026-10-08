# website/admin_service.py
"""Read models and account actions for administrators (drivers, Umusare, assistance, pricing).

Administrators never see live coordinates here (approximate areas only) and there is
no operation that changes an AI sobriety assessment: monitoring status is read-only.
"""
from . import audit
from .assistance_service import (ACTIVE, P_COMPLETED, P_DISPUTED, P_PENDING, P_SENT, AssistanceError, TYPE_LABELS,
                                 assistance_code, transaction, umusare_code, utcnow)
from .auth import normalize_phone
from .pricing import format_money

VERIFICATION_STATES = ("PENDING", "VERIFIED", "SUSPENDED", "REJECTED")
FILTERS = {
    "status": {"active": "r.status IN ('REQUESTED','MATCHING','ACCEPTED','DRIVER_CONNECTED')",
               "completed": "r.status = 'COMPLETED'", "cancelled": "r.status = 'CANCELLED'",
               "no_umusare": "r.status = 'NO_UMUSARE_AVAILABLE'"},
    "type": {"ai": "r.trigger_source = 'AI_TRIGGERED'", "driver": "r.trigger_source = 'DRIVER_INITIATED'"},
    "payment": {"pending": "r.payment_status IN ('PAYMENT_PENDING','PAYMENT_SENT','PAYMENT_DISPUTED')",
                "completed": "r.payment_status = 'PAYMENT_COMPLETED'"},
}


def driver_code(user_id):
    return f"DRV-{int(user_id):05d}"


def monitoring_status(user_id):
    """Current AI assessment of a driver's live session, read-only (None when not monitoring)."""
    try:
        from . import monitoring
        engine, recorder = monitoring._engine, monitoring._recorder
        if engine is None or recorder is None or recorder.active_owner() != user_id or not engine.is_running:
            return None
        a = engine.snapshot().assessment
        return a.get("assessment") if a else "NO_CLASSIFIER"
    except Exception:
        return None


# ---------------------------------------------------------------- drivers
def drivers_overview():
    with transaction() as cursor:
        cursor.execute(
            "SELECT u.id, u.username, u.email, u.phone, u.is_active, u.created_at, u.email_verified_at, c.id AS cooperative_id, "
            "c.name AS cooperative, m.status AS membership_status, g.name AS group_name, dp.vehicle_plate_number, "
            "COALESCE(dp.verification_status, 'PENDING') AS verification_status, "
            "(SELECT r.status FROM assistance_requests r WHERE r.driver_id = u.id AND r.status IN "
            " ('REQUESTED','MATCHING','ACCEPTED','DRIVER_CONNECTED') LIMIT 1) AS active_assistance, "
            "(SELECT COUNT(*) FROM assistance_requests r WHERE r.driver_id = u.id AND r.status = 'COMPLETED') AS completed "
            "FROM users u LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
            "LEFT JOIN cooperatives c ON c.id = m.cooperative_id LEFT JOIN cooperative_groups g ON g.id = m.group_id "
            "LEFT JOIN driver_profiles dp ON dp.user_id = u.id WHERE u.role = 'driver' ORDER BY u.username")
        rows = cursor.fetchall()
    for r in rows:
        r["code"] = driver_code(r["id"])
        r["monitoring"] = monitoring_status(r["id"])
    return rows


def driver_detail(user_id):
    with transaction() as cursor:
        cursor.execute("SELECT u.id, u.username, u.email, u.phone, u.is_active, u.created_at, u.last_login_at, "
                       "c.name AS cooperative, c.code AS cooperative_code, m.status AS membership_status, "
                       "m.reviewed_at AS membership_reviewed_at FROM users u "
                       "LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
                       "LEFT JOIN cooperatives c ON c.id = m.cooperative_id WHERE u.id=%s AND u.role='driver'", (user_id,))
        driver = cursor.fetchone()
        if driver is None:
            return None, []
        cursor.execute("SELECT r.id, r.status, r.trigger_source, r.created_at, r.final_distance_km, r.final_fare, "
                       "r.fare_currency, r.payment_status, u.username AS umusare FROM assistance_requests r "
                       "LEFT JOIN users u ON u.id = r.accepted_umusare_id WHERE r.driver_id=%s "
                       "ORDER BY r.created_at DESC LIMIT 25", (user_id,))
        history = cursor.fetchall()
    driver["code"] = driver_code(user_id)
    driver["monitoring"] = monitoring_status(user_id)
    for h in history:
        h["code"] = assistance_code(h["id"])
        h["type_label"] = TYPE_LABELS.get(h["trigger_source"])
        h["fare_label"] = format_money(h["final_fare"], h["fare_currency"] or "RWF") if h["final_fare"] is not None else None
    return driver, history


# ---------------------------------------------------------------- Umusare
def umusare_overview():
    with transaction() as cursor:
        cursor.execute(
            "SELECT u.id, u.username, u.email, u.phone, u.is_active, u.created_at, c.name AS cooperative, "
            "m.status AS membership_status, p.verification_status, p.verified_at, p.availability, "
            "p.requests_received, p.requests_accepted, "
            "(SELECT r.status FROM assistance_requests r WHERE r.accepted_umusare_id = u.id AND r.status IN "
            " ('ACCEPTED','DRIVER_CONNECTED') LIMIT 1) AS active_assistance, "
            "(SELECT COUNT(*) FROM assistance_requests r WHERE r.accepted_umusare_id = u.id AND r.status='COMPLETED') "
            "AS completed, "
            "(SELECT COUNT(*) FROM assistance_offers o WHERE o.umusare_id = u.id AND o.status = 'DECLINED') AS declined "
            "FROM users u LEFT JOIN umusare_profiles p ON p.user_id = u.id "
            "LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
            "LEFT JOIN cooperatives c ON c.id = m.cooperative_id WHERE u.role = 'umusare' ORDER BY u.username")
        rows = cursor.fetchall()
    for r in rows:
        r["code"] = umusare_code(r["id"])
        received = r["requests_received"] or 0
        r["acceptance_rate"] = round(100 * (r["requests_accepted"] or 0) / received) if received else None
    return rows


# ---------------------------------------------------------------- assistance monitoring
def assistance_monitor(filters=None, limit=100):
    filters = filters or {}
    where, args = [], []
    for key, options in FILTERS.items():
        clause = options.get(filters.get(key) or "")
        if clause:
            where.append(clause)
    coop = str(filters.get("cooperative") or "")
    if coop.isdigit():
        where.append("(r.driver_cooperative_id = %s OR um.cooperative_id = %s)")
        args += [int(coop), int(coop)]
    sql = ("SELECT r.id, r.status, r.trigger_source, r.created_at, r.accepted_at, r.journey_ended_at, "
           "r.payment_completed_at, r.approx_lat, r.approx_lon, r.final_distance_km, r.journey_distance_km, "
           "r.final_fare, r.fare_currency, r.payment_status, d.username AS driver, c.name AS driver_cooperative, "
           "u.username AS umusare, uc.name AS umusare_cooperative "
           "FROM assistance_requests r JOIN users d ON d.id = r.driver_id "
           "LEFT JOIN cooperatives c ON c.id = r.driver_cooperative_id "
           "LEFT JOIN users u ON u.id = r.accepted_umusare_id "
           "LEFT JOIN cooperative_memberships um ON um.user_id = r.accepted_umusare_id "
           "LEFT JOIN cooperatives uc ON uc.id = um.cooperative_id "
           + ("WHERE " + " AND ".join(where) + " " if where else "")
           + "ORDER BY r.created_at DESC LIMIT %s")
    with transaction() as cursor:
        cursor.execute(sql, (*args, limit))
        rows = cursor.fetchall()
        cursor.execute("SELECT id, name FROM cooperatives ORDER BY name")
        cooperatives = cursor.fetchall()
        from .pricing import compute_fare, get_pricing
        pricing = get_pricing(cursor)
    for r in rows:
        r["code"] = assistance_code(r["id"])
        r["active"] = r["status"] in ACTIVE
        r["type_label"] = "AI-triggered" if r["trigger_source"] == "AI_TRIGGERED" else "Driver-initiated"
        r["approx_area"] = f"{float(r.pop('approx_lat')):.2f}, {float(r.pop('approx_lon')):.2f}"
        cur = r["fare_currency"] or pricing["currency"]
        if r["final_fare"] is not None:
            r["distance_label"] = f"{float(r['final_distance_km']):.2f} km"
            r["final_fare_label"] = format_money(r["final_fare"], cur)
            r["estimated_fare_label"] = None
        elif r["status"] == "DRIVER_CONNECTED":
            km = float(r["journey_distance_km"] or 0)
            r["distance_label"] = f"{km:.2f} km so far"
            r["final_fare_label"] = None
            r["estimated_fare_label"] = format_money(compute_fare(km, pricing), pricing["currency"])
        else:
            r["distance_label"] = r["final_fare_label"] = r["estimated_fare_label"] = None
    return rows, cooperatives


# ---------------------------------------------------------------- account actions (all audited)
def _target(cursor, user_id):
    cursor.execute("SELECT id, role, is_active FROM users WHERE id=%s FOR UPDATE", (user_id,))
    user = cursor.fetchone()
    if user is None:
        raise AssistanceError(404, "NOT_FOUND", "User not found.")
    return user


def _take_offline(cursor, umusare_id):
    cursor.execute("SELECT availability FROM umusare_profiles WHERE user_id=%s FOR UPDATE", (umusare_id,))
    prof = cursor.fetchone()
    if prof and prof["availability"] == "BUSY":
        raise AssistanceError(409, "BUSY", "This Umusare is assisting a driver right now; try again after the journey.")
    cursor.execute("DELETE FROM user_locations WHERE user_id=%s", (umusare_id,))
    cursor.execute("UPDATE umusare_profiles SET availability='OFFLINE' WHERE user_id=%s", (umusare_id,))
    cursor.execute("UPDATE assistance_offers SET status='EXPIRED', responded_at=%s WHERE umusare_id=%s AND status='OFFERED'",
                   (utcnow(), umusare_id))


def set_account_active(admin_id, user_id, active):
    if int(user_id) == int(admin_id):
        raise AssistanceError(400, "SELF", "You cannot change your own account status.")
    with transaction() as cursor:
        user = _target(cursor, user_id)
        if user["role"] not in ("driver", "umusare"):
            raise AssistanceError(400, "ROLE", "Only driver and Umusare accounts are managed here.")
        if not active and user["role"] == "umusare":
            _take_offline(cursor, user_id)
        cursor.execute("UPDATE users SET is_active=%s WHERE id=%s", (int(bool(active)), user_id))
        audit.record(audit.ACCOUNT_STATUS_CHANGED, actor_id=admin_id, target_type="user", target_id=user_id,
                     details={"active": bool(active), "role": user["role"]}, cursor=cursor)


def set_umusare_verification(admin_id, umusare_id, status):
    if status not in VERIFICATION_STATES:
        raise AssistanceError(400, "INVALID", "Unknown verification status.")
    with transaction() as cursor:
        user = _target(cursor, umusare_id)
        if user["role"] != "umusare":
            raise AssistanceError(400, "ROLE", "Only Umusare accounts have a verification status.")
        cursor.execute("SELECT verification_status FROM umusare_profiles WHERE user_id=%s FOR UPDATE", (umusare_id,))
        prof = cursor.fetchone()
        if prof is None:
            cursor.execute("INSERT INTO umusare_profiles (user_id) VALUES (%s)", (umusare_id,))
            prof = {"verification_status": "PENDING"}
        if status != "VERIFIED":
            _take_offline(cursor, umusare_id)       # an unverified Umusare can never stay matchable
            cursor.execute("UPDATE umusare_profiles SET verification_status=%s WHERE user_id=%s", (status, umusare_id))
        else:
            cursor.execute("UPDATE umusare_profiles SET verification_status='VERIFIED', verified_by=%s, verified_at=%s "
                           "WHERE user_id=%s", (admin_id, utcnow(), umusare_id))
        audit.record(audit.UMUSARE_VERIFICATION_CHANGED, actor_id=admin_id, target_type="user", target_id=umusare_id,
                     details={"from": prof["verification_status"], "to": status}, cursor=cursor)


def set_phone(admin_id, user_id, raw):
    try:
        phone = normalize_phone(raw)
    except ValueError as exc:
        raise AssistanceError(400, "INVALID_PHONE", str(exc))
    with transaction() as cursor:
        user = _target(cursor, user_id)
        if user["role"] not in ("driver", "umusare"):
            raise AssistanceError(400, "ROLE", "Only driver and Umusare accounts are managed here.")
        cursor.execute("UPDATE users SET phone=%s WHERE id=%s", (phone, user_id))
        audit.record(audit.PHONE_UPDATED, actor_id=admin_id, target_type="user", target_id=user_id, cursor=cursor)
    return phone


def payment_states():
    return (P_PENDING, P_SENT, P_COMPLETED, P_DISPUTED)
