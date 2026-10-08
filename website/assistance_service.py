# website/assistance_service.py
"""Driver -> nearby verified Umusare assistance workflow (state machine, matching, privacy).

All decisions are made here, server-side, inside one database transaction per
call; the browser only supplies its own position and button presses.

States and transitions
----------------------
    REQUESTED -> MATCHING | CANCELLED
    MATCHING  -> ACCEPTED | NO_UMUSARE_AVAILABLE | CANCELLED
    ACCEPTED  -> DRIVER_CONNECTED | CANCELLED
    DRIVER_CONNECTED -> COMPLETED
    COMPLETED, CANCELLED, NO_UMUSARE_AVAILABLE are final.

Matching
--------
Eligible Umusare: role umusare, account active, VERIFIED, availability AVAILABLE,
APPROVED membership in an APPROVED cooperative (any cooperative), and a location
shared within UMUSARE_LOCATION_MAX_AGE_S (proof they are online). Candidates are
ranked by website.geo.rank_candidates (distance first; same cooperative is a
bounded bonus). Each round offers the request to the best OFFERS_PER_ROUND
candidates; the first to accept wins (row lock). When every offer is declined or
expires, the next round runs, widening the radius along SEARCH_RADII_KM. When no
candidate exists within the largest radius: NO_UMUSARE_AVAILABLE.

Privacy
-------
Before acceptance an Umusare sees only the ~1 km area and a rounded distance.
Exact positions are shared only between the driver and the accepted Umusare
while the request is ACCEPTED/DRIVER_CONNECTED. When the request ends, the exact
pickup point is erased and the driver's live location row is deleted; positions
are never stored as history and never written to the audit log.
"""
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import math

from . import ROLE_DRIVER, ROLE_UMUSARE, get_connection
from . import audit
from .geo import Candidate, approximate, approximate_distance_label, haversine_km, rank_candidates, valid_coordinates
from .pricing import compute_fare, format_money, get_pricing, pricing_public

REQUESTED, MATCHING, ACCEPTED, DRIVER_CONNECTED = "REQUESTED", "MATCHING", "ACCEPTED", "DRIVER_CONNECTED"
COMPLETED, CANCELLED, NO_UMUSARE = "COMPLETED", "CANCELLED", "NO_UMUSARE_AVAILABLE"
ACTIVE = (REQUESTED, MATCHING, ACCEPTED, DRIVER_CONNECTED)
SHARING = (ACCEPTED, DRIVER_CONNECTED)            # live location sharing allowed only in these states
FINAL = (COMPLETED, CANCELLED, NO_UMUSARE)
TRANSITIONS = {
    REQUESTED: {MATCHING, CANCELLED},
    MATCHING: {ACCEPTED, NO_UMUSARE, CANCELLED},
    ACCEPTED: {DRIVER_CONNECTED, CANCELLED},
    DRIVER_CONNECTED: {COMPLETED},
}
# AI_TRIGGERED: requested from the POTENTIALLY_NOT_SOBER safety alert (verified server-side against the
# driver's live monitoring session). DRIVER_INITIATED: manual request, whatever the AI says. The request
# type never changes, and is never fed back into, the AI assessment.
AI_TRIGGERED, DRIVER_INITIATED = "AI_TRIGGERED", "DRIVER_INITIATED"
TRIGGERS = (AI_TRIGGERED, DRIVER_INITIATED)
TYPE_LABELS = {AI_TRIGGERED: "AI-triggered safety assistance", DRIVER_INITIATED: "Driver-initiated support"}

# Payment (a separate server-side state on the request; the browser can never set it directly).
P_ESTIMATED, P_PENDING, P_SENT = "ESTIMATED", "PAYMENT_PENDING", "PAYMENT_SENT"
P_COMPLETED, P_DISPUTED, P_CANCELLED = "PAYMENT_COMPLETED", "PAYMENT_DISPUTED", "PAYMENT_CANCELLED"
# (current, new) -> the only role allowed to make that change
PAYMENT_TRANSITIONS = {
    (P_PENDING, P_SENT): "driver",          # driver paid externally and says so
    (P_DISPUTED, P_SENT): "driver",         # driver re-confirms after a reported problem
    (P_SENT, P_COMPLETED): "umusare",       # Umusare confirms the money arrived
    (P_DISPUTED, P_COMPLETED): "umusare",
    (P_SENT, P_DISPUTED): "umusare",        # Umusare has not received it
}

# Journey distance is added up server-side from the accepted Umusare's location updates.
JOURNEY_MAX_ACCURACY_M = 100.0     # fixes less precise than this are ignored for distance
JOURNEY_MIN_STEP_KM = 0.01         # < 10 m: GPS jitter, not movement
JOURNEY_MAX_SPEED_KMH = 150.0      # faster implied speed = GPS jump; not counted


def _env_list(name, default):
    try:
        values = sorted(float(v) for v in os.environ.get(name, default).split(",") if v.strip())
        return tuple(v for v in values if v > 0) or tuple(float(v) for v in default.split(","))
    except ValueError:
        return tuple(float(v) for v in default.split(","))


def search_radii_km():
    return _env_list("ASSISTANCE_SEARCH_RADII_KM", "5,15,50")


def offers_per_round():
    return max(1, int(os.environ.get("ASSISTANCE_OFFERS_PER_ROUND", "3")))


def offer_timeout_s():
    return max(10, int(os.environ.get("ASSISTANCE_OFFER_TIMEOUT_S", "90")))


def location_max_age_s():
    return max(30, int(os.environ.get("UMUSARE_LOCATION_MAX_AGE_S", "1800")))


def eta_speed_kmh():
    """Assumed average urban speed for the arrival estimate (shown as an estimate, never a promise)."""
    try:
        return max(5.0, float(os.environ.get("ASSISTANCE_ETA_SPEED_KMH", "25")))
    except ValueError:
        return 25.0


def eta_minutes(distance_km):
    return None if distance_km is None else max(1, math.ceil(distance_km / eta_speed_kmh() * 60))


def check_payment_transition(current, new, role):
    allowed = PAYMENT_TRANSITIONS.get((current, new))
    if allowed is None:
        raise AssistanceError(409, "INVALID_PAYMENT_TRANSITION", f"Payment cannot change from {current} to {new}.")
    if allowed != role:
        raise AssistanceError(403, "FORBIDDEN", "You are not allowed to make this payment change.")


class AssistanceError(Exception):
    def __init__(self, http_status, code, message):
        super().__init__(message)
        self.http_status, self.code, self.message = http_status, code, message


class InvalidTransition(AssistanceError):
    def __init__(self, current, new):
        super().__init__(409, "INVALID_TRANSITION", f"Request cannot change from {current} to {new}.")


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@contextmanager
def transaction():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_transition(current, new):
    if new not in TRANSITIONS.get(current, set()):
        raise InvalidTransition(current, new)


def _coords(lat, lon, accuracy=None):
    if not valid_coordinates(lat, lon):
        raise AssistanceError(400, "INVALID_LOCATION", "A valid location (latitude/longitude) is required.")
    acc = None
    if accuracy not in (None, ""):
        try:
            acc = max(0.0, min(float(accuracy), 100000.0))
        except (TypeError, ValueError):
            acc = None
    return float(lat), float(lon), acc


# ---------------------------------------------------------------- loading / locking
def _lock_request(cursor, request_id):
    cursor.execute("SELECT * FROM assistance_requests WHERE id=%s FOR UPDATE", (request_id,))
    req = cursor.fetchone()
    if req is None:
        raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
    return req


def _set_status(cursor, req, new, actor_id=None, **fields):
    check_transition(req["status"], new)
    now = utcnow()
    stamp = {MATCHING: "matched_at", ACCEPTED: "accepted_at", DRIVER_CONNECTED: "connected_at",
             COMPLETED: "completed_at", CANCELLED: "cancelled_at"}.get(new)
    values = dict(fields, status=new)
    if stamp and not req.get(stamp):
        values[stamp] = now
    if new in FINAL:
        values.update(ended_at=now, pickup_lat=None, pickup_lon=None, pickup_accuracy_m=None,
                      journey_start_lat=None, journey_start_lon=None, journey_last_lat=None, journey_last_lon=None)
        if new != COMPLETED:
            values["payment_status"] = P_CANCELLED
    cursor.execute(f"UPDATE assistance_requests SET {', '.join(f'{k}=%s' for k in values)} WHERE id=%s",
                   (*values.values(), req["id"]))
    req.update(values)
    if new in FINAL:
        _stop_sharing(cursor, req)
    return req


def _stop_sharing(cursor, req):
    """End of request: withdraw open offers, delete the driver's live location, free the Umusare."""
    cursor.execute("UPDATE assistance_offers SET status='WITHDRAWN', responded_at=%s "
                   "WHERE request_id=%s AND status='OFFERED'", (utcnow(), req["id"]))
    cursor.execute("DELETE FROM user_locations WHERE user_id=%s", (req["driver_id"],))
    if req.get("accepted_umusare_id"):
        # Back to AVAILABLE if they are still sharing their availability location, else OFFLINE.
        cursor.execute("UPDATE umusare_profiles p LEFT JOIN user_locations l ON l.user_id = p.user_id "
                       "SET p.availability = IF(l.user_id IS NULL, 'OFFLINE', 'AVAILABLE') "
                       "WHERE p.user_id=%s AND p.availability='BUSY'", (req["accepted_umusare_id"],))


def _notify_driver(cursor, req, kind, title):
    """In-app notification for the driver of a request (no locations, no amounts)."""
    from .notifications import notify
    notify(cursor, req["driver_id"], kind, title, "/driver-dashboard")


def _membership(cursor, user_id):
    cursor.execute("SELECT m.cooperative_id, m.status, m.member_role, c.status AS coop_status, c.name AS coop_name "
                   "FROM cooperative_memberships m JOIN cooperatives c ON c.id=m.cooperative_id WHERE m.user_id=%s",
                   (user_id,))
    return cursor.fetchone()


# ---------------------------------------------------------------- matching
def _eligible_candidates(cursor, exclude_ids):
    cutoff = utcnow() - timedelta(seconds=location_max_age_s())
    cursor.execute(
        "SELECT u.id, l.lat, l.lon, m.cooperative_id, p.requests_received, p.requests_accepted "
        "FROM users u "
        "JOIN umusare_profiles p ON p.user_id = u.id "
        "JOIN cooperative_memberships m ON m.user_id = u.id AND m.member_role = 'umusare' AND m.status = 'APPROVED' "
        "JOIN cooperatives c ON c.id = m.cooperative_id AND c.status = 'APPROVED' "
        "JOIN user_locations l ON l.user_id = u.id "
        "WHERE u.role = 'umusare' AND u.is_active = 1 AND p.verification_status = 'VERIFIED' "
        "AND p.availability = 'AVAILABLE' AND l.updated_at >= %s", (cutoff,))
    return [Candidate(r["id"], float(r["lat"]), float(r["lon"]), r["cooperative_id"],
                      r["requests_received"], r["requests_accepted"])
            for r in cursor.fetchall() if r["id"] not in exclude_ids]


def _run_matching_round(cursor, req):
    """Offer the request to the next best candidates, widening the radius; else NO_UMUSARE_AVAILABLE."""
    if req["pickup_lat"] is None:
        return _set_status(cursor, req, NO_UMUSARE)
    cursor.execute("SELECT umusare_id FROM assistance_offers WHERE request_id=%s", (req["id"],))
    already = {r["umusare_id"] for r in cursor.fetchall()}
    candidates = _eligible_candidates(cursor, already)
    current = req.get("search_radius_km") or 0
    for radius in [r for r in search_radii_km() if r >= current]:
        ranked = rank_candidates(float(req["pickup_lat"]), float(req["pickup_lon"]),
                                 req["driver_cooperative_id"], candidates, radius)[:offers_per_round()]
        if not ranked:
            continue
        rnd, now = req["matching_round"] + 1, utcnow()
        for i, rc in enumerate(ranked, 1):
            cursor.execute("INSERT INTO assistance_offers (request_id, umusare_id, matching_round, rank_in_round, "
                           "distance_km, same_cooperative, score_km, offered_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                           (req["id"], rc.candidate.umusare_id, rnd, i, round(rc.distance_km, 3),
                            int(rc.same_cooperative), round(rc.score_km, 3), now))
            cursor.execute("UPDATE umusare_profiles SET requests_received = requests_received + 1 WHERE user_id=%s",
                           (rc.candidate.umusare_id,))
        cursor.execute("UPDATE assistance_requests SET matching_round=%s, search_radius_km=%s WHERE id=%s",
                       (rnd, radius, req["id"]))
        req.update(matching_round=rnd, search_radius_km=radius)
        audit.record(audit.ASSISTANCE_MATCHING, target_type="assistance_request", target_id=req["id"],
                     cooperative_id=req["driver_cooperative_id"],
                     details={"round": rnd, "radius_km": radius, "offered": len(ranked),
                              "other_cooperative": sum(not rc.same_cooperative for rc in ranked)}, cursor=cursor)
        return req
    audit.record(audit.ASSISTANCE_NO_UMUSARE, target_type="assistance_request", target_id=req["id"],
                 cooperative_id=req["driver_cooperative_id"],
                 details={"max_radius_km": max(search_radii_km()), "rounds": req["matching_round"]}, cursor=cursor)
    return _set_status(cursor, req, NO_UMUSARE, search_radius_km=max(search_radii_km()))


def _refresh_matching(cursor, req):
    """Expire stale offers / offers to Umusare who went unavailable, then rematch if nobody is left."""
    if req["status"] != MATCHING:
        return req
    now = utcnow()
    cursor.execute("UPDATE assistance_offers o JOIN umusare_profiles p ON p.user_id = o.umusare_id "
                   "SET o.status='EXPIRED', o.responded_at=%s "
                   "WHERE o.request_id=%s AND o.status='OFFERED' AND (o.offered_at < %s OR p.availability <> 'AVAILABLE')",
                   (now, req["id"], now - timedelta(seconds=offer_timeout_s())))
    cursor.execute("SELECT COUNT(*) AS n FROM assistance_offers WHERE request_id=%s AND status='OFFERED'", (req["id"],))
    if cursor.fetchone()["n"] == 0:
        _run_matching_round(cursor, req)
    return req


# ---------------------------------------------------------------- driver actions
def create_request(driver_id, lat, lon, accuracy=None, trigger=DRIVER_INITIATED):
    lat, lon, acc = _coords(lat, lon, accuracy)
    trigger = trigger if trigger in TRIGGERS else DRIVER_INITIATED
    with transaction() as cursor:
        # Lock the driver row: concurrent clicks are serialised, so only one active request can exist.
        cursor.execute("SELECT id, role, is_active FROM users WHERE id=%s FOR UPDATE", (driver_id,))
        user = cursor.fetchone()
        if not user or user["role"] != ROLE_DRIVER or not user["is_active"]:
            raise AssistanceError(403, "FORBIDDEN", "Only drivers can request assistance.")
        cursor.execute("SELECT id FROM assistance_requests WHERE driver_id=%s AND status IN (%s,%s,%s,%s)",
                       (driver_id, *ACTIVE))
        existing = cursor.fetchone()
        if existing:
            raise AssistanceError(409, "ACTIVE_REQUEST_EXISTS", "You already have an active assistance request.")
        m = _membership(cursor, driver_id)
        coop_id = m["cooperative_id"] if m and m["status"] in ("APPROVED", "PENDING") else None
        ax, ay = approximate(lat, lon)
        now = utcnow()
        cursor.execute("INSERT INTO assistance_requests (driver_id, driver_cooperative_id, status, trigger_source, "
                       "pickup_lat, pickup_lon, pickup_accuracy_m, approx_lat, approx_lon, created_at) "
                       "VALUES (%s,%s,'REQUESTED',%s,%s,%s,%s,%s,%s,%s)",
                       (driver_id, coop_id, trigger, lat, lon, acc, ax, ay, now))
        req = _lock_request(cursor, cursor.lastrowid)
        audit.record(audit.ASSISTANCE_REQUESTED, actor_id=driver_id, target_type="assistance_request",
                     target_id=req["id"], cooperative_id=coop_id, details={"trigger": trigger}, cursor=cursor)
        _set_status(cursor, req, MATCHING)
        _run_matching_round(cursor, req)
        return req["id"]


def cancel_request(driver_id, request_id):
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        if req["driver_id"] != driver_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        _set_status(cursor, req, CANCELLED, cancelled_by=driver_id)
        audit.record(audit.ASSISTANCE_CANCELLED, actor_id=driver_id, target_type="assistance_request",
                     target_id=request_id, cooperative_id=req["driver_cooperative_id"], cursor=cursor)


# ---------------------------------------------------------------- Umusare actions
def _lock_eligible_umusare(cursor, umusare_id):
    cursor.execute("SELECT p.verification_status, p.availability, u.is_active, u.role FROM umusare_profiles p "
                   "JOIN users u ON u.id = p.user_id WHERE p.user_id=%s FOR UPDATE", (umusare_id,))
    prof = cursor.fetchone()
    if not prof or prof["role"] != ROLE_UMUSARE or not prof["is_active"]:
        raise AssistanceError(403, "FORBIDDEN", "Only Umusare can do this.")
    if prof["verification_status"] != "VERIFIED":
        raise AssistanceError(403, "NOT_VERIFIED", "Only verified Umusare can accept assistance requests.")
    m = _membership(cursor, umusare_id)
    if not m or m["status"] != "APPROVED" or m["coop_status"] != "APPROVED":
        raise AssistanceError(403, "NO_COOPERATIVE", "An approved cooperative membership is required.")
    return prof


def accept_request(umusare_id, request_id):
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)            # serialises concurrent accepts
        cursor.execute("SELECT id, status FROM assistance_offers WHERE request_id=%s AND umusare_id=%s",
                       (request_id, umusare_id))
        offer = cursor.fetchone()
        if offer is None:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        prof = _lock_eligible_umusare(cursor, umusare_id)
        if req["status"] != MATCHING:
            raise AssistanceError(409, "NOT_OPEN", "This request is no longer open.")
        if offer["status"] != "OFFERED":
            raise AssistanceError(409, "OFFER_CLOSED", "This request is no longer offered to you.")
        if prof["availability"] != "AVAILABLE":
            raise AssistanceError(409, "NOT_AVAILABLE", "Set yourself as available before accepting.")
        _set_status(cursor, req, ACCEPTED, accepted_umusare_id=umusare_id)
        now = utcnow()
        cursor.execute("UPDATE assistance_offers SET status='ACCEPTED', responded_at=%s WHERE id=%s", (now, offer["id"]))
        cursor.execute("UPDATE assistance_offers SET status='WITHDRAWN', responded_at=%s "
                       "WHERE request_id=%s AND status='OFFERED'", (now, request_id))
        cursor.execute("UPDATE umusare_profiles SET availability='BUSY', requests_accepted = requests_accepted + 1 "
                       "WHERE user_id=%s", (umusare_id,))
        audit.record(audit.ASSISTANCE_ACCEPTED, actor_id=umusare_id, target_type="assistance_request",
                     target_id=request_id, cooperative_id=req["driver_cooperative_id"], cursor=cursor)
        _notify_driver(cursor, req, "ASSISTANCE_ACCEPTED", "Umusare accepted your request and is on the way.")


def decline_request(umusare_id, request_id):
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        cursor.execute("SELECT id, status FROM assistance_offers WHERE request_id=%s AND umusare_id=%s",
                       (request_id, umusare_id))
        offer = cursor.fetchone()
        if offer is None:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        if offer["status"] != "OFFERED" or req["status"] != MATCHING:
            raise AssistanceError(409, "OFFER_CLOSED", "This request is no longer offered to you.")
        cursor.execute("UPDATE assistance_offers SET status='DECLINED', responded_at=%s WHERE id=%s",
                       (utcnow(), offer["id"]))
        audit.record(audit.ASSISTANCE_DECLINED, actor_id=umusare_id, target_type="assistance_request",
                     target_id=request_id, cooperative_id=req["driver_cooperative_id"], cursor=cursor)
        _refresh_matching(cursor, req)                       # next round when nobody else holds the offer


def _participant_request(cursor, user_id, request_id):
    req = _lock_request(cursor, request_id)
    if user_id not in (req["driver_id"], req["accepted_umusare_id"]):
        raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
    return req


def mark_connected(user_id, request_id):
    """Umusare arrived and is with the driver: the journey starts here."""
    with transaction() as cursor:
        req = _participant_request(cursor, user_id, request_id)
        _set_status(cursor, req, DRIVER_CONNECTED)
        start = (_location(cursor, req["accepted_umusare_id"]) or _location(cursor, req["driver_id"])
                 or {"lat": float(req["pickup_lat"]), "lon": float(req["pickup_lon"])})
        now = utcnow()
        cursor.execute("UPDATE assistance_requests SET journey_started_at=%s, journey_start_lat=%s, journey_start_lon=%s, "
                       "journey_last_lat=%s, journey_last_lon=%s, journey_last_at=%s, journey_distance_km=0 WHERE id=%s",
                       (now, start["lat"], start["lon"], start["lat"], start["lon"], now, request_id))
        audit.record(audit.ASSISTANCE_CONNECTED, actor_id=user_id, target_type="assistance_request",
                     target_id=request_id, cooperative_id=req["driver_cooperative_id"], cursor=cursor)
        audit.record(audit.JOURNEY_STARTED, actor_id=user_id, target_type="assistance_request",
                     target_id=request_id, cooperative_id=req["driver_cooperative_id"], cursor=cursor)
        _notify_driver(cursor, req, "ASSISTANCE_ARRIVED", "Umusare arrived. Your journey has started.")


def _accumulate_journey(cursor, req, lat, lon, acc):
    """Add the step since the last accepted point (running total only; no location history)."""
    if req["journey_last_lat"] is None:
        return
    if acc is not None and acc > JOURNEY_MAX_ACCURACY_M:
        return
    now = utcnow()
    step = haversine_km(float(req["journey_last_lat"]), float(req["journey_last_lon"]), lat, lon)
    if step < JOURNEY_MIN_STEP_KM:
        return
    hours = max((now - (req["journey_last_at"] or now)).total_seconds(), 1.0) / 3600.0
    add = step if step / hours <= JOURNEY_MAX_SPEED_KMH else 0.0     # implausible jump: move on, do not count
    cursor.execute("UPDATE assistance_requests SET journey_distance_km = journey_distance_km + %s, journey_last_lat=%s, "
                   "journey_last_lon=%s, journey_last_at=%s WHERE id=%s", (round(add, 3), lat, lon, now, req["id"]))


def complete_request(user_id, request_id):
    """COMPLETE JOURNEY (accepted Umusare only): stop tracking, compute distance and fare server-side."""
    with transaction() as cursor:
        req = _participant_request(cursor, user_id, request_id)
        if user_id != req["accepted_umusare_id"]:
            raise AssistanceError(403, "FORBIDDEN", "Only the assisting Umusare can complete the journey.")
        check_transition(req["status"], COMPLETED)

        start = ({"lat": float(req["journey_start_lat"]), "lon": float(req["journey_start_lon"])}
                 if req["journey_start_lat"] is not None else None)
        end = (_location(cursor, user_id) or _location(cursor, req["driver_id"])
               or ({"lat": float(req["journey_last_lat"]), "lon": float(req["journey_last_lon"])}
                   if req["journey_last_lat"] is not None else start))
        straight = haversine_km(start["lat"], start["lon"], end["lat"], end["lon"]) if start and end else 0.0
        # The travelled path can never be shorter than the straight line between start and end.
        final_km = round(max(float(req["journey_distance_km"] or 0), straight), 2)
        pricing = get_pricing(cursor)
        fare = compute_fare(final_km, pricing)
        cursor.execute("SELECT phone FROM users WHERE id=%s", (user_id,))
        phone = (cursor.fetchone() or {}).get("phone")
        now = utcnow()
        _set_status(cursor, req, COMPLETED, journey_ended_at=now, final_distance_km=final_km,
                    journey_start_approx="%.2f, %.2f" % approximate(start["lat"], start["lon"]) if start else None,
                    journey_end_approx="%.2f, %.2f" % approximate(end["lat"], end["lon"]) if end else None,
                    fare_currency=pricing["currency"], fare_price_per_km=pricing["price_per_km"],
                    fare_base_fee=pricing["base_fee"], fare_minimum=pricing["minimum_fare"],
                    fare_maximum=pricing["maximum_fare"], final_fare=fare, payment_status=P_PENDING,
                    payment_pending_at=now, payment_phone=phone)
        audit.record(audit.ASSISTANCE_COMPLETED, actor_id=user_id, target_type="assistance_request",
                     target_id=request_id, cooperative_id=req["driver_cooperative_id"], cursor=cursor)
        _notify_driver(cursor, req, "ASSISTANCE_COMPLETED", "Journey completed. Please complete the payment.")
        audit.record(audit.FARE_CALCULATED, actor_id=user_id, target_type="assistance_request", target_id=request_id,
                     cooperative_id=req["driver_cooperative_id"],
                     details={"distance_km": final_km, "price_per_km": str(pricing["price_per_km"]),
                              "base_fee": str(pricing["base_fee"]), "fare": str(fare), "currency": pricing["currency"]},
                     cursor=cursor)
        return {"distance_km": final_km, "fare": int(fare), "currency": pricing["currency"]}


def update_live_location(user_id, request_id, lat, lon, accuracy=None):
    lat, lon, acc = _coords(lat, lon, accuracy)
    with transaction() as cursor:
        req = _participant_request(cursor, user_id, request_id)
        if req["status"] not in SHARING:
            raise AssistanceError(409, "SHARING_STOPPED", "Location sharing is only active during an accepted assistance.")
        _upsert_location(cursor, user_id, lat, lon, acc)
        if req["status"] == DRIVER_CONNECTED and user_id == req["accepted_umusare_id"]:
            _accumulate_journey(cursor, req, lat, lon, acc)


# ---------------------------------------------------------------- payment (two-step confirmation)
def _completed_request(cursor, request_id):
    req = _lock_request(cursor, request_id)
    if req["status"] != COMPLETED:
        raise AssistanceError(409, "NOT_COMPLETED", "Payment is only possible after the journey is completed.")
    return req


def _set_payment(cursor, req, new, role, actor_id, **fields):
    check_payment_transition(req["payment_status"], new, role)
    stamp = {P_SENT: "payment_sent_at", P_COMPLETED: "payment_completed_at", P_DISPUTED: "payment_disputed_at"}[new]
    values = dict(fields, payment_status=new, **{stamp: utcnow()})
    cursor.execute(f"UPDATE assistance_requests SET {', '.join(f'{k}=%s' for k in values)} WHERE id=%s",
                   (*values.values(), req["id"]))
    action = {P_SENT: audit.PAYMENT_SENT, P_COMPLETED: audit.PAYMENT_CONFIRMED, P_DISPUTED: audit.PAYMENT_DISPUTED}[new]
    audit.record(action, actor_id=actor_id, target_type="assistance_request", target_id=req["id"],
                 cooperative_id=req["driver_cooperative_id"],
                 details={"amount": str(req["final_fare"]), "currency": req["fare_currency"], "from": req["payment_status"]},
                 cursor=cursor)


def mark_payment_sent(driver_id, request_id):
    """The driver paid with an external method (e.g. Mobile Money). No money moves through SafeDrive."""
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        if req["driver_id"] != driver_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        if req["status"] != COMPLETED:
            raise AssistanceError(409, "NOT_COMPLETED", "Payment is only possible after the journey is completed.")
        _set_payment(cursor, req, P_SENT, "driver", driver_id)


def confirm_payment_received(umusare_id, request_id):
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        if req["accepted_umusare_id"] != umusare_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        if req["status"] != COMPLETED:
            raise AssistanceError(409, "NOT_COMPLETED", "Payment is only possible after the journey is completed.")
        _set_payment(cursor, req, P_COMPLETED, "umusare", umusare_id)


def report_payment_problem(umusare_id, request_id, note=None):
    note = (note or "").strip()[:255] or "Payment not received"
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        if req["accepted_umusare_id"] != umusare_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        if req["status"] != COMPLETED:
            raise AssistanceError(409, "NOT_COMPLETED", "Payment is only possible after the journey is completed.")
        _set_payment(cursor, req, P_DISPUTED, "umusare", umusare_id, payment_dispute_note=note)


def rate_assistance(driver_id, request_id, rating, comment=None):
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        rating = 0
    if not 1 <= rating <= 5:
        raise AssistanceError(400, "INVALID_RATING", "Choose a rating from 1 to 5 stars.")
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        if req["driver_id"] != driver_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        if req["status"] != COMPLETED or req["payment_status"] != P_COMPLETED:
            raise AssistanceError(409, "NOT_FINISHED", "You can rate the assistance after payment is confirmed.")
        if req["rating"] is not None:
            raise AssistanceError(409, "ALREADY_RATED", "This assistance has already been rated.")
        cursor.execute("UPDATE assistance_requests SET rating=%s, rating_comment=%s, rated_at=%s WHERE id=%s",
                       (rating, (comment or "").strip()[:500] or None, utcnow(), request_id))
        audit.record(audit.ASSISTANCE_RATING, actor_id=driver_id, target_type="assistance_request", target_id=request_id,
                     details={"rating": rating}, cursor=cursor)


def report_problem(driver_id, request_id, text):
    text = (text or "").strip()
    if not 3 <= len(text) <= 500:
        raise AssistanceError(400, "INVALID_REPORT", "Describe the problem in 3 to 500 characters.")
    with transaction() as cursor:
        req = _lock_request(cursor, request_id)
        if req["driver_id"] != driver_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        if req["status"] != COMPLETED:
            raise AssistanceError(409, "NOT_COMPLETED", "Problems can be reported after the journey is completed.")
        cursor.execute("UPDATE assistance_requests SET problem_report=%s, problem_reported_at=%s WHERE id=%s",
                       (text, utcnow(), request_id))
        audit.record(audit.PROBLEM_REPORTED, actor_id=driver_id, target_type="assistance_request",
                     target_id=request_id, cursor=cursor)


def _upsert_location(cursor, user_id, lat, lon, acc):
    cursor.execute("INSERT INTO user_locations (user_id, lat, lon, accuracy_m, updated_at) VALUES (%s,%s,%s,%s,%s) "
                   "ON DUPLICATE KEY UPDATE lat=VALUES(lat), lon=VALUES(lon), accuracy_m=VALUES(accuracy_m), "
                   "updated_at=VALUES(updated_at)", (user_id, lat, lon, acc, utcnow()))


def set_availability(umusare_id, available, lat=None, lon=None, accuracy=None):
    """Go available (shares current position for matching) or offline (position deleted)."""
    with transaction() as cursor:
        prof = _lock_eligible_umusare(cursor, umusare_id) if available else None
        if prof is None:
            cursor.execute("SELECT availability FROM umusare_profiles WHERE user_id=%s FOR UPDATE", (umusare_id,))
            prof = cursor.fetchone()
            if prof is None:
                raise AssistanceError(403, "FORBIDDEN", "Only Umusare can do this.")
        if prof["availability"] == "BUSY":
            if available:                                     # location refresh during an assistance
                _upsert_location(cursor, umusare_id, *_coords(lat, lon, accuracy))
                return "BUSY"
            raise AssistanceError(409, "BUSY", "Finish or hand over your active assistance first.")
        if available:
            _upsert_location(cursor, umusare_id, *_coords(lat, lon, accuracy))
            cursor.execute("UPDATE umusare_profiles SET availability='AVAILABLE' WHERE user_id=%s", (umusare_id,))
            new = "AVAILABLE"
        else:
            cursor.execute("DELETE FROM user_locations WHERE user_id=%s", (umusare_id,))
            cursor.execute("UPDATE umusare_profiles SET availability='OFFLINE' WHERE user_id=%s", (umusare_id,))
            cursor.execute("UPDATE assistance_offers SET status='EXPIRED', responded_at=%s "
                           "WHERE umusare_id=%s AND status='OFFERED'", (utcnow(), umusare_id))
            new = "OFFLINE"
        if prof["availability"] != new:
            audit.record(audit.UMUSARE_AVAILABILITY_CHANGED, actor_id=umusare_id, target_type="user",
                         target_id=umusare_id, details={"availability": new}, cursor=cursor)
        return new


# ---------------------------------------------------------------- read models (privacy-scoped)
def _iso(dt):
    return dt.isoformat(timespec="seconds") + "Z" if dt else None


def _location(cursor, user_id):
    cursor.execute("SELECT lat, lon, accuracy_m, updated_at FROM user_locations WHERE user_id=%s", (user_id,))
    r = cursor.fetchone()
    return {"lat": float(r["lat"]), "lon": float(r["lon"]), "accuracy_m": r["accuracy_m"],
            "updated_at": _iso(r["updated_at"])} if r else None


def _person(cursor, user_id):
    cursor.execute("SELECT u.username, u.phone, c.name AS cooperative FROM users u "
                   "LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
                   "LEFT JOIN cooperatives c ON c.id = m.cooperative_id WHERE u.id=%s", (user_id,))
    return cursor.fetchone() or {}


def assistance_code(request_id):
    return f"AS-{int(request_id):06d}"


def umusare_code(user_id):
    return f"UMS-{int(user_id):05d}"


def _initials(name):
    parts = [p for p in (name or "").replace(".", " ").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def _umusare_identity(cursor, umusare_id):
    """Identity card for the driver. Every claim comes from a database record; nothing is invented."""
    cursor.execute(
        "SELECT u.id, u.username, u.email, u.phone, u.is_active, p.verification_status, p.verified_at, p.availability, "
        "m.id AS membership_id, m.status AS membership_status, m.reviewed_at AS membership_approved_at, "
        "c.name AS cooperative, c.code AS cooperative_code, c.status AS cooperative_status "
        "FROM users u JOIN umusare_profiles p ON p.user_id = u.id "
        "LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
        "LEFT JOIN cooperatives c ON c.id = m.cooperative_id WHERE u.id=%s", (umusare_id,))
    r = cursor.fetchone() or {}
    member_ok = r.get("membership_status") == "APPROVED" and r.get("cooperative_status") == "APPROVED"
    verified = r.get("verification_status") == "VERIFIED"
    return {
        "umusare_id": umusare_code(umusare_id), "name": r.get("username"), "initials": _initials(r.get("username")),
        "photo_url": None,                                  # no profile photos are stored yet
        "email": r.get("email"), "phone": r.get("phone"), "cooperative": r.get("cooperative"),
        "member_id": f"{r['cooperative_code']}-M{r['membership_id']:04d}" if r.get("membership_id") else None,
        "availability": r.get("availability"),
        "verification": {
            "account_active": bool(r.get("is_active")),
            "identity_verified": verified, "verified_at": _iso(r.get("verified_at")),
            "membership_approved": member_ok, "membership_approved_at": _iso(r.get("membership_approved_at")),
            "eligible": bool(r.get("is_active")) and verified and member_ok,
        },
    }


def _pricing_view(cursor):
    return pricing_public(get_pricing(cursor))


def _journey_view(cursor, req):
    """Running distance and ESTIMATED fare during the journey (current pricing, not final)."""
    pricing = get_pricing(cursor)
    km = round(float(req.get("journey_distance_km") or 0), 2)
    fare = compute_fare(km, pricing)
    return {"started_at": _iso(req.get("journey_started_at")), "distance_so_far_km": km,
            "estimated_fare": int(fare), "estimated_fare_label": format_money(fare, pricing["currency"]),
            "rate_label": pricing_public(pricing)["rate_label"]}


def _payment_view(req, payee=None):
    """Final fare (stored snapshot) and payment state of a completed journey."""
    cur = req["fare_currency"] or "RWF"
    return {"status": req["payment_status"], "assistance_id": assistance_code(req["id"]),
            "final_distance_km": float(req["final_distance_km"]) if req["final_distance_km"] is not None else None,
            "amount": int(req["final_fare"]) if req["final_fare"] is not None else None,
            "amount_label": format_money(req["final_fare"], cur), "currency": cur,
            "rate_label": f"{format_money(req['fare_price_per_km'], cur)} / km" if req["fare_price_per_km"] is not None
            else None,
            "base_fee_label": format_money(req["fare_base_fee"], cur) if req["fare_base_fee"] else None,
            "pay_to_phone": req["payment_phone"], "payee": payee,
            "sent_at": _iso(req["payment_sent_at"]), "completed_at": _iso(req["payment_completed_at"]),
            "dispute_note": req["payment_dispute_note"] if req["payment_status"] == P_DISPUTED else None}


def _duration_min(req):
    start, end = req.get("accepted_at") or req.get("created_at"), req.get("journey_ended_at") or req.get("ended_at")
    return max(1, round((end - start).total_seconds() / 60)) if start and end else None


def _fallback_contacts(cursor, cooperative_id):
    if cooperative_id is None:
        return []
    cursor.execute("SELECT u.username, u.phone FROM cooperative_memberships m JOIN users u ON u.id = m.user_id "
                   "WHERE m.cooperative_id=%s AND m.member_role='manager' AND m.status='APPROVED' AND u.is_active=1",
                   (cooperative_id,))
    return [{"name": r["username"], "phone": r["phone"]} for r in cursor.fetchall()]


def driver_view(driver_id, request_id=None):
    """The driver's own request (latest one when ``request_id`` is None)."""
    with transaction() as cursor:
        if request_id is None:
            cursor.execute("SELECT id FROM assistance_requests WHERE driver_id=%s ORDER BY created_at DESC, id DESC LIMIT 1",
                           (driver_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            request_id = row["id"]
        req = _lock_request(cursor, request_id)
        if req["driver_id"] != driver_id:
            raise AssistanceError(404, "NOT_FOUND", "Assistance request not found.")
        _refresh_matching(cursor, req)
        cursor.execute("SELECT COUNT(*) AS n FROM assistance_offers WHERE request_id=%s AND status='OFFERED'", (req["id"],))
        contacted = cursor.fetchone()["n"]
        view = {"id": req["id"], "assistance_id": assistance_code(req["id"]), "status": req["status"],
                "type": req["trigger_source"], "type_label": TYPE_LABELS.get(req["trigger_source"]),
                "created_at": _iso(req["created_at"]), "accepted_at": _iso(req["accepted_at"]),
                "connected_at": _iso(req["connected_at"]),
                "search_radius_km": req["search_radius_km"], "matching_round": req["matching_round"],
                "umusare_contacted": contacted, "sharing": req["status"] in SHARING,
                "approx_area": {"lat": float(req["approx_lat"]), "lon": float(req["approx_lon"])}}
        if req["status"] in SHARING:
            u = _person(cursor, req["accepted_umusare_id"])
            uloc = _location(cursor, req["accepted_umusare_id"])
            dloc = _location(cursor, driver_id) or {"lat": float(req["pickup_lat"]), "lon": float(req["pickup_lon"])}
            distance = round(haversine_km(dloc["lat"], dloc["lon"], uloc["lat"], uloc["lon"]), 2) if uloc else None
            view["umusare"] = {"name": u.get("username"), "phone": u.get("phone"), "cooperative": u.get("cooperative"),
                               "verified": True, "location": uloc, "distance_km": distance,
                               "eta_min": eta_minutes(distance) if req["status"] == ACCEPTED else None,
                               "profile": _umusare_identity(cursor, req["accepted_umusare_id"])}
            view["my_location"] = dloc
            view["pricing"] = _pricing_view(cursor)
            if req["status"] == DRIVER_CONNECTED:
                view["journey"] = _journey_view(cursor, req)
        elif req["status"] in (REQUESTED, MATCHING):
            view["pricing"] = _pricing_view(cursor)
        if req["status"] == COMPLETED and req["final_fare"] is not None:     # journeys completed before fares had none
            ident = _umusare_identity(cursor, req["accepted_umusare_id"]) if req["accepted_umusare_id"] else {}
            payee = {"name": ident.get("name"), "umusare_id": ident.get("umusare_id"),
                     "cooperative": ident.get("cooperative"),
                     "verified": ident.get("verification", {}).get("identity_verified", False)}
            view["payment"] = _payment_view(req, payee)
            view["summary"] = {"assistance_id": assistance_code(req["id"]), "duration_min": _duration_min(req),
                               "rating": req["rating"], "problem_reported": req["problem_report"] is not None}
        if req["status"] == NO_UMUSARE:
            view["fallback_contacts"] = _fallback_contacts(cursor, req["driver_cooperative_id"])
        return view


def umusare_view(umusare_id):
    with transaction() as cursor:
        cursor.execute("SELECT p.verification_status, p.availability, c.name AS cooperative, m.status AS membership "
                       "FROM umusare_profiles p LEFT JOIN cooperative_memberships m ON m.user_id = p.user_id "
                       "LEFT JOIN cooperatives c ON c.id = m.cooperative_id WHERE p.user_id=%s", (umusare_id,))
        profile = cursor.fetchone() or {}
        my_loc = _location(cursor, umusare_id)

        # refresh requests offered to me so expired offers / rematches are current
        cursor.execute("SELECT DISTINCT request_id FROM assistance_offers WHERE umusare_id=%s AND status='OFFERED'",
                       (umusare_id,))
        for row in cursor.fetchall():
            _refresh_matching(cursor, _lock_request(cursor, row["request_id"]))

        pricing = _pricing_view(cursor)
        cursor.execute("SELECT r.id, r.status, r.created_at, r.approx_lat, r.approx_lon, r.trigger_source, o.distance_km "
                       "FROM assistance_offers o JOIN assistance_requests r ON r.id = o.request_id "
                       "WHERE o.umusare_id=%s AND o.status='OFFERED' AND r.status='MATCHING' ORDER BY r.created_at",
                       (umusare_id,))
        # Before acceptance: no name, phone or exact position of the person asking for help.
        incoming = [{"id": r["id"], "assistance_id": assistance_code(r["id"]), "status": r["status"],
                     "reason": "Safety assistance requested", "type": r["trigger_source"],
                     "type_label": TYPE_LABELS.get(r["trigger_source"]), "created_at": _iso(r["created_at"]),
                     "approx_area": {"lat": float(r["approx_lat"]), "lon": float(r["approx_lon"])},
                     "approx_distance": approximate_distance_label(r["distance_km"]),
                     "fare": {"rate_label": pricing["rate_label"],
                              "minimum_label": format_money(pricing["minimum_fare"], pricing["currency"])
                              if pricing["minimum_fare"] else None,
                              "note": "Final fare = actual journey distance x rate, calculated by SafeDrive"}}
                    for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM assistance_requests WHERE accepted_umusare_id=%s AND status IN (%s,%s) LIMIT 1",
                       (umusare_id, *SHARING))
        req = cursor.fetchone()
        active = None
        if req:
            d = _person(cursor, req["driver_id"])
            cursor.execute("SELECT vehicle_plate_number, vehicle_make, vehicle_model FROM driver_profiles WHERE user_id=%s",
                           (req["driver_id"],))
            veh = cursor.fetchone() or {}
            dloc = _location(cursor, req["driver_id"])
            pickup = {"lat": float(req["pickup_lat"]), "lon": float(req["pickup_lon"])}
            target = dloc or pickup
            distance = round(haversine_km(target["lat"], target["lon"], my_loc["lat"], my_loc["lon"]), 2) if my_loc else None
            active = {"id": req["id"], "assistance_id": assistance_code(req["id"]), "status": req["status"],
                      "type": req["trigger_source"], "type_label": TYPE_LABELS.get(req["trigger_source"]),
                      "accepted_at": _iso(req["accepted_at"]),
                      "driver": {"name": d.get("username"), "phone": d.get("phone"),
                                 "vehicle_plate": veh.get("vehicle_plate_number"),
                                 "vehicle": " ".join(x for x in (veh.get("vehicle_make"), veh.get("vehicle_model")) if x)
                                 or None},
                      "pickup": pickup, "driver_location": dloc, "my_location": my_loc, "distance_km": distance,
                      "eta_min": eta_minutes(distance) if req["status"] == ACCEPTED else None,
                      "pricing": pricing,
                      "journey": _journey_view(cursor, req) if req["status"] == DRIVER_CONNECTED else None}

        # Completed journeys of mine still waiting for payment confirmation, plus the latest finished one.
        cursor.execute("SELECT r.*, u.username AS driver_name FROM assistance_requests r JOIN users u ON u.id = r.driver_id "
                       "WHERE r.accepted_umusare_id=%s AND r.status='COMPLETED' AND r.payment_status IN (%s,%s,%s) "
                       "ORDER BY r.journey_ended_at DESC", (umusare_id, P_PENDING, P_SENT, P_DISPUTED))
        payments = [dict(_payment_view(r), id=r["id"], driver_name=r["driver_name"]) for r in cursor.fetchall()]
        cursor.execute("SELECT r.*, u.username AS driver_name FROM assistance_requests r JOIN users u ON u.id = r.driver_id "
                       "WHERE r.accepted_umusare_id=%s AND r.payment_status=%s ORDER BY r.payment_completed_at DESC LIMIT 1",
                       (umusare_id, P_COMPLETED))
        last = cursor.fetchone()
        last_completed = dict(_payment_view(last), id=last["id"], driver_name=last["driver_name"],
                              duration_min=_duration_min(last)) if last else None
        return {"profile": {"verification": profile.get("verification_status"),
                            "availability": profile.get("availability"), "cooperative": profile.get("cooperative"),
                            "membership": profile.get("membership"), "umusare_id": umusare_code(umusare_id)},
                "sharing_location": my_loc is not None, "incoming": incoming, "active": active,
                "payments": payments, "last_completed": last_completed, "pricing": pricing}


def admin_overview(limit=25):
    """Operational list for administrators: no exact coordinates, no AI scores, read-only."""
    with transaction() as cursor:
        cursor.execute(
            "SELECT r.id, r.status, r.trigger_source, r.created_at, r.ended_at, r.approx_lat, r.approx_lon, "
            "r.search_radius_km, d.username AS driver, c.name AS cooperative, u.username AS umusare, "
            "uc.name AS umusare_cooperative "
            "FROM assistance_requests r JOIN users d ON d.id = r.driver_id "
            "LEFT JOIN cooperatives c ON c.id = r.driver_cooperative_id "
            "LEFT JOIN users u ON u.id = r.accepted_umusare_id "
            "LEFT JOIN cooperative_memberships um ON um.user_id = r.accepted_umusare_id "
            "LEFT JOIN cooperatives uc ON uc.id = um.cooperative_id "
            "ORDER BY (r.status IN ('REQUESTED','MATCHING','ACCEPTED','DRIVER_CONNECTED')) DESC, r.created_at DESC "
            "LIMIT %s", (limit,))
        rows = cursor.fetchall()
    for r in rows:
        r["active"] = r["status"] in ACTIVE
        r["approx_area"] = f"{float(r.pop('approx_lat')):.2f}, {float(r.pop('approx_lon')):.2f}"
    return rows
