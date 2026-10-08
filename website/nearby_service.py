# website/nearby_service.py
"""Privacy-safe "nearby verified drivers".

Authoritative location
----------------------
The server's `driver_presence` row is the ONLY location a nearby lookup uses. The browser can
only *propose* an update (update_presence), which is accepted when it is plausible:

* at most one update per PRESENCE_MIN_INTERVAL_S (rejected attempts count too);
* the move from the last accepted position must be physically possible:
  distance <= NEARBY_MAX_SPEED_KMH x elapsed time + CELL_TOLERANCE_KM (grid rounding).

A lookup (nearby) takes no coordinates at all, needs a fresh accepted position and is limited to
LOOKUPS_PER_WINDOW per LOOKUP_WINDOW_S. So a driver cannot query arbitrary places by editing a
request, and cannot sweep an area faster than a vehicle could drive it. (Like any web app, the
server cannot prove a phone's GPS is genuine; these rules bound what a dishonest client can do.)

Privacy
-------
* Only the ~1.1 km grid cell (website.geo.approximate, 2 decimals) is stored or returned; the exact
  position in an update is used transiently to find that cell and is never stored.
* One row per driver (latest only, no history); rows unused for an hour are deleted.
* Others see a driver only if that driver switched visibility on, is SafeDrive-verified (blue
  badge), active, in an active cooperative, and has a fresh position (NEARBY_MAX_AGE_S).
  Switching visibility off hides the driver immediately.
* Returned per driver: name, cooperative, group and the verified flag only (no id, phone,
  email, plate or coordinates). Drivers in the same cell are aggregated into one marker.
"""
from datetime import timedelta

from . import audit
from .assistance_service import AssistanceError, transaction, utcnow
from .badges import DRIVER_VERIFIED_SQL, evaluate, _row
from .geo import APPROX_DECIMALS, approximate, haversine_km, valid_coordinates

NEARBY_RADIUS_KM = 10.0
NEARBY_MAX_AGE_S = 300              # a position older than this is not "current": not shown, not usable
PRESENCE_MIN_INTERVAL_S = 15        # minimum time between location updates (accepted or not)
NEARBY_MAX_SPEED_KMH = 150.0        # faster implied movement = implausible jump, rejected
CELL_TOLERANCE_KM = 2.5             # two ~1.1 km cells can be this far apart without real movement
MAX_ACCURACY_M = 2000.0             # fixes less precise than this are not used
LOOKUP_WINDOW_S = 600
LOOKUPS_PER_WINDOW = 15             # the dashboard needs one per minute
PRESENCE_RETENTION = timedelta(hours=1)
MAX_CELLS = 40
MAX_NAMES_PER_CELL = 8
DRIVER_FIELDS = ("name", "cooperative", "group", "verified")      # documented in /privacy and /terms


def _require_verified_driver(cursor, user_id):
    row = _row(cursor, user_id)
    if not row or row["role"] != "driver":
        raise AssistanceError(403, "FORBIDDEN", "Only drivers can use nearby drivers.")
    if not evaluate(row)["verified"]:
        raise AssistanceError(403, "NOT_VERIFIED", "Nearby drivers are available to SafeDrive-verified drivers. "
                                                   "Complete your verification to use this feature.")


def set_visibility(driver_id, visible):
    """Opt in / out of being seen. Opting out hides the driver at once (lookups filter on this flag)."""
    with transaction() as cursor:
        cursor.execute("SELECT role FROM users WHERE id=%s", (driver_id,))
        row = cursor.fetchone()
        if not row or row["role"] != "driver":
            raise AssistanceError(403, "FORBIDDEN", "Only drivers can change nearby visibility.")
        cursor.execute("INSERT IGNORE INTO driver_profiles (user_id) VALUES (%s)", (driver_id,))
        cursor.execute("UPDATE driver_profiles SET nearby_visibility=%s WHERE user_id=%s", (int(bool(visible)), driver_id))
        audit.record(audit.NEARBY_VISIBILITY_CHANGED, actor_id=driver_id, target_type="user", target_id=driver_id,
                     details={"visible": bool(visible)}, cursor=cursor)
    return bool(visible)


def update_presence(driver_id, lat, lon, accuracy=None):
    """Propose my current position. Stores only the ~1 km cell, and only if the update is plausible."""
    if not valid_coordinates(lat, lon):
        raise AssistanceError(400, "INVALID_LOCATION", "A valid location is required.")
    try:
        acc = float(accuracy) if accuracy not in (None, "") else None
    except (TypeError, ValueError):
        acc = None
    if acc is not None and acc > MAX_ACCURACY_M:
        raise AssistanceError(422, "INACCURATE", "Your location is too imprecise right now. Try again outdoors.")
    cell_lat, cell_lon = approximate(float(lat), float(lon))
    now = utcnow()
    failure = None
    with transaction() as cursor:
        _require_verified_driver(cursor, driver_id)
        cursor.execute("SELECT * FROM driver_presence WHERE user_id=%s FOR UPDATE", (driver_id,))
        prev = cursor.fetchone()
        if prev and prev["updated_at"] < now - PRESENCE_RETENTION:
            prev = None                                   # forgotten: treated like a first position
        if prev and prev["last_attempt_at"] and (now - prev["last_attempt_at"]).total_seconds() < PRESENCE_MIN_INTERVAL_S:
            wait = int(PRESENCE_MIN_INTERVAL_S - (now - prev["last_attempt_at"]).total_seconds()) + 1
            raise AssistanceError(429, "TOO_FREQUENT", f"Location updates are limited; try again in {wait} s.")
        if prev is None:
            cursor.execute("INSERT INTO driver_presence (user_id, approx_lat, approx_lon, updated_at, last_attempt_at) "
                           "VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE approx_lat=VALUES(approx_lat), "
                           "approx_lon=VALUES(approx_lon), updated_at=VALUES(updated_at), "
                           "last_attempt_at=VALUES(last_attempt_at), lookup_window_start=NULL, lookup_count=0",
                           (driver_id, cell_lat, cell_lon, now, now))
        else:
            hours = max((now - prev["updated_at"]).total_seconds(), 0) / 3600.0
            moved = haversine_km(float(prev["approx_lat"]), float(prev["approx_lon"]), cell_lat, cell_lon)
            if moved > NEARBY_MAX_SPEED_KMH * hours + CELL_TOLERANCE_KM:
                # implausible jump: keep the server's position; the attempt still counts for the interval
                cursor.execute("UPDATE driver_presence SET last_attempt_at=%s WHERE user_id=%s", (now, driver_id))
                failure = AssistanceError(409, "IMPLAUSIBLE_LOCATION",
                                          "This location update was not accepted. Nearby drivers keep using your "
                                          "last confirmed area.")
            else:
                cursor.execute("UPDATE driver_presence SET approx_lat=%s, approx_lon=%s, updated_at=%s, last_attempt_at=%s "
                               "WHERE user_id=%s", (cell_lat, cell_lon, now, now, driver_id))
        cursor.execute("DELETE FROM driver_presence WHERE updated_at < %s", (now - PRESENCE_RETENTION,))
    if failure:
        raise failure
    return {"accepted": True, "approx_lat": cell_lat, "approx_lon": cell_lon}


def nearby(driver_id):
    """Nearby verified drivers around MY server-known position. Takes no coordinates."""
    now = utcnow()
    with transaction() as cursor:
        _require_verified_driver(cursor, driver_id)
        cursor.execute("SELECT * FROM driver_presence WHERE user_id=%s FOR UPDATE", (driver_id,))
        me = cursor.fetchone()
        if me is None or me["updated_at"] < now - timedelta(seconds=NEARBY_MAX_AGE_S):
            raise AssistanceError(409, "NO_CURRENT_LOCATION", "Share your current location to see nearby drivers.")
        window_open = me["lookup_window_start"] and (now - me["lookup_window_start"]).total_seconds() < LOOKUP_WINDOW_S
        if window_open and me["lookup_count"] >= LOOKUPS_PER_WINDOW:
            raise AssistanceError(429, "TOO_MANY_LOOKUPS", "Too many nearby-driver requests. Please wait a few minutes.")
        if window_open:
            cursor.execute("UPDATE driver_presence SET lookup_count=lookup_count+1 WHERE user_id=%s", (driver_id,))
        else:
            cursor.execute("UPDATE driver_presence SET lookup_window_start=%s, lookup_count=1 WHERE user_id=%s",
                           (now, driver_id))
        cursor.execute("SELECT nearby_visibility FROM driver_profiles WHERE user_id=%s", (driver_id,))
        visible = bool((cursor.fetchone() or {}).get("nearby_visibility"))
        my_lat, my_lon = float(me["approx_lat"]), float(me["approx_lon"])
        box = NEARBY_RADIUS_KM / 111.0 + 0.02
        cursor.execute(
            "SELECT u.username, pr.approx_lat, pr.approx_lon, c.name AS cooperative, g.name AS group_name "
            "FROM driver_presence pr JOIN users u ON u.id = pr.user_id "
            "JOIN cooperative_memberships m ON m.user_id = u.id JOIN cooperatives c ON c.id = m.cooperative_id "
            "JOIN driver_profiles dp ON dp.user_id = u.id "
            "LEFT JOIN cooperative_groups g ON g.id = m.group_id AND g.status = 'ACTIVE' "
            f"WHERE {DRIVER_VERIFIED_SQL} AND dp.nearby_visibility = 1 AND pr.updated_at >= %s AND u.id <> %s "
            "AND pr.approx_lat BETWEEN %s AND %s AND pr.approx_lon BETWEEN %s AND %s ORDER BY u.username",
            (now - timedelta(seconds=NEARBY_MAX_AGE_S), driver_id, my_lat - box, my_lat + box, my_lon - box,
             my_lon + box))
        rows = cursor.fetchall()
    cells = {}
    for r in rows:
        clat, clon = float(r["approx_lat"]), float(r["approx_lon"])
        km = haversine_km(my_lat, my_lon, clat, clon)
        if km > NEARBY_RADIUS_KM:
            continue
        cell = cells.setdefault((clat, clon), {"lat": clat, "lon": clon, "count": 0, "drivers": [],
                                              "distance_label": _label(km), "_km": km})
        cell["count"] += 1
        if len(cell["drivers"]) < MAX_NAMES_PER_CELL:
            cell["drivers"].append({"name": r["username"], "cooperative": r["cooperative"], "group": r["group_name"],
                                    "verified": True})
    ordered = sorted(cells.values(), key=lambda c: c["_km"])[:MAX_CELLS]
    for c in ordered:
        c.pop("_km")
    return {"you": {"approx_lat": my_lat, "approx_lon": my_lon}, "visible": visible,
            "radius_km": NEARBY_RADIUS_KM, "precision_km": 1.1, "decimals": APPROX_DECIMALS,
            "total": sum(c["count"] for c in ordered), "cells": ordered}


def _label(km):
    if km < 1.5:
        return "Within ~1 km"
    return f"About {round(km)} km away"
