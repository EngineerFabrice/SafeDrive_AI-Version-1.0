# website/vehicle.py
"""Driver vehicle information (driver_profiles.vehicle_*), owned and edited only by the driver.

Plate numbers are normalised to upper case with single spaces between the letter and digit
groups, e.g. "rab123a" / "RAB-123-A" / "RAB  123 A" -> "RAB 123 A". Rwandan plates (RAB 123 A,
RC 123 A, GR 123 A, ...) and similar letter-digit-letter registrations are accepted; anything
else (symbols, digits only, too long) is rejected rather than guessed.
"""
import re

from . import audit
from . import notifications as notes
from .assistance_service import AssistanceError, transaction, utcnow

VEHICLE_TYPES = {"CAR": "Car", "MOTORCYCLE": "Motorcycle", "MINIBUS": "Minibus", "BUS": "Bus",
                 "VAN": "Van", "PICKUP": "Pickup", "TRUCK": "Truck", "OTHER": "Other"}
_PLATE = re.compile(r"^([A-Z]{1,4})([0-9]{1,4})([A-Z]{0,2})$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .'/-]{0,39}$")


def normalize_plate(raw):
    """Canonical plate text, or ValueError for anything that is not a plausible registration."""
    compact = re.sub(r"[\s.-]+", "", str(raw or "")).upper()
    m = _PLATE.match(compact)
    if not m:
        raise ValueError("Enter a valid vehicle plate number, for example RAB 123 A.")
    return " ".join(part for part in m.groups() if part)


def _clean_name(raw, label):
    value = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not value:
        return None
    if not _NAME.match(value):
        raise ValueError(f"{label} may contain letters, numbers, spaces and . ' / - (max 40 characters).")
    return value


def validate(form, require_plate=True):
    """Validated vehicle fields from a form/dict; raises ValueError with a user-facing message."""
    plate_raw = (form.get("vehicle_plate_number") or "").strip()
    if not plate_raw:
        if require_plate:
            raise ValueError("The vehicle plate number is required.")
        plate = None
    else:
        plate = normalize_plate(plate_raw)
    vtype = (form.get("vehicle_type") or "").strip().upper() or None
    if vtype and vtype not in VEHICLE_TYPES:
        raise ValueError("Choose a vehicle type from the list.")
    return {"vehicle_plate_number": plate, "vehicle_make": _clean_name(form.get("vehicle_make"), "Vehicle make"),
            "vehicle_model": _clean_name(form.get("vehicle_model"), "Vehicle model"), "vehicle_type": vtype}


def update_own_vehicle(driver_id, form):
    """The driver updates their OWN vehicle (the id comes from the session, never the form).

    Changing the plate of an already VERIFIED driver sends them back to PENDING, because the
    cooperative has not reviewed the new vehicle; their manager is notified.
    """
    try:
        v = validate(form, require_plate=True)
    except ValueError as exc:
        raise AssistanceError(400, "INVALID_VEHICLE", str(exc))
    with transaction() as cursor:
        cursor.execute("SELECT role FROM users WHERE id=%s", (driver_id,))
        row = cursor.fetchone()
        if not row or row["role"] != "driver":
            raise AssistanceError(403, "FORBIDDEN", "Only drivers have vehicle information.")
        cursor.execute("INSERT IGNORE INTO driver_profiles (user_id) VALUES (%s)", (driver_id,))
        cursor.execute("SELECT verification_status, vehicle_plate_number FROM driver_profiles WHERE user_id=%s FOR UPDATE",
                       (driver_id,))
        prof = cursor.fetchone()
        now = utcnow()
        plate_changed = prof["vehicle_plate_number"] != v["vehicle_plate_number"]
        cursor.execute("UPDATE driver_profiles SET vehicle_plate_number=%s, vehicle_make=%s, vehicle_model=%s, "
                       "vehicle_type=%s, vehicle_updated_at=%s WHERE user_id=%s",
                       (v["vehicle_plate_number"], v["vehicle_make"], v["vehicle_model"], v["vehicle_type"], now,
                        driver_id))
        reverify = plate_changed and prof["vehicle_plate_number"] and prof["verification_status"] == "VERIFIED"
        if reverify:
            cursor.execute("UPDATE driver_profiles SET verification_status='PENDING', verification_note=%s, "
                           "reviewed_at=%s, info_requested_at=NULL WHERE user_id=%s",
                           ("Vehicle plate changed: re-verification required.", now, driver_id))
        cursor.execute("SELECT cooperative_id FROM cooperative_memberships WHERE user_id=%s", (driver_id,))
        coop = (cursor.fetchone() or {}).get("cooperative_id")
        audit.record(audit.VEHICLE_UPDATED, actor_id=driver_id, target_type="user", target_id=driver_id,
                     cooperative_id=coop, details={"plate_changed": bool(plate_changed), "reverify": bool(reverify)},
                     cursor=cursor)
        notes.notify(cursor, driver_id, "VEHICLE_UPDATED", "Vehicle information updated.", "/driver-dashboard")
        if reverify and coop:
            from .cooperative_service import cooperative_manager
            manager = cooperative_manager(cursor, coop)
            if manager:
                notes.notify(cursor, manager["user_id"], notes.VERIFICATION_REQUESTED,
                             "A verified driver changed their vehicle plate: re-verification needed",
                             f"/manager/members/{driver_id}")
    return v, bool(reverify)
