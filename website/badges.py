# website/badges.py
"""The blue SafeDrive "VERIFIED" badge and the separate states it is built from.

These states are deliberately kept apart and are never collapsed into one boolean:
email verified, terms accepted, cooperative membership, manager verification, account
active, AI safety status (not used here at all) and operational availability.

The badge means only "Verified by SafeDrive" on the listed basis. It is not a government,
licensing or medical certification and never states that a driver is safe to drive.
"""
from . import ROLE_ADMIN, ROLE_DRIVER, ROLE_MANAGER, ROLE_UMUSARE
from .assistance_service import transaction
from .legal import PRIVACY_VERSION, TERMS_VERSION


def _row(cursor, user_id):
    cursor.execute(
        "SELECT u.id, u.role, u.is_active, u.phone, u.email_verified_at, u.terms_version, u.privacy_version, "
        "m.member_role, m.status AS membership_status, c.status AS cooperative_status, "
        "dp.verification_status AS driver_status, dp.vehicle_plate_number, "
        "up.verification_status AS umusare_status "
        "FROM users u LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
        "LEFT JOIN cooperatives c ON c.id = m.cooperative_id "
        "LEFT JOIN driver_profiles dp ON dp.user_id = u.id LEFT JOIN umusare_profiles up ON up.user_id = u.id "
        "WHERE u.id=%s", (user_id,))
    return cursor.fetchone()


def evaluate(row):
    """{"verified": bool, "applies": bool, "checks": [(label, ok)], "states": {...}} from one user row."""
    role = row["role"]
    email_ok = row["email_verified_at"] is not None
    member_ok = (row["member_role"] == role and row["membership_status"] == "APPROVED"
                 and row["cooperative_status"] == "APPROVED")
    active = bool(row["is_active"])
    states = {
        "email": "VERIFIED" if email_ok else "NOT VERIFIED",
        "terms": "ACCEPTED" if (row["terms_version"] == TERMS_VERSION and row["privacy_version"] == PRIVACY_VERSION)
        else ("UPDATE REQUIRED" if row["terms_version"] else "NOT ACCEPTED"),
        "cooperative": ("MEMBER" if member_ok else (row["membership_status"] or "NONE")) if role != ROLE_ADMIN else None,
        "account": "ACTIVE" if active else "DEACTIVATED",
    }
    if role == ROLE_ADMIN:
        return {"verified": False, "applies": False, "checks": [], "states": states}
    if role == ROLE_DRIVER:
        manager_ok = row["driver_status"] == "VERIFIED"
        states["manager"] = row["driver_status"] or "PENDING"
        checks = [("Email verified", email_ok), ("Cooperative membership verified", member_ok),
                  ("Vehicle plate number provided", bool(row["vehicle_plate_number"])),
                  ("Required profile information completed", bool(row["vehicle_plate_number"])),
                  ("Verified by Cooperative Manager", manager_ok), ("Account active", active)]
    elif role == ROLE_UMUSARE:
        manager_ok = row["umusare_status"] == "VERIFIED"
        states["manager"] = row["umusare_status"] or "PENDING"
        checks = [("Email verified", email_ok), ("Cooperative membership verified", member_ok),
                  ("Required profile information completed (phone for payments)", bool(row["phone"])),
                  ("Verified by Cooperative Manager", manager_ok), ("Account active", active)]
    elif role == ROLE_MANAGER:
        states["manager"] = "ASSIGNED" if member_ok else "NOT ASSIGNED"
        checks = [("Email verified", email_ok),
                  ("Assigned to a cooperative by a SafeDrive administrator", member_ok), ("Account active", active)]
    else:
        checks = []
    return {"verified": bool(checks) and all(ok for _, ok in checks), "applies": True, "checks": checks,
            "states": states}


def for_user(user_id):
    with transaction() as cursor:
        row = _row(cursor, user_id)
    return evaluate(row) if row else None


# SQL condition (aliases u, m, c, dp) equal to evaluate()["verified"] for drivers; used by nearby search.
DRIVER_VERIFIED_SQL = (
    "u.role = 'driver' AND u.is_active = 1 AND u.email_verified_at IS NOT NULL "
    "AND m.member_role = 'driver' AND m.status = 'APPROVED' AND c.status = 'APPROVED' "
    "AND dp.verification_status = 'VERIFIED' AND dp.vehicle_plate_number IS NOT NULL")
