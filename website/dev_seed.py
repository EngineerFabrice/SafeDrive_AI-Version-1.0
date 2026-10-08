# website/dev_seed.py
"""DEVELOPMENT / DEMO accounts for every current role. Never use them in production.

    python -m website.dev_seed

Creates (only when missing) a demo cooperative and one account per role in the
configured database (default safedrive_ai_v2; the legacy database is refused by
website.config). Idempotent: an existing account is left completely untouched
(password, name, role and memberships are never overwritten), so running the
command again creates nothing new.

Passwords go through the same validation and bcrypt hashing as registration.
The command refuses to run when SAFEDRIVE_ENV=production.
"""
import os
import sys
from datetime import datetime, timezone

import pymysql

from . import ROLE_ADMIN, ROLE_DRIVER, ROLE_MANAGER, ROLE_UMUSARE, bcrypt, get_connection
from . import audit
from .auth import validate_password
from .config import db_settings

DEMO_COOPERATIVE = {"name": "SafeDrive Demo Cooperative", "code": "DEMO-01", "district": "Demo"}

# DEVELOPMENT / DEMO CREDENTIALS ONLY (documented in README.md). Do not reuse these passwords anywhere.
DEV_ACCOUNTS = (
    {"role": ROLE_ADMIN, "username": "Demo Admin", "email": "admin@safedrive.ai", "password": "SafeDriveAdmin2026!"},
    {"role": ROLE_MANAGER, "username": "Demo Manager", "email": "manager@safedrive.ai",
     "password": "SafeDriveManager2026!"},
    {"role": ROLE_DRIVER, "username": "Demo Driver", "email": "driver@safedrive.ai", "password": "SafeDriveDriver2026!"},
    {"role": ROLE_UMUSARE, "username": "Demo Umusare", "email": "umusare@safedrive.ai",
     "password": "SafeDriveUmusare2026!"},
)


class SeedRefused(RuntimeError):
    pass


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ensure_cooperative(cursor, created_by):
    cursor.execute("SELECT id, status FROM cooperatives WHERE code=%s", (DEMO_COOPERATIVE["code"],))
    row = cursor.fetchone()
    if row:
        return row["id"], False
    cursor.execute("INSERT INTO cooperatives (name, code, district, status, created_by) "
                   "VALUES (%s, %s, %s, 'APPROVED', %s)",
                   (DEMO_COOPERATIVE["name"], DEMO_COOPERATIVE["code"], DEMO_COOPERATIVE["district"], created_by))
    coop_id = cursor.lastrowid
    audit.record(audit.DEV_SEED, actor_id=created_by, target_type="cooperative", target_id=coop_id,
                 cooperative_id=coop_id, details={"code": DEMO_COOPERATIVE["code"]}, cursor=cursor)
    return coop_id, True


def seed_dev_accounts():
    """Create missing demo accounts; returns {"cooperative": ..., "accounts": [{email, role, status}]}."""
    if os.environ.get("SAFEDRIVE_ENV", "development").strip().lower() == "production":
        raise SeedRefused("development accounts must not be created when SAFEDRIVE_ENV=production")
    for acc in DEV_ACCOUNTS:                       # same rules as registration; never weakened for demo data
        problems = validate_password(acc["password"])
        if problems:
            raise SeedRefused(f"demo password for {acc['email']} fails validation: {problems}")

    results = []
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            existing = {}
            for acc in DEV_ACCOUNTS:
                cursor.execute("SELECT id, role FROM users WHERE email=%s", (acc["email"],))
                existing[acc["email"]] = cursor.fetchone()

            # Admin first: it is recorded as creator of the cooperative and verifier of the demo Umusare.
            admin = next(a for a in DEV_ACCOUNTS if a["role"] == ROLE_ADMIN)
            admin_id = existing[admin["email"]]["id"] if existing[admin["email"]] else None
            if admin_id is None:
                admin_id = _create_user(cursor, admin)
                results.append({"email": admin["email"], "role": ROLE_ADMIN, "status": "created"})
            else:
                results.append({"email": admin["email"], "role": existing[admin["email"]]["role"],
                                "status": "exists (unchanged)"})

            coop_id, coop_created = _ensure_cooperative(cursor, admin_id)

            for acc in DEV_ACCOUNTS:
                if acc["role"] == ROLE_ADMIN:
                    continue
                row = existing[acc["email"]]
                if row:                            # never touch an existing account
                    results.append({"email": acc["email"], "role": row["role"], "status": "exists (unchanged)"})
                    continue
                uid = _create_user(cursor, acc)
                cursor.execute("INSERT INTO cooperative_memberships (user_id, cooperative_id, member_role, status, "
                               "reviewed_by, reviewed_at, review_note) VALUES (%s, %s, %s, 'APPROVED', %s, %s, %s)",
                               (uid, coop_id, acc["role"], admin_id, _utcnow(), "development seed"))
                if acc["role"] == ROLE_MANAGER:
                    # the demo manager is the demo cooperative's contact person (only if none is assigned yet)
                    cursor.execute("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s AND manager_user_id IS NULL",
                                   (uid, coop_id))
                if acc["role"] == ROLE_DRIVER:
                    cursor.execute("INSERT INTO driver_profiles (user_id) VALUES (%s)", (uid,))
                elif acc["role"] == ROLE_UMUSARE:
                    # Demo only: pre-verified by the demo admin so the account is usable immediately; real
                    # Umusare are verified by their cooperative manager. Availability stays OFFLINE.
                    cursor.execute("INSERT INTO umusare_profiles (user_id, verification_status, verified_by, "
                                   "verified_at) VALUES (%s, 'VERIFIED', %s, %s)", (uid, admin_id, _utcnow()))
                results.append({"email": acc["email"], "role": acc["role"], "status": "created"})
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"database": db_settings()["database"], "cooperative": {**DEMO_COOPERATIVE, "id": coop_id,
                                                                   "created": coop_created},
            "accounts": results}


def _create_user(cursor, acc):
    hashed = bcrypt.generate_password_hash(acc["password"]).decode("utf-8")
    cursor.execute("INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                   (acc["username"], acc["email"], hashed, acc["role"]))
    uid = cursor.lastrowid
    audit.record(audit.DEV_SEED, actor_id=uid, target_type="user", target_id=uid,
                 details={"role": acc["role"], "seed": "development"}, cursor=cursor)
    return uid


if __name__ == "__main__":
    from . import migrate
    try:
        migrate.migrate()                          # additive only; makes sure the v2 schema exists
        result = seed_dev_accounts()
    except (SeedRefused, pymysql.err.MySQLError) as exc:
        sys.exit(f"Seed refused/failed: {exc}")
    print(f"Database: {result['database']}")
    c = result["cooperative"]
    print(f"Cooperative: {c['name']} ({c['code']}) - {'created' if c['created'] else 'already present'}")
    for a in result["accounts"]:
        print(f"  {a['role']:<8} {a['email']:<24} {a['status']}")
    print("DEVELOPMENT / DEMO credentials only - see README.md. Never use them in production.")
