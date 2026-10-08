"""Flask app + MySQL test-database fixtures.

The database is SAFEDRIVE_DB_NAME as set by tests/conftest.py (default
safedrive_ai_v2_test). It is created and migrated with the normal migration
code, and rows are deleted between tests. A name that does not end in "_test"
is refused, so the application databases can never be emptied by a test run.
"""
import pytest

from website.config import db_settings

TABLES = ("driver_presence", "email_otps", "legal_acceptances", "notifications", "messages", "conversation_participants", "conversations", "pricing_settings", "assistance_offers", "assistance_requests", "user_locations", "audit_logs",
          "umusare_profiles", "driver_profiles", "cooperative_memberships", "cooperative_groups", "cooperatives", "monitoring_events", "monitoring_sessions",
          "users")


@pytest.fixture(scope="session")
def database():
    name = db_settings()["database"]
    if not name.endswith("_test"):
        pytest.exit(f"refusing to run database tests against {name!r} (name must end in _test)")
    from website import migrate
    try:
        migrate.create_database()
        migrate.migrate()
    except Exception as exc:  # MySQL not running on this machine
        pytest.skip(f"MySQL not reachable for tests: {type(exc).__name__}: {exc}")
    return name


@pytest.fixture
def db(database):
    """Empty test tables before each test; yields a helper with a cursor factory."""
    from website import get_connection
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        for table in TABLES:
            cur.execute(f"DELETE FROM {table}")
        cur.execute("SET FOREIGN_KEY_CHECKS=1")
        # same default row as migration 0004 (500 RWF per km), so every test starts from it
        cur.execute("INSERT INTO pricing_settings (id, currency, price_per_km, base_fee, minimum_fare, maximum_fare) "
                    "VALUES (1, 'RWF', 500.00, 0.00, 0.00, NULL)")
    conn.commit()
    conn.close()
    return Db()


class Db:
    def query(self, sql, args=()):
        from website import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
            conn.commit()
            return rows
        finally:
            conn.close()

    def cooperative(self, name="Koperative Test", code="KT-01", status="APPROVED"):
        self.query("INSERT INTO cooperatives (name, code, status) VALUES (%s, %s, %s)", (name, code, status))
        return self.query("SELECT id FROM cooperatives WHERE code=%s", (code,))[0]["id"]

    def user(self, email, role="driver", password="Passw0rd!", coop_id=None, membership="APPROVED", email_verified=True):
        from website import bcrypt
        hashed = bcrypt.generate_password_hash(password, rounds=4).decode()
        self.query("INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                   (email.split("@")[0], email, hashed, role))
        uid = self.query("SELECT id FROM users WHERE email=%s", (email,))[0]["id"]
        if email_verified:          # test accounts own their address unless a test says otherwise
            self.query("UPDATE users SET email_verified_at=UTC_TIMESTAMP(3) WHERE id=%s", (uid,))
        if coop_id is not None and role != "admin":
            self.query("INSERT INTO cooperative_memberships (user_id, cooperative_id, member_role, status) "
                       "VALUES (%s, %s, %s, %s)", (uid, coop_id, role, membership))
        return uid


@pytest.fixture
def app():
    from website import create_app
    return create_app({"WTF_CSRF_ENABLED": False, "BCRYPT_LOG_ROUNDS": 4, "TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, email, password="Passw0rd!"):
    return client.post("/login", data={"email": email, "password": password})
