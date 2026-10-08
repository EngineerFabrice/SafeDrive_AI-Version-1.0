"""Registration, login, CSRF, role authorization and monitoring ownership (MySQL test database)."""
import pytest

import website.monitoring as monitoring_module
from tests.web.conftest import login

pytestmark = pytest.mark.db


# ---------------------------------------------------------------- registration / login
def test_register_driver_creates_pending_membership_profile_and_audit(client, db):
    coop = db.cooperative()
    r = client.post("/register", data={"username": "Kevia Mugabo", "email": "Kevia@Example.com",
                                       "password": "Passw0rd!", "confirm_password": "Passw0rd!",
                                       "role": "driver", "cooperative_id": str(coop), "accept_terms": "1"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify-email")   # next step: email OTP
    user = db.query("SELECT id, email, role, password_hash FROM users")[0]
    assert user["email"] == "kevia@example.com" and user["role"] == "driver"
    assert user["password_hash"].startswith("$2") and "Passw0rd" not in user["password_hash"]
    m = db.query("SELECT member_role, status, cooperative_id FROM cooperative_memberships")[0]
    assert (m["member_role"], m["status"], m["cooperative_id"]) == ("driver", "PENDING", coop)
    assert db.query("SELECT user_id FROM driver_profiles")[0]["user_id"] == user["id"]
    assert "USER_REGISTERED" in [r["action"] for r in db.query("SELECT action FROM audit_logs ORDER BY id")]


def test_register_umusare_starts_unverified(client, db):
    coop = db.cooperative()
    client.post("/register", data={"username": "Umusare One", "email": "u1@example.com",
                                   "password": "Passw0rd!", "confirm_password": "Passw0rd!",
                                   "role": "umusare", "cooperative_id": str(coop), "accept_terms": "1"})
    p = db.query("SELECT verification_status, availability FROM umusare_profiles")[0]
    assert (p["verification_status"], p["availability"]) == ("PENDING", "OFFLINE")


@pytest.mark.parametrize("role, coop_ok", [("admin", True), ("manager", True), ("driver", False)])
def test_register_rejects_privileged_roles_and_missing_cooperative(client, db, role, coop_ok):
    coop = db.cooperative(status="APPROVED")
    client.post("/register", data={"username": "Someone", "email": "s@example.com", "password": "Passw0rd!",
                                   "confirm_password": "Passw0rd!", "role": role,
                                   "cooperative_id": str(coop) if coop_ok else "999999"})
    assert db.query("SELECT COUNT(*) AS n FROM users")[0]["n"] == 0


def test_register_rejects_suspended_cooperative_and_duplicate_email(client, db):
    suspended = db.cooperative(name="Old", code="OLD", status="SUSPENDED")
    form = {"username": "Driver A", "email": "a@example.com", "password": "Passw0rd!",
            "confirm_password": "Passw0rd!", "role": "driver", "cooperative_id": str(suspended)}
    client.post("/register", data=form)
    assert db.query("SELECT COUNT(*) AS n FROM users")[0]["n"] == 0
    ok = db.cooperative()
    db.user("a@example.com", coop_id=ok)
    client.post("/register", data={**form, "cooperative_id": str(ok)})
    assert db.query("SELECT COUNT(*) AS n FROM users")[0]["n"] == 1


@pytest.mark.parametrize("role, landing", [("driver", "/driver-dashboard"), ("umusare", "/umusare-dashboard"),
                                           ("manager", "/manager-dashboard"), ("admin", "/admin-dashboard")])
def test_login_redirects_to_role_dashboard(client, db, role, landing):
    db.user(f"{role}@example.com", role=role, coop_id=db.cooperative())
    r = login(client, f"{role}@example.com")
    assert r.status_code == 302 and r.headers["Location"].endswith(landing)
    assert db.query("SELECT last_login_at FROM users")[0]["last_login_at"] is not None


def test_login_failure_and_inactive_account(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    assert login(client, "d@example.com", "wrong-pass1").status_code == 200       # form shown again
    db.query("UPDATE users SET is_active=0")
    assert login(client, "d@example.com").status_code == 200
    actions = [r["action"] for r in db.query("SELECT action FROM audit_logs")]
    assert actions.count("LOGIN_FAILED") == 2


def test_login_ignores_external_next_redirect(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    r = client.post("/login?next=//evil.example/x", data={"email": "d@example.com", "password": "Passw0rd!"})
    assert r.headers["Location"].endswith("/driver-dashboard")


def test_logout_requires_post(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    login(client, "d@example.com")
    assert client.get("/logout").status_code == 405
    assert client.post("/logout").status_code == 302


# ---------------------------------------------------------------- CSRF
def test_csrf_required_for_state_changing_requests(db):
    from website import create_app
    app = create_app({"BCRYPT_LOG_ROUNDS": 4})          # CSRF enabled (default)
    c = app.test_client()
    assert c.post("/login", data={"email": "x@example.com", "password": "Passw0rd!"}).status_code == 400
    r = c.post("/monitoring/start", headers={"Accept": "application/json"})
    assert r.status_code in (400, 401, 302)


# ---------------------------------------------------------------- role authorization
@pytest.fixture
def fake_engine(monkeypatch):
    """Replace the camera engine so no device is opened during API tests."""
    class Snap:
        impairment_model = None

        def to_dict(self):
            return {"status": "STOPPED"}

    class Store:
        def __init__(self):
            self.sessions = {}

        def list_sessions(self, limit, started_by=None):
            return [s for s in self.sessions.values() if started_by is None or s["started_by"] == started_by]

        def get_session(self, sid):
            return self.sessions.get(sid), []

    class Recorder:
        def __init__(self):
            self.owner, self.store = None, Store()

        def active_owner(self):
            return self.owner

        def start_session(self, started_by=None):
            self.owner = started_by

        def end_session(self, status="COMPLETED"):
            self.owner = None

        def status(self):
            return {"database_ok": True}

    class Engine:
        is_running = False
        impairment_model = None

        def snapshot(self):
            return Snap()

        def start(self):
            Engine.is_running = True

        def stop(self):
            Engine.is_running = False

    engine, recorder = Engine(), Recorder()
    monkeypatch.setattr(monitoring_module, "_engine", engine)
    monkeypatch.setattr(monitoring_module, "_recorder", recorder)
    yield engine, recorder
    Engine.is_running = False


@pytest.mark.parametrize("role, allowed", [("driver", True), ("admin", True),
                                           ("manager", False), ("umusare", False)])
def test_monitoring_api_roles(client, db, fake_engine, role, allowed):
    db.user(f"{role}@example.com", role=role, coop_id=db.cooperative())
    login(client, f"{role}@example.com")
    r = client.get("/monitoring/status")
    assert r.status_code == (200 if allowed else 403)
    assert client.post("/monitoring/start").status_code == (200 if allowed else 403)


def test_monitoring_requires_login(client, db):
    assert client.get("/monitoring/status").status_code == 302


@pytest.mark.parametrize("path, role", [("/admin-dashboard", "driver"), ("/manager-dashboard", "driver"),
                                        ("/driver-dashboard", "umusare"), ("/umusare-dashboard", "manager")])
def test_dashboards_reject_other_roles(client, db, path, role):
    db.user("x@example.com", role=role, coop_id=db.cooperative())
    login(client, "x@example.com")
    assert client.get(path, headers={"Accept": "application/json"}).status_code == 403


def test_only_session_owner_or_admin_can_stop(client, db, fake_engine):
    coop = db.cooperative()
    a = db.user("a@example.com", coop_id=coop)
    db.user("b@example.com", coop_id=coop)
    db.user("admin@example.com", role="admin")
    login(client, "a@example.com")
    assert client.post("/monitoring/start").status_code == 200
    client.post("/logout")

    login(client, "b@example.com")
    assert client.post("/monitoring/start").status_code == 409         # running for driver A
    assert client.post("/monitoring/stop").status_code == 403
    client.post("/logout")

    login(client, "admin@example.com")
    assert client.post("/monitoring/stop").status_code == 200
    actions = [r["action"] for r in db.query("SELECT action FROM audit_logs WHERE actor_user_id=%s", (a,))]
    assert "MONITORING_STARTED" in actions


def test_history_is_private_to_its_driver(client, db, fake_engine):
    _, recorder = fake_engine
    coop = db.cooperative()
    a = db.user("a@example.com", coop_id=coop)
    b = db.user("b@example.com", coop_id=coop)
    sid_a = "11111111-1111-1111-1111-111111111111"
    recorder.store.sessions = {sid_a: {"id": sid_a, "started_by": a, "status": "COMPLETED"},
                               "x": {"id": "x", "started_by": b, "status": "COMPLETED"}}
    login(client, "b@example.com")
    assert [s["started_by"] for s in client.get("/monitoring/history").get_json()["sessions"]] == [b]
    assert client.get(f"/monitoring/history/{sid_a}").status_code == 404   # another driver's session


# ---------------------------------------------------------------- admin actions
def test_admin_assigns_role_and_cooperative(client, db):
    coop = db.cooperative()
    uid = db.user("u@example.com", coop_id=coop, membership="PENDING")
    db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    r = client.post("/admin/update-member", data={"user_id": uid, "role": "umusare", "cooperative_id": coop})
    assert r.status_code == 302
    assert db.query("SELECT role FROM users WHERE id=%s", (uid,))[0]["role"] == "umusare"
    m = db.query("SELECT member_role, status FROM cooperative_memberships WHERE user_id=%s", (uid,))[0]
    assert (m["member_role"], m["status"]) == ("umusare", "APPROVED")
    v = db.query("SELECT verification_status FROM umusare_profiles WHERE user_id=%s", (uid,))[0]
    assert v["verification_status"] == "PENDING"          # role change never auto-verifies an Umusare


def test_admin_cannot_demote_last_admin_or_self(client, db):
    me = db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    client.post("/admin/update-member", data={"user_id": me, "role": "driver", "cooperative_id": ""})
    client.post("/admin/delete-user", data={"user_id": me})
    assert db.query("SELECT role FROM users WHERE id=%s", (me,))[0]["role"] == "admin"


def test_member_roles_require_a_cooperative(client, db):
    uid = db.user("u@example.com", role="admin")
    db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    client.post("/admin/update-member", data={"user_id": uid, "role": "driver", "cooperative_id": ""})
    assert db.query("SELECT role FROM users WHERE id=%s", (uid,))[0]["role"] == "admin"


def test_admin_creates_cooperative_with_validation(client, db):
    db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    client.post("/admin/cooperatives", data={"name": "Koperative XYZ", "code": "xyz-1", "district": "Gasabo"})
    client.post("/admin/cooperatives", data={"name": "Koperative XYZ", "code": "XYZ-2"})   # duplicate name
    client.post("/admin/cooperatives", data={"name": "Bad", "code": "has space"})
    rows = db.query("SELECT name, code FROM cooperatives")
    assert rows == [{"name": "Koperative XYZ", "code": "XYZ-1"}]
