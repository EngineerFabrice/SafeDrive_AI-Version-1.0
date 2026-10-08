"""Cooperative manager ownership, member verification workflow and cooperative isolation (MySQL test database)."""
import json

import pytest

from tests.web.conftest import login

pytestmark = pytest.mark.db


class World:
    """Two cooperatives, each with an assigned manager, a driver and an Umusare (all PENDING), plus an admin."""

    def __init__(self, app, db):
        self.app, self.db = app, db
        self.a = db.cooperative("Coop A", "CA")
        self.b = db.cooperative("Coop B", "CB")
        self.admin = db.user("admin@example.com", role="admin")
        self.mgr_a = db.user("mgr.a@example.com", role="manager", coop_id=self.a)
        self.mgr_b = db.user("mgr.b@example.com", role="manager", coop_id=self.b)
        db.query("UPDATE users SET phone='+250788000111' WHERE id=%s", (self.mgr_a,))
        db.query("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s", (self.mgr_a, self.a))
        db.query("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s", (self.mgr_b, self.b))
        self.drv_a = db.user("drv.a@example.com", coop_id=self.a, membership="PENDING")
        self.drv_b = db.user("drv.b@example.com", coop_id=self.b, membership="PENDING")
        self.ums_a = db.user("ums.a@example.com", role="umusare", coop_id=self.a, membership="PENDING")
        self.ums_b = db.user("ums.b@example.com", role="umusare", coop_id=self.b, membership="PENDING")
        for uid in (self.drv_a, self.drv_b):         # drivers need a plate before they can be verified
            db.query("INSERT INTO driver_profiles (user_id, vehicle_plate_number) VALUES (%s, 'RAB 123 A')", (uid,))
        for uid in (self.ums_a, self.ums_b):
            db.query("INSERT INTO umusare_profiles (user_id) VALUES (%s)", (uid,))

    def client(self, email):
        c = self.app.test_client()
        login(c, email)
        return c

    def status(self, uid, table="driver_profiles"):
        return self.db.query(f"SELECT * FROM {table} WHERE user_id=%s", (uid,))[0]


@pytest.fixture
def w(app, db):
    return World(app, db)


def review(client, uid, action, note=None):
    return client.post(f"/manager/members/{uid}/review", data={"action": action, "note": note or ""})


def actions(db):
    return [r["action"] for r in db.query("SELECT action FROM audit_logs ORDER BY id")]


# ---------------------------------------------------------------- registration starts PENDING
@pytest.mark.parametrize("role, table", [("driver", "driver_profiles"), ("umusare", "umusare_profiles")])
def test_new_member_starts_pending_and_manager_is_notified(client, w, role, table):
    from website import mailer
    mailer.OUTBOX.clear()
    client.post("/register", data={"username": "New Member", "email": "new@example.com", "password": "Passw0rd!",
                                   "confirm_password": "Passw0rd!", "role": role, "cooperative_id": str(w.a),
                                   "accept_terms": "1"})
    uid = w.db.query("SELECT id FROM users WHERE email='new@example.com'")[0]["id"]
    assert w.status(uid, table)["verification_status"] == "PENDING"
    assert w.db.query("SELECT COUNT(*) AS n FROM notifications")[0]["n"] == 0     # manager is told after email OTP
    code = mailer.OUTBOX[-1]["body"].split("code is: ")[1][:6]
    client.post("/verify-email", data={"code": code})
    assert "USER_VERIFICATION_REQUESTED" in actions(w.db)
    note = w.db.query("SELECT user_id, kind, title FROM notifications")
    assert [(n["user_id"], n["kind"]) for n in note] == [(w.mgr_a, "VERIFICATION_REQUESTED")]   # own manager only
    assert "New Member" in note[0]["title"]


# ---------------------------------------------------------------- manager verification
def test_manager_verifies_own_member_and_records_verifier_time_and_cooperative(w):
    r = review(w.client("mgr.a@example.com"), w.drv_a, "VERIFY")
    assert r.status_code == 302
    p = w.status(w.drv_a)
    assert p["verification_status"] == "VERIFIED" and p["verified_by"] == w.mgr_a
    assert p["verified_at"] is not None and p["verified_cooperative_id"] == w.a
    m = w.db.query("SELECT status, reviewed_by FROM cooperative_memberships WHERE user_id=%s", (w.drv_a,))[0]
    assert (m["status"], m["reviewed_by"]) == ("APPROVED", w.mgr_a)
    row = w.db.query("SELECT actor_user_id, cooperative_id FROM audit_logs WHERE action='USER_VERIFIED'")[0]
    assert (row["actor_user_id"], row["cooperative_id"]) == (w.mgr_a, w.a)
    assert w.db.query("SELECT kind FROM notifications WHERE user_id=%s", (w.drv_a,))[0]["kind"] == "VERIFICATION_APPROVED"


def test_manager_verifies_umusare_who_then_becomes_eligible_to_go_available(w):
    review(w.client("mgr.a@example.com"), w.ums_a, "VERIFY")
    assert w.status(w.ums_a, "umusare_profiles")["verification_status"] == "VERIFIED"
    u = w.client("ums.a@example.com")
    r = u.post("/assistance/umusare/availability", data=json.dumps({"available": True, "lat": -1.95, "lon": 30.06}),
               content_type="application/json")
    assert r.status_code == 200 and r.get_json()["availability"] == "AVAILABLE"


def test_manager_cannot_verify_or_view_another_cooperatives_member(w):
    mgr = w.client("mgr.a@example.com")
    for uid in (w.drv_b, w.ums_b):
        assert review(mgr, uid, "VERIFY").status_code == 404
        assert mgr.get(f"/manager/members/{uid}").status_code == 404
        assert mgr.post(f"/manager/members/{uid}/active", data={"active": "0"}).status_code == 404
    assert w.status(w.drv_b)["verification_status"] == "PENDING"
    assert w.db.query("SELECT is_active FROM users WHERE id=%s", (w.drv_b,))[0]["is_active"] == 1
    assert "USER_VERIFIED" not in actions(w.db)


def test_manager_cannot_review_managers_admins_or_unknown_ids(w):
    mgr = w.client("mgr.a@example.com")
    for uid in (w.mgr_b, w.admin, w.mgr_a, 999999):
        assert review(mgr, uid, "VERIFY").status_code == 404


@pytest.mark.parametrize("email", ["drv.a@example.com", "ums.a@example.com"])
def test_members_cannot_verify_anyone(w, email):
    c = w.client(email)
    target = w.ums_a if email.startswith("drv") else w.drv_a
    assert review(c, target, "VERIFY").status_code == 403
    assert review(c, w.db.query("SELECT id FROM users WHERE email=%s", (email,))[0]["id"], "VERIFY").status_code == 403
    assert c.get(f"/manager/members/{target}").status_code == 403
    assert w.status(w.drv_a)["verification_status"] == "PENDING"


def test_verification_state_cannot_be_set_from_request_fields(w):
    mgr = w.client("mgr.a@example.com")
    r = mgr.post(f"/manager/members/{w.drv_a}/review",
                 data={"action": "VERIFIED", "verification_status": "VERIFIED", "cooperative_id": str(w.b)})
    assert r.status_code == 302 and w.status(w.drv_a)["verification_status"] == "PENDING"


def test_rejection_requires_reason_stores_it_and_informs_member(w):
    mgr = w.client("mgr.a@example.com")
    review(mgr, w.drv_a, "REJECT")                                              # no reason: refused
    assert w.status(w.drv_a)["verification_status"] == "PENDING"
    review(mgr, w.drv_a, "REJECT", "Name does not match cooperative register")
    p = w.status(w.drv_a)
    assert p["verification_status"] == "REJECTED" and p["verification_note"] == "Name does not match cooperative register"
    assert "USER_VERIFICATION_REJECTED" in actions(w.db)
    html = w.client("drv.a@example.com").get("/driver-dashboard").get_data(as_text=True)
    assert "Verification was not approved" in html and "Name does not match cooperative register" in html
    assert "Contact Cooperative Manager" in html and "mgr.a@example.com" in html
    details = " ".join(str(r["details"]) for r in w.db.query("SELECT details FROM audit_logs"))
    assert "Name does not match" not in details                                # reasons are not copied into the audit log


def test_request_more_information_keeps_pending_and_shows_message(w):
    review(w.client("mgr.a@example.com"), w.ums_a, "REQUEST_INFO", "Please add your phone number")
    p = w.status(w.ums_a, "umusare_profiles")
    assert p["verification_status"] == "PENDING" and p["info_requested_at"] is not None
    html = w.client("ums.a@example.com").get("/umusare-dashboard").get_data(as_text=True)
    assert "More information requested" in html and "Please add your phone number" in html
    assert "USER_VERIFICATION_INFO_REQUESTED" in actions(w.db)


def test_suspending_an_available_umusare_takes_them_offline(w):
    w.db.query("UPDATE umusare_profiles SET verification_status='VERIFIED', availability='AVAILABLE' WHERE user_id=%s",
               (w.ums_a,))
    w.db.query("INSERT INTO user_locations (user_id, lat, lon, updated_at) VALUES (%s,-1.95,30.06,UTC_TIMESTAMP(3))",
               (w.ums_a,))
    review(w.client("mgr.a@example.com"), w.ums_a, "SUSPEND", "Under review")
    p = w.status(w.ums_a, "umusare_profiles")
    assert (p["verification_status"], p["availability"]) == ("SUSPENDED", "OFFLINE")
    assert w.db.query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s", (w.ums_a,))[0]["n"] == 0
    assert "USER_SUSPENDED" in actions(w.db)


def test_invalid_transition_is_refused(w):
    mgr = w.client("mgr.a@example.com")
    review(mgr, w.drv_a, "VERIFY")
    review(mgr, w.drv_a, "VERIFY")                                             # already verified
    review(mgr, w.drv_a, "REQUEST_INFO", "x")                                  # not from VERIFIED
    assert w.status(w.drv_a)["verification_status"] == "VERIFIED"
    assert actions(w.db).count("USER_VERIFIED") == 1


def test_manager_can_deactivate_own_member(w):
    w.client("mgr.a@example.com").post(f"/manager/members/{w.drv_a}/active", data={"active": "0"})
    assert w.db.query("SELECT is_active FROM users WHERE id=%s", (w.drv_a,))[0]["is_active"] == 0


# ---------------------------------------------------------------- unverified / verified experience
def test_unverified_driver_sees_own_manager_contact_from_database(w):
    html = w.client("drv.a@example.com").get("/driver-dashboard").get_data(as_text=True)
    assert "Your account is awaiting cooperative verification." in html
    assert "Please contact your cooperative manager by phone or email to complete verification." in html
    assert "mgr.a@example.com" in html and "+250788000111" in html and 'href="tel:+250788000111"' in html
    assert "Call Manager" in html and "Email Manager" in html and "Chat with Manager" in html
    assert "mgr.b@example.com" not in html                                     # never another cooperative's manager
    assert "Start monitoring" in html                                          # safety features stay available


def test_unverified_umusare_is_told_why_and_cannot_go_available(w):
    u = w.client("ums.a@example.com")
    html = u.get("/umusare-dashboard").get_data(as_text=True)
    assert "Waiting for verification" in html and "mgr.a@example.com" in html and 'id="btn-online"' not in html
    r = u.post("/assistance/umusare/availability", data=json.dumps({"available": True, "lat": -1.95, "lon": 30.06}),
               content_type="application/json")
    assert r.status_code == 403 and "verified" in r.get_json()["error"].lower()


def test_verified_driver_sees_verified_badge_and_no_waiting_panel(w):
    review(w.client("mgr.a@example.com"), w.drv_a, "VERIFY")
    html = w.client("drv.a@example.com").get("/driver-dashboard").get_data(as_text=True)
    assert "✓ Cooperative verified" in html and "awaiting cooperative verification" not in html


def test_member_without_manager_gets_clear_explanation(w):
    w.db.query("UPDATE cooperative_memberships SET status='REVOKED' WHERE user_id=%s", (w.mgr_a,))
    html = w.client("drv.a@example.com").get("/driver-dashboard").get_data(as_text=True)
    assert "no manager assigned yet" in html and "mgr.a@example.com" not in html


# ---------------------------------------------------------------- manager dashboard isolation
def test_manager_dashboard_shows_only_own_cooperative(w):
    html = w.client("mgr.a@example.com").get("/manager-dashboard").get_data(as_text=True)
    assert "drv.a@example.com" in html and "ums.a@example.com" in html and "CA" in html
    assert "drv.b@example.com" not in html and "ums.b@example.com" not in html and "Coop B" not in html
    assert "Awaiting your review" in html and "Members" in html


def test_manager_dashboard_never_shows_coordinates(w, app):
    w.db.query("UPDATE umusare_profiles SET verification_status='VERIFIED', availability='AVAILABLE' WHERE user_id=%s",
               (w.ums_a,))
    w.db.query("INSERT INTO user_locations (user_id, lat, lon, updated_at) VALUES (%s,-1.944123,30.061987,UTC_TIMESTAMP(3))",
               (w.ums_a,))
    html = w.client("mgr.a@example.com").get("/manager-dashboard").get_data(as_text=True)
    assert "AVAILABLE" in html and "-1.944123" not in html and "30.061987" not in html


# ---------------------------------------------------------------- admin: global authority
def test_admin_can_review_any_cooperatives_member(w):
    admin = w.client("admin@example.com")
    assert admin.get(f"/manager/members/{w.drv_b}").status_code == 200
    review(admin, w.drv_b, "VERIFY")
    assert w.status(w.drv_b)["verified_by"] == w.admin


def test_admin_pages_render_and_are_admin_only(w):
    admin = w.client("admin@example.com")
    assert "Coop A" in admin.get("/admin/cooperatives").get_data(as_text=True)
    detail = admin.get(f"/admin/cooperatives/{w.b}").get_data(as_text=True)
    assert "drv.b@example.com" in detail and "mgr.b@example.com" in detail
    queue = admin.get("/admin/verification?status=PENDING").get_data(as_text=True)
    assert "drv.a@example.com" in queue and "drv.b@example.com" in queue
    for email in ("mgr.a@example.com", "drv.a@example.com", "ums.a@example.com"):
        c = w.client(email)
        for path in ("/admin/cooperatives", f"/admin/cooperatives/{w.a}", "/admin/verification"):
            assert c.get(path).status_code == 403, (email, path)
        assert c.post(f"/admin/cooperatives/{w.a}/manager", data={"manager_id": str(w.mgr_b)}).status_code == 403


def test_admin_assigns_and_changes_manager(w):
    new = w.db.user("mgr.new@example.com", role="manager")
    admin = w.client("admin@example.com")
    admin.post(f"/admin/cooperatives/{w.a}/manager", data={"manager_id": str(new)})
    assert w.db.query("SELECT manager_user_id FROM cooperatives WHERE id=%s", (w.a,))[0]["manager_user_id"] == new
    old = w.db.query("SELECT status FROM cooperative_memberships WHERE user_id=%s", (w.mgr_a,))[0]["status"]
    assert old == "REVOKED" and "MANAGER_CHANGED" in actions(w.db)
    assert "No active cooperative" in w.client("mgr.a@example.com").get("/manager-dashboard").get_data(as_text=True)
    assert review(w.client("mgr.a@example.com"), w.drv_a, "VERIFY").status_code == 404     # access is gone
    assert "drv.a@example.com" in w.client("mgr.new@example.com").get("/manager-dashboard").get_data(as_text=True)


def test_assigning_a_manager_moves_them_to_exactly_one_cooperative(w):
    admin = w.client("admin@example.com")
    w.db.query("UPDATE cooperatives SET manager_user_id=NULL WHERE id=%s", (w.b,))
    w.db.query("UPDATE cooperative_memberships SET status='REVOKED' WHERE user_id=%s", (w.mgr_b,))
    admin.post(f"/admin/cooperatives/{w.b}/manager", data={"manager_id": str(w.mgr_a)})
    rows = w.db.query("SELECT id, manager_user_id FROM cooperatives ORDER BY id")
    assert [r["manager_user_id"] for r in rows] == [None, w.mgr_a]
    m = w.db.query("SELECT cooperative_id, status FROM cooperative_memberships WHERE user_id=%s", (w.mgr_a,))
    assert m == [{"cooperative_id": w.b, "status": "APPROVED"}]
    assert "MANAGER_ASSIGNED" in actions(w.db)


def test_only_manager_accounts_can_be_assigned(w):
    w.client("admin@example.com").post(f"/admin/cooperatives/{w.a}/manager", data={"manager_id": str(w.drv_a)})
    assert w.db.query("SELECT manager_user_id FROM cooperatives WHERE id=%s", (w.a,))[0]["manager_user_id"] == w.mgr_a


def test_admin_edits_and_deactivates_cooperative(w):
    w.client("admin@example.com").post(f"/admin/cooperatives/{w.a}/edit",
                                       data={"name": "Coop A Renamed", "district": "Gasabo", "status": "SUSPENDED"})
    c = w.db.query("SELECT name, district, status FROM cooperatives WHERE id=%s", (w.a,))[0]
    assert (c["name"], c["district"], c["status"]) == ("Coop A Renamed", "Gasabo", "SUSPENDED")
    assert "COOPERATIVE_UPDATED" in actions(w.db)
