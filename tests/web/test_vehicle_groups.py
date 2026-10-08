"""Vehicle plates, groups under cooperatives, cooperative creation rules and the redesigned driver dashboard."""
import json

import pymysql
import pytest

from tests.web.conftest import login
from tests.web.journey_helpers import Journey
from tests.web.test_cooperative_verification import World

pytestmark = pytest.mark.db


@pytest.fixture
def w(app, db):
    return World(app, db)


def actions(db):
    return [r["action"] for r in db.query("SELECT action FROM audit_logs ORDER BY id")]


# ---------------------------------------------------------------- plate validation (pure)
@pytest.mark.parametrize("raw, expected", [
    ("RAB 123 A", "RAB 123 A"), ("rab123a", "RAB 123 A"), ("RAB-123-A", "RAB 123 A"), ("  rab  123   a ", "RAB 123 A"),
    ("RC 456 B", "RC 456 B"), ("GR 012 A", "GR 012 A"), ("RNP 77 C", "RNP 77 C"), ("IT 123", "IT 123"),
])
def test_plate_normalization(raw, expected):
    from website.vehicle import normalize_plate
    assert normalize_plate(raw) == expected


@pytest.mark.parametrize("raw", ["", "123", "ABCDE 123 A", "RAB 12345 A", "RAB 123 ABC", "R@B 123", "<script>", "RAB 123 A; DROP"])
def test_invalid_plates_are_rejected(raw):
    from website.vehicle import normalize_plate
    with pytest.raises(ValueError):
        normalize_plate(raw)


# ---------------------------------------------------------------- driver vehicle
def test_driver_updates_own_vehicle_normalized(w):
    c = w.client("drv.a@example.com")
    c.post("/driver/vehicle", data={"vehicle_plate_number": "rad 456 b", "vehicle_make": "Toyota", "vehicle_model": "Prius",
                                    "vehicle_type": "CAR"})
    p = w.status(w.drv_a)
    assert (p["vehicle_plate_number"], p["vehicle_make"], p["vehicle_type"]) == ("RAD 456 B", "Toyota", "CAR")
    assert "VEHICLE_UPDATED" in actions(w.db)


def test_invalid_vehicle_is_rejected_and_nothing_changes(w):
    c = w.client("drv.a@example.com")
    r = c.post("/driver/vehicle", data={"vehicle_plate_number": "not a plate!!"}, follow_redirects=True)
    assert "valid vehicle plate number" in r.get_data(as_text=True)
    c.post("/driver/vehicle", data={"vehicle_plate_number": "", "vehicle_make": "X"})
    assert w.status(w.drv_a)["vehicle_plate_number"] == "RAB 123 A"


def test_vehicle_cannot_be_edited_for_someone_else_or_by_other_roles(w):
    c = w.client("drv.a@example.com")
    c.post("/driver/vehicle", data={"vehicle_plate_number": "RAC 111 A", "user_id": str(w.drv_b)})
    assert w.status(w.drv_b)["vehicle_plate_number"] == "RAB 123 A"            # only the signed-in driver's row
    for email in ("mgr.a@example.com", "ums.a@example.com", "admin@example.com"):
        assert w.client(email).post("/driver/vehicle", data={"vehicle_plate_number": "RAC 111 A"}).status_code == 403


def test_changing_plate_after_verification_requires_reverification(w):
    w.client("mgr.a@example.com").post(f"/manager/members/{w.drv_a}/review", data={"action": "VERIFY"})
    assert w.status(w.drv_a)["verification_status"] == "VERIFIED"
    w.client("drv.a@example.com").post("/driver/vehicle", data={"vehicle_plate_number": "RAE 999 Z"})
    assert w.status(w.drv_a)["verification_status"] == "PENDING"
    kinds = [r["kind"] for r in w.db.query("SELECT kind FROM notifications WHERE user_id=%s", (w.mgr_a,))]
    assert "VERIFICATION_REQUESTED" in kinds


def test_missing_plate_prevents_verification(w):
    w.db.query("UPDATE driver_profiles SET vehicle_plate_number=NULL WHERE user_id=%s", (w.drv_a,))
    r = w.client("mgr.a@example.com").post(f"/manager/members/{w.drv_a}/review", data={"action": "VERIFY"},
                                           follow_redirects=True)
    assert "vehicle plate number is missing" in r.get_data(as_text=True)
    assert w.status(w.drv_a)["verification_status"] == "PENDING"


def test_missing_plate_shows_vehicle_required_card(w):
    w.db.query("UPDATE driver_profiles SET vehicle_plate_number=NULL WHERE user_id=%s", (w.drv_a,))
    html = w.client("drv.a@example.com").get("/driver-dashboard").get_data(as_text=True)
    assert "VEHICLE INFORMATION REQUIRED" in html and "Add your vehicle plate number before completing verification." in html
    assert "ADD VEHICLE INFORMATION" in html


def test_manager_and_admin_see_plate_but_other_cooperatives_do_not(w):
    assert "RAB 123 A" in w.client("mgr.a@example.com").get(f"/manager/members/{w.drv_a}").get_data(as_text=True)
    assert "RAB 123 A" in w.client("admin@example.com").get("/admin/drivers").get_data(as_text=True)
    assert w.client("mgr.b@example.com").get(f"/manager/members/{w.drv_a}").status_code == 404


def test_accepted_umusare_sees_driver_plate(app, db):
    j = Journey(app, db)
    db.query("INSERT INTO driver_profiles (user_id, vehicle_plate_number, vehicle_make) VALUES (%s,'RAB 123 A','Toyota')",
             (j.driver_id,))
    j.request()
    before = j.u.get("/assistance/umusare/status").get_json()
    assert "RAB 123 A" not in json.dumps(before)                              # not before acceptance
    j.accept()
    active = j.u.get("/assistance/umusare/status").get_json()["active"]
    assert active["driver"]["vehicle_plate"] == "RAB 123 A" and active["driver"]["vehicle"] == "Toyota"
    kinds = [r["kind"] for r in db.query("SELECT kind FROM notifications WHERE user_id=%s", (j.driver_id,))]
    assert "ASSISTANCE_ACCEPTED" in kinds


# ---------------------------------------------------------------- groups
def create_group(client, name, coop_id=None):
    data = {"name": name}
    if coop_id is not None:
        data["cooperative_id"] = str(coop_id)
    return client.post("/manager/groups", data=data)


def group_id(db, name):
    return db.query("SELECT id FROM cooperative_groups WHERE name=%s", (name,))[0]["id"]


def test_manager_creates_group_only_in_own_cooperative(w):
    create_group(w.client("mgr.a@example.com"), "Kigali Central Group", coop_id=w.b)   # cooperative_id is ignored
    g = w.db.query("SELECT cooperative_id, status, created_by FROM cooperative_groups")[0]
    assert (g["cooperative_id"], g["status"], g["created_by"]) == (w.a, "ACTIVE", w.mgr_a)
    assert "GROUP_CREATED" in actions(w.db)


def test_group_names_are_validated_and_unique_per_cooperative(w):
    mgr = w.client("mgr.a@example.com")
    create_group(mgr, "Night Drivers")
    create_group(mgr, "Night Drivers")
    create_group(mgr, "<b>")
    create_group(w.client("mgr.b@example.com"), "Night Drivers")                    # same name in another cooperative is fine
    assert w.db.query("SELECT COUNT(*) AS n FROM cooperative_groups")[0]["n"] == 2


def test_manager_a_cannot_view_or_manage_cooperative_b_groups(w):
    create_group(w.client("mgr.b@example.com"), "Gasabo Group")
    gid = group_id(w.db, "Gasabo Group")
    mgr = w.client("mgr.a@example.com")
    assert mgr.get(f"/manager/groups/{gid}").status_code == 404
    assert mgr.post(f"/manager/groups/{gid}/edit", data={"name": "Hijacked", "status": "INACTIVE"}).status_code == 404
    assert mgr.post(f"/manager/groups/{gid}/members", data={"user_id": str(w.drv_a)}).status_code == 404
    assert mgr.post(f"/manager/groups/{gid}/members/{w.drv_b}/remove").status_code == 404
    g = w.db.query("SELECT name, status FROM cooperative_groups WHERE id=%s", (gid,))[0]
    assert (g["name"], g["status"]) == ("Gasabo Group", "ACTIVE")
    assert "Gasabo Group" not in mgr.get("/manager-dashboard").get_data(as_text=True)


def test_members_of_another_cooperative_cannot_join_a_group(w):
    create_group(w.client("mgr.a@example.com"), "Kicukiro Group")
    gid = group_id(w.db, "Kicukiro Group")
    assert w.client("mgr.a@example.com").post(f"/manager/groups/{gid}/members", data={"user_id": str(w.drv_b)}).status_code == 404
    assert w.client("admin@example.com").post(f"/manager/groups/{gid}/members", data={"user_id": str(w.drv_b)}).status_code == 404
    with pytest.raises(pymysql.err.IntegrityError):                              # the database refuses it too
        w.db.query("UPDATE cooperative_memberships SET group_id=%s WHERE user_id=%s", (gid, w.drv_b))
    assert w.db.query("SELECT group_id FROM cooperative_memberships WHERE user_id=%s", (w.drv_b,))[0]["group_id"] is None


def test_assign_move_and_remove_group_members(w):
    mgr = w.client("mgr.a@example.com")
    create_group(mgr, "Group One"); create_group(mgr, "Group Two")
    one, two = group_id(w.db, "Group One"), group_id(w.db, "Group Two")
    mgr.post(f"/manager/groups/{one}/members", data={"user_id": str(w.drv_a)})
    mgr.post(f"/manager/members/{w.ums_a}/group", data={"group_id": str(one)})
    gid = lambda uid: w.db.query("SELECT group_id FROM cooperative_memberships WHERE user_id=%s", (uid,))[0]["group_id"]
    assert gid(w.drv_a) == one and gid(w.ums_a) == one
    mgr.post(f"/manager/groups/{two}/members", data={"user_id": str(w.drv_a)})          # move
    assert gid(w.drv_a) == two
    mgr.post(f"/manager/groups/{two}/members/{w.drv_a}/remove")
    assert gid(w.drv_a) is None
    page = mgr.get(f"/manager/groups/{one}").get_data(as_text=True)
    assert "ums.a" in page
    assert {"GROUP_MEMBER_ADDED", "GROUP_MEMBER_REMOVED"} <= set(actions(w.db))


def test_inactive_group_accepts_no_members(w):
    mgr = w.client("mgr.a@example.com")
    create_group(mgr, "Old Group")
    gid = group_id(w.db, "Old Group")
    mgr.post(f"/manager/groups/{gid}/edit", data={"name": "Old Group", "status": "INACTIVE"})
    mgr.post(f"/manager/groups/{gid}/members", data={"user_id": str(w.drv_a)})
    assert w.db.query("SELECT group_id FROM cooperative_memberships WHERE user_id=%s", (w.drv_a,))[0]["group_id"] is None


def test_admin_manages_groups_in_any_cooperative(w):
    admin = w.client("admin@example.com")
    create_group(admin, "Admin Made", coop_id=w.b)
    gid = group_id(w.db, "Admin Made")
    assert w.db.query("SELECT cooperative_id FROM cooperative_groups WHERE id=%s", (gid,))[0]["cooperative_id"] == w.b
    admin.post(f"/manager/groups/{gid}/members", data={"user_id": str(w.drv_b)})
    assert w.db.query("SELECT group_id FROM cooperative_memberships WHERE user_id=%s", (w.drv_b,))[0]["group_id"] == gid
    assert "Admin Made" in admin.get(f"/admin/cooperatives/{w.b}").get_data(as_text=True)


@pytest.mark.parametrize("email", ["drv.a@example.com", "ums.a@example.com"])
def test_members_cannot_manage_groups(w, email):
    c = w.client(email)
    assert create_group(c, "Rogue Group").status_code == 403
    assert w.db.query("SELECT COUNT(*) AS n FROM cooperative_groups")[0]["n"] == 0


def test_registration_offers_only_own_cooperative_groups(w, client):
    from website import mailer
    create_group(w.client("mgr.b@example.com"), "B Group")
    b_group = group_id(w.db, "B Group")
    client.post("/register", data={"username": "Jean Paul", "email": "jp@example.com", "password": "Passw0rd!",
                                   "confirm_password": "Passw0rd!", "role": "driver", "cooperative_id": str(w.a),
                                   "group_id": str(b_group), "accept_terms": "1"})
    assert w.db.query("SELECT COUNT(*) AS n FROM users WHERE email='jp@example.com'")[0]["n"] == 0
    mailer.OUTBOX.clear()


def test_moving_a_member_to_another_cooperative_clears_their_group(w):
    mgr = w.client("mgr.a@example.com")
    create_group(mgr, "Leaving Group")
    gid = group_id(w.db, "Leaving Group")
    mgr.post(f"/manager/groups/{gid}/members", data={"user_id": str(w.drv_a)})
    w.client("admin@example.com").post("/admin/update-member", data={"user_id": str(w.drv_a), "role": "driver",
                                                                      "cooperative_id": str(w.b)})
    m = w.db.query("SELECT cooperative_id, group_id FROM cooperative_memberships WHERE user_id=%s", (w.drv_a,))[0]
    assert (m["cooperative_id"], m["group_id"]) == (w.b, None)


# ---------------------------------------------------------------- cooperative creation needs a manager
def test_cooperative_without_manager_is_an_inactive_draft(w):
    admin = w.client("admin@example.com")
    admin.post("/admin/cooperatives", data={"name": "Draft Coop", "code": "DRF"})
    c = w.db.query("SELECT id, status FROM cooperatives WHERE code='DRF'")[0]
    assert c["status"] == "SUSPENDED"
    r = admin.post(f"/admin/cooperatives/{c['id']}/edit", data={"name": "Draft Coop", "district": "", "status": "APPROVED"},
                   follow_redirects=True)
    assert "Assign an active manager" in r.get_data(as_text=True)
    assert w.db.query("SELECT status FROM cooperatives WHERE id=%s", (c["id"],))[0]["status"] == "SUSPENDED"
    assert "Draft Coop" not in w.client("admin@example.com").application.test_client().get("/register").get_data(as_text=True)


def test_cooperative_with_manager_is_active(w):
    new = w.db.user("mgr.new@example.com", role="manager")
    w.client("admin@example.com").post("/admin/cooperatives", data={"name": "Kigali Safe Drivers", "code": "KSD",
                                                                    "manager_id": str(new)})
    c = w.db.query("SELECT status, manager_user_id FROM cooperatives WHERE code='KSD'")[0]
    assert (c["status"], c["manager_user_id"]) == ("APPROVED", new)


# ---------------------------------------------------------------- redesigned driver dashboard
def test_driver_dashboard_structure_and_states(w):
    mgr = w.client("mgr.a@example.com")
    create_group(mgr, "Kigali Central")
    mgr.post(f"/manager/groups/{group_id(w.db, 'Kigali Central')}/members", data={"user_id": str(w.drv_a)})
    html = w.client("drv.a@example.com").get("/driver-dashboard").get_data(as_text=True)
    for text in ('id="status-card"', 'id="support-card"', 'id="assist"', 'id="nearby"', 'id="vehicle"', "RAB 123 A",
                 "Kigali Central", "DRV-", "Coop A", "OPEN CHAT", "AI safety", "Monitoring", "Account", "Assistance",
                 "Manager verification", "REQUEST UMUSARE SUPPORT", "Start monitoring"):
        assert text in html, text
    # safety status comes before help, assistance, nearby map and profile
    order = [html.index(x) for x in ('id="status-card"', 'id="support-card"', 'id="assist"', 'id="nearby"', 'id="vehicle"')]
    assert order == sorted(order)
    assert "drunk" not in html.lower() and "BAC" not in html.replace("blood alcohol", "")


@pytest.mark.parametrize("email", ["mgr.a@example.com", "ums.a@example.com", "admin@example.com"])
def test_driver_dashboard_and_driver_api_are_driver_only(w, email):
    c = w.client(email)
    assert c.get("/driver-dashboard").status_code == 403
    assert c.post("/driver/nearby", data="{}", content_type="application/json").status_code == 403
