"""Umusare identity/verification card, location visibility, and admin management."""
import json

import pytest

import website.monitoring as monitoring_module
from tests.web.conftest import login
from tests.web.journey_helpers import START, Journey, post

pytestmark = pytest.mark.db


@pytest.fixture
def j(app, db):
    db.query("INSERT IGNORE INTO pricing_settings (id) VALUES (1)")
    return Journey(app, db)


# ---------------------------------------------------------------- profile / verification
def test_driver_sees_verified_umusare_identity_after_acceptance(j):
    before = j.request()
    assert "umusare" not in before                                   # no identity before acceptance
    j.accept()
    u = j.d.get("/assistance/requests/current").get_json()["request"]["umusare"]
    p = u["profile"]
    assert p["name"] == "Jean Claude M." and p["umusare_id"] == f"UMS-{j.umusare_id:05d}"
    assert p["phone"] == "+250788123456" and p["email"] == "umusare@example.com"
    assert p["cooperative"] == "Kigali Safe Transport" and p["member_id"].startswith("KST-M")
    v = p["verification"]
    assert v["identity_verified"] and v["membership_approved"] and v["account_active"] and v["eligible"]
    assert v["verified_at"] is not None and p["photo_url"] is None              # nothing invented
    assert u["eta_min"] >= 1 and u["distance_km"] == pytest.approx(1.0, abs=0.05)


def test_verification_claims_reflect_database(j):
    j.db.query("UPDATE cooperative_memberships SET status='PENDING' WHERE user_id=%s", (j.umusare_id,))
    p = None
    from website import assistance_service as svc
    with svc.transaction() as cur:
        p = svc._umusare_identity(cur, j.umusare_id)
    assert p["verification"]["identity_verified"] and not p["verification"]["membership_approved"]
    assert not p["verification"]["eligible"]


def test_offer_shows_type_and_rate_but_no_private_information(j):
    j.request(trigger="DRIVER_INITIATED")
    offer = j.u.get("/assistance/umusare/status").get_json()["incoming"][0]
    raw = json.dumps(offer)
    assert offer["type_label"] == "Driver-initiated support" and offer["fare"]["rate_label"] == "RWF 500 / km"
    assert "driver@example.com" not in raw and '"name"' not in raw and "phone" not in raw
    assert "-1.9441" not in raw and "30.0619" not in raw


# ---------------------------------------------------------------- live locations
def test_live_locations_only_between_driver_and_accepted_umusare(j, app, db):
    j.request(); j.accept()
    assert post(j.d, f"/assistance/requests/{j.id}/location", {"lat": START[0], "lon": START[1]}).status_code == 200
    umu = j.u.get("/assistance/umusare/status").get_json()["active"]
    assert umu["driver_location"]["lat"] == START[0] and umu["my_location"] is not None
    drv = j.d.get("/assistance/requests/current").get_json()["request"]
    assert drv["umusare"]["location"] is not None and drv["my_location"]["lat"] == START[0]
    db.user("other@example.com", role="umusare", coop_id=db.cooperative("Other", "OT"))
    o = app.test_client(); login(o, "other@example.com")
    raw = o.get("/assistance/umusare/status").get_data(as_text=True)
    assert "-1.9441" not in raw and '"active":null' in raw.replace(" ", "")


def test_tracking_stops_after_cancellation(j):
    j.request(); j.accept()
    post(j.d, f"/assistance/requests/{j.id}/location", {"lat": START[0], "lon": START[1]})
    post(j.d, f"/assistance/requests/{j.id}/cancel")
    assert post(j.u, f"/assistance/requests/{j.id}/location", {"lat": START[0], "lon": START[1]}).status_code == 409
    view = j.d.get("/assistance/requests/current").get_json()["request"]
    assert "umusare" not in view and not view["sharing"]


# ---------------------------------------------------------------- admin management
@pytest.fixture
def admin(app, db):
    db.user("admin@example.com", role="admin")
    c = app.test_client()
    login(c, "admin@example.com")
    return c


def test_admin_driver_management_and_history(j, admin):
    j.to_payment_pending()
    html = admin.get("/admin/drivers").get_data(as_text=True)
    assert f"DRV-{j.driver_id:05d}" in html and "Kigali Safe Transport" in html and "Not monitoring" in html
    detail = admin.get(f"/admin/drivers/{j.driver_id}").get_data(as_text=True)
    assert f"AS-{j.id:06d}" in detail and "PAYMENT PENDING" in detail and "Driver-initiated support" in detail
    assert admin.get("/admin/drivers/999999").status_code == 404


def test_admin_umusare_management_and_assistance_monitor(j, admin):
    j.to_payment_pending()
    html = admin.get("/admin/umusare").get_data(as_text=True)
    assert f"UMS-{j.umusare_id:05d}" in html and "VERIFIED" in html and "+250788123456" in html
    mon = admin.get("/admin/assistance?payment=pending&type=driver").get_data(as_text=True)
    assert f"AS-{j.id:06d}" in mon and "Driver-initiated" in mon and "RWF " in mon
    assert "-1.9441" not in mon and "30.0619" not in mon                       # approximate area only
    assert f"AS-{j.id:06d}" not in admin.get("/admin/assistance?type=ai").get_data(as_text=True)
    assert f"AS-{j.id:06d}" not in admin.get("/admin/assistance?payment=completed").get_data(as_text=True)


@pytest.mark.parametrize("path", ["/admin/drivers", "/admin/umusare", "/admin/assistance", "/admin/pricing"])
@pytest.mark.parametrize("role", ["driver", "umusare", "manager"])
def test_admin_pages_reject_other_roles(client, db, path, role):
    db.user("x@example.com", role=role, coop_id=db.cooperative())
    login(client, "x@example.com")
    assert client.get(path, headers={"Accept": "application/json"}).status_code == 403


def test_admin_suspends_and_reverifies_umusare(j, admin):
    r = admin.post(f"/admin/umusare/{j.umusare_id}/verification", data={"status": "SUSPENDED"})
    assert r.status_code == 302
    prof = j.db.query("SELECT verification_status, availability FROM umusare_profiles")[0]
    assert (prof["verification_status"], prof["availability"]) == ("SUSPENDED", "OFFLINE")
    assert j.db.query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s", (j.umusare_id,))[0]["n"] == 0
    assert j.request()["status"] == "NO_UMUSARE_AVAILABLE"                  # no longer matchable
    admin.post(f"/admin/umusare/{j.umusare_id}/verification", data={"status": "VERIFIED"})
    assert j.db.query("SELECT verification_status FROM umusare_profiles")[0]["verification_status"] == "VERIFIED"
    actions = [r["action"] for r in j.db.query("SELECT action FROM audit_logs")]
    assert actions.count("UMUSARE_VERIFICATION_CHANGED") == 2


def test_admin_cannot_suspend_umusare_during_assistance(j, admin):
    j.request(); j.accept()
    admin.post(f"/admin/umusare/{j.umusare_id}/verification", data={"status": "SUSPENDED"})
    assert j.db.query("SELECT verification_status FROM umusare_profiles")[0]["verification_status"] == "VERIFIED"


def test_admin_deactivates_driver_and_sets_phone(j, admin, app):
    admin.post(f"/admin/users/{j.driver_id}/active", data={"active": "0"})
    assert j.db.query("SELECT is_active FROM users WHERE id=%s", (j.driver_id,))[0]["is_active"] == 0
    c = app.test_client()
    assert login(c, "driver@example.com").status_code == 200                # login refused (form shown again)
    admin.post(f"/admin/users/{j.driver_id}/phone", data={"phone": "078 812 3456"})
    assert j.db.query("SELECT phone FROM users WHERE id=%s", (j.driver_id,))[0]["phone"] == "+250788123456"
    admin.post(f"/admin/users/{j.driver_id}/phone", data={"phone": "not a phone"})
    assert j.db.query("SELECT phone FROM users WHERE id=%s", (j.driver_id,))[0]["phone"] == "+250788123456"
    actions = [r["action"] for r in j.db.query("SELECT action FROM audit_logs")]
    assert "ACCOUNT_STATUS_CHANGED" in actions and "PHONE_UPDATED" in actions


def test_admin_cannot_change_ai_assessment(j, admin, monkeypatch):
    class Snap:
        assessment = {"assessment": "SOBER", "confidence": 0.9}

    class Engine:
        is_running = True

        def snapshot(self):
            return Snap()

    class Rec:
        def active_owner(self):
            return j.driver_id
    monkeypatch.setattr(monitoring_module, "_engine", Engine())
    monkeypatch.setattr(monitoring_module, "_recorder", Rec())
    html = admin.get("/admin/drivers").get_data(as_text=True)
    assert "SOBER" in html and "cannot change an" in html
    for path in (f"/admin/drivers/{j.driver_id}/assessment", "/monitoring/assessment", f"/admin/users/{j.driver_id}/assessment"):
        assert admin.post(path, data={"assessment": "POTENTIALLY_NOT_SOBER"}).status_code in (404, 405)
    assert Snap.assessment["assessment"] == "SOBER"


def test_self_service_phone_validation(j):
    assert post(j.u, "/assistance/profile/phone", {"phone": "12"}).status_code == 400
    r = post(j.u, "/assistance/profile/phone", {"phone": "0788 111 222"})
    assert r.status_code == 200 and r.get_json()["phone"] == "+250788111222"


def test_pages_render_with_new_sections(j, admin):
    j.request(); j.accept()
    dhtml = j.d.get("/driver-dashboard").get_data(as_text=True)
    for text in ("VERIFIED UMUSARE", "ESTIMATED FARE", "PAYMENT REQUIRED", "PAY NOW", "PAYMENT SENT",
                 "CALL UMUSARE", "VIEW LIVE LOCATION", "ASSISTANCE COMPLETED", "leaflet", "assist-map.js"):
        assert text in dhtml, text
    uhtml = j.u.get("/umusare-dashboard").get_data(as_text=True)
    for text in ("NEW SAFETY ASSISTANCE", "COMPLETE JOURNEY", "ARRIVED", "CONFIRM PAYMENT RECEIVED",
                 "REPORT PAYMENT PROBLEM", "NAVIGATE", "Payment phone"):
        assert text in uhtml, text
    assert admin.get("/admin/pricing").status_code == 200
