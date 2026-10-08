"""Every page renders for its role with the v2 schema, carries CSRF tokens and makes no false claims."""
import re

import pytest

from tests.web.conftest import login

pytestmark = pytest.mark.db

FORBIDDEN_CLAIMS = re.compile(r"drows|sleep|fatigue|yawn|\d+(\.\d+)?\s*%\s*(accuracy|detection)", re.I)


def test_public_pages_render_without_false_claims(client, db):
    db.cooperative()
    for path in ("/", "/login", "/register"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert not FORBIDDEN_CLAIMS.search(r.get_data(as_text=True)), path


def test_register_page_offers_roles_and_cooperatives(client, db):
    db.cooperative(name="Koperative XYZ", code="XYZ")
    html = client.get("/register").get_data(as_text=True)
    assert 'name="csrf_token"' in html and "Koperative XYZ" in html
    assert 'value="driver"' in html and 'value="umusare"' in html
    assert 'value="admin"' not in html and 'value="manager"' not in html


@pytest.mark.parametrize("role, path, expected", [
    ("driver", "/driver-dashboard", "Start monitoring"),
    ("umusare", "/umusare-dashboard", "Waiting for verification"),
    ("manager", "/manager-dashboard", "Members"),
    ("admin", "/admin-dashboard", "Cooperatives"),
])
def test_role_dashboards_render(client, db, role, path, expected):
    coop = db.cooperative()
    uid = db.user(f"{role}@example.com", role=role, coop_id=coop)
    if role == "umusare":
        db.query("INSERT INTO umusare_profiles (user_id) VALUES (%s)", (uid,))
    login(client, f"{role}@example.com")
    r = client.get(path)
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and expected in html
    assert 'action="/logout"' in html and 'name="csrf_token"' in html     # logout is a CSRF-protected POST
    assert "chef" not in html.lower() and not FORBIDDEN_CLAIMS.search(html)


def test_manager_sees_only_own_cooperative_members(client, db):
    a, b = db.cooperative("Coop A", "A"), db.cooperative("Coop B", "B")
    db.user("mgr@example.com", role="manager", coop_id=a)
    db.user("member@example.com", coop_id=a)
    db.user("outsider@example.com", coop_id=b)
    login(client, "mgr@example.com")
    html = client.get("/manager-dashboard").get_data(as_text=True)
    assert "member@example.com" in html and "outsider@example.com" not in html


def test_monitoring_page_sends_csrf_token_and_has_no_legacy_roles(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    login(client, "d@example.com")
    html = client.get("/monitoring/").get_data(as_text=True)
    assert 'name="csrf-token"' in html and '"X-CSRFToken": token' in html
    assert "chef" not in html.lower()


def test_html_errors_use_error_page(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    login(client, "d@example.com")
    r = client.get("/admin-dashboard")
    assert r.status_code == 403 and "permission" in r.get_data(as_text=True)
    assert client.get("/no-such-page").status_code == 404


def test_driver_dashboard_has_safety_alert_with_required_wording(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    login(client, "d@example.com")
    html = client.get("/driver-dashboard").get_data(as_text=True)
    for text in ("SAFETY ALERT", "POTENTIALLY NOT SOBER",
                 "Our AI detected visual patterns that may be associated with alcohol-related impairment.",
                 "Please do not continue driving if you are not safe to drive.",
                 "REQUEST UMUSARE ASSISTANCE", "CONTINUE MONITORING",
                 "Unable to reliably assess driver. Please improve camera visibility."):
        assert text in html, text
    assert '"X-CSRFToken": token' in html                 # start/stop are CSRF-protected POSTs
    assert "drunk" not in html.lower() and "BAC" not in html.replace("blood alcohol", "")


def test_driver_dashboard_has_working_assistance_request_not_placeholder(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    login(client, "d@example.com")
    html = client.get("/driver-dashboard").get_data(as_text=True)
    assert "not available yet in this version" not in html
    assert 'requestAssistance("AI_TRIGGERED")' in html and "/assistance/requests" in html
    assert "Finding a nearby verified Umusare" in html and "ASSISTANCE COMPLETED" in html


def test_verified_umusare_dashboard_has_availability_and_request_controls(client, db):
    uid = db.user("u@example.com", role="umusare", coop_id=db.cooperative())
    db.query("INSERT INTO umusare_profiles (user_id, verification_status) VALUES (%s, 'VERIFIED')", (uid,))
    login(client, "u@example.com")
    html = client.get("/umusare-dashboard").get_data(as_text=True)
    assert "Go available" in html and "/assistance/umusare/status" in html
    assert "ACCEPT" in html and "DECLINE" in html and "Waiting for verification" not in html
