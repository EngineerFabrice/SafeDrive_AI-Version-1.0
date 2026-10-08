"""AI-triggered vs driver-initiated (manual) assistance, admin monitoring view, dev-server port guard."""
import json
import socket

import pytest

import website.monitoring as monitoring_module
from tests.web.conftest import login
from website.config import port_in_use

pytestmark = pytest.mark.db

KIGALI = (-1.944100, 30.061900)


class FakeSnapshot:
    def __init__(self, assessment):
        self.assessment = {"assessment": assessment, "confidence": 0.9} if assessment else None


class FakeEngine:
    def __init__(self, assessment):
        self.snap = FakeSnapshot(assessment)

    def snapshot(self):
        return self.snap


class FakeRecorder:
    def __init__(self, owner):
        self.owner = owner

    def active_owner(self):
        return self.owner


@pytest.fixture
def setup(db):
    a, b = db.cooperative("Coop A", "A"), db.cooperative("Coop B", "B")
    driver = db.user("driver@example.com", role="driver", coop_id=a)
    umu = db.user("u@example.com", role="umusare", coop_id=b)            # other cooperative
    db.query("INSERT INTO umusare_profiles (user_id, verification_status, availability) VALUES (%s,'VERIFIED','AVAILABLE')",
             (umu,))
    db.query("INSERT INTO user_locations (user_id, lat, lon, updated_at) VALUES (%s,%s,%s,UTC_TIMESTAMP(3))",
             (umu, KIGALI[0] + 0.01, KIGALI[1]))
    return {"db": db, "driver": driver, "umusare": umu}


def live_session(monkeypatch, owner, assessment):
    """The driver's monitoring session as the assistance API sees it."""
    engine = FakeEngine(assessment)
    monkeypatch.setattr(monitoring_module, "_engine", engine)
    monkeypatch.setattr(monitoring_module, "_recorder", FakeRecorder(owner))
    return engine


def request_help(client, trigger):
    return client.post("/assistance/requests", content_type="application/json",
                       data=json.dumps({"lat": KIGALI[0], "lon": KIGALI[1], "trigger": trigger}))


def stored_type(db):
    return db.query("SELECT trigger_source FROM assistance_requests ORDER BY id DESC LIMIT 1")[0]["trigger_source"]


def test_ai_triggered_request_is_marked_ai_triggered_and_still_matches(client, setup, monkeypatch):
    live_session(monkeypatch, setup["driver"], "POTENTIALLY_NOT_SOBER")
    login(client, "driver@example.com")
    r = request_help(client, "AI_TRIGGERED")
    assert r.status_code == 201 and r.get_json()["type"] == "AI_TRIGGERED"
    assert r.get_json()["status"] == "MATCHING" and r.get_json()["umusare_contacted"] == 1   # other-coop Umusare
    assert stored_type(setup["db"]) == "AI_TRIGGERED"


@pytest.mark.parametrize("assessment", ["SOBER", "UNCERTAIN"])
def test_manual_request_works_when_ai_is_not_alarmed_and_never_changes_the_ai(client, setup, monkeypatch, assessment):
    engine = live_session(monkeypatch, setup["driver"], assessment)
    login(client, "driver@example.com")
    r = request_help(client, "DRIVER_INITIATED")
    assert r.status_code == 201 and r.get_json()["type"] == "DRIVER_INITIATED"
    assert r.get_json()["status"] == "MATCHING"
    assert stored_type(setup["db"]) == "DRIVER_INITIATED"
    assert engine.snapshot().assessment["assessment"] == assessment        # AI decision untouched


@pytest.mark.parametrize("assessment, owner_is_driver", [("SOBER", True), ("UNCERTAIN", True),
                                                         ("POTENTIALLY_NOT_SOBER", False), (None, True)])
def test_ai_claim_is_verified_server_side(client, setup, monkeypatch, assessment, owner_is_driver):
    """A browser claiming AI_TRIGGERED is downgraded unless this driver's live session really is POTENTIALLY_NOT_SOBER."""
    live_session(monkeypatch, setup["driver"] if owner_is_driver else 999999, assessment)
    login(client, "driver@example.com")
    assert request_help(client, "AI_TRIGGERED").status_code == 201
    assert stored_type(setup["db"]) == "DRIVER_INITIATED"


def test_manual_and_ai_requests_share_duplicate_protection(client, setup, monkeypatch):
    live_session(monkeypatch, setup["driver"], "SOBER")
    login(client, "driver@example.com")
    assert request_help(client, "DRIVER_INITIATED").status_code == 201
    assert request_help(client, "AI_TRIGGERED").status_code == 409


def test_unauthenticated_manual_request_rejected(client, setup):
    assert request_help(client, "DRIVER_INITIATED").status_code in (302, 401)


def test_manual_request_keeps_privacy_and_verified_only_accept(client, app, setup, monkeypatch):
    live_session(monkeypatch, setup["driver"], "SOBER")
    login(client, "driver@example.com")
    req_id = request_help(client, "DRIVER_INITIATED").get_json()["id"]
    u = app.test_client()
    login(u, "u@example.com")
    raw = u.get("/assistance/umusare/status").get_data(as_text=True)
    assert "-1.9441" not in raw and "30.0619" not in raw                   # approximate area only
    setup["db"].query("UPDATE umusare_profiles SET verification_status='PENDING'")
    assert u.post(f"/assistance/requests/{req_id}/accept").status_code == 403


def test_admin_sees_assistance_requests_without_exact_location(client, app, setup, monkeypatch):
    live_session(monkeypatch, setup["driver"], "POTENTIALLY_NOT_SOBER")
    login(client, "driver@example.com")
    request_help(client, "AI_TRIGGERED")
    monkeypatch.setattr(monitoring_module, "_recorder", FakeRecorder(None))
    request_help(client, "DRIVER_INITIATED")       # duplicate: refused, the AI-triggered one stays active
    setup["db"].user("admin@example.com", role="admin")
    admin = app.test_client()
    login(admin, "admin@example.com")
    html = admin.get("/admin-dashboard").get_data(as_text=True)
    section = html[html.index('id="assistance-monitor"'):]
    assert "Assistance requests" in section and "AI-triggered" in section and "MATCHING" in section
    assert "driver" in section and "Coop A" in section and "-1.94, 30.06" in section
    assert "-1.9441" not in section and "30.0619" not in section            # no exact coordinates
    # the section is secondary: it comes after the main user-management content
    assert html.index('id="assistance-monitor"') > html.index("<h2>Users</h2>")


def test_non_admin_cannot_open_admin_assistance_view(client, setup):
    login(client, "driver@example.com")
    assert client.get("/admin-dashboard", headers={"Accept": "application/json"}).status_code == 403


def test_driver_dashboard_manual_support_is_secondary_and_confirmed(client, setup):
    login(client, "driver@example.com")
    html = client.get("/driver-dashboard").get_data(as_text=True)
    assert "REQUEST UMUSARE SUPPORT" in html and "Need help even if the AI says you are okay?" in html
    assert "Do you want to request assistance from a nearby verified Umusare?" in html
    assert "You can request assistance even if SafeDrive AI has not detected potential alcohol-related impairment." in html
    assert 'requestAssistance("AI_TRIGGERED")' in html and 'requestAssistance("DRIVER_INITIATED")' in html
    assert "Allow location access to request nearby Umusare assistance." in html
    assert html.index('id="status-card"') < html.index('id="support-card"')   # below the main AI status


def test_port_guard_detects_a_server_already_listening():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert port_in_use("127.0.0.1", port)
    assert not port_in_use("127.0.0.1", port)
