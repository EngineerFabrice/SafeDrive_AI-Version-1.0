"""Mobile API (/api/mobile/v1): bearer tokens, CSRF isolation, registration + OTP, role scoping,
the assistance workflow and phone-camera monitoring through the real pipeline code path."""
import hashlib
import json
import re

import cv2
import numpy as np
import pytest

from engine.detectors import BoundingBox
from engine.detectors.face import FaceDetection
from engine.detectors.person import PersonDetection
from engine.impairment import ImpairmentModel, ModelInfo
from engine.state import ModelStatus
from tests.web.conftest import login
from website import mobile_monitoring

pytestmark = pytest.mark.db

API = "/api/mobile/v1"
KIGALI = (-1.944100, 30.061900)


def post(client, path, body=None, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(API + path, data=json.dumps(body or {}), content_type="application/json", headers=headers)


def get(client, path, token=None):
    return client.get(API + path, headers={"Authorization": f"Bearer {token}"} if token else {})


def sign_in(client, email, password="Passw0rd!"):
    r = post(client, "/auth/login", {"email": email, "password": password, "device_name": "pytest"})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["token"]


@pytest.fixture
def world(db):
    a, b = db.cooperative("Coop A", "A"), db.cooperative("Coop B", "B")
    mgr_a = db.user("mgr-a@example.com", role="manager", coop_id=a)
    db.query("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s", (mgr_a, a))
    mgr_b = db.user("mgr-b@example.com", role="manager", coop_id=b)
    db.query("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s", (mgr_b, b))
    driver = db.user("driver@example.com", role="driver", coop_id=a)
    db.query("INSERT INTO driver_profiles (user_id, vehicle_plate_number) VALUES (%s, 'RAB123A')", (driver,))
    umu = db.user("umu@example.com", role="umusare", coop_id=b)
    db.query("INSERT INTO umusare_profiles (user_id, verification_status, availability) VALUES (%s,'VERIFIED','OFFLINE')",
             (umu,))
    admin = db.user("admin@example.com", role="admin")
    return {"db": db, "a": a, "b": b, "driver": driver, "umusare": umu, "mgr_a": mgr_a, "mgr_b": mgr_b, "admin": admin}


# ================================================================== tokens
def test_public_meta_has_labels_notice_and_credit_but_no_secrets(client, db):
    body = get(client, "/meta").get_json()
    assert body["assessment_labels"] == ["SOBER", "UNCERTAIN", "POTENTIALLY_NOT_SOBER"]
    assert "blood alcohol" in body["system_notice"]
    assert body["credit"] == {"author": "Fabrice NDAYISABA", "contact": "fabricendayisaba16@gmail.com"}
    text = json.dumps(body).lower()
    assert "secret" not in text and "password" not in text


def test_login_returns_token_stored_only_as_hash(client, world):
    r = post(client, "/auth/login", {"email": "DRIVER@example.com ", "password": "Passw0rd!"})
    body = r.get_json()
    assert r.status_code == 200 and body["token_type"] == "Bearer"
    assert body["user"]["role"] == "driver" and body["user"]["membership"]["cooperative_name"] == "Coop A"
    rows = world["db"].query("SELECT token_hash FROM mobile_api_tokens")
    assert rows == [{"token_hash": hashlib.sha256(body["token"].encode()).hexdigest()}]
    assert "password_hash" not in json.dumps(body)


def test_wrong_password_and_unknown_email_get_the_same_answer(client, world):
    r1 = post(client, "/auth/login", {"email": "driver@example.com", "password": "wrong-pass1"})
    r2 = post(client, "/auth/login", {"email": "nobody@example.com", "password": "wrong-pass1"})
    assert r1.status_code == r2.status_code == 401 and r1.get_json() == r2.get_json()


def test_endpoints_require_a_valid_token(client, world):
    assert get(client, "/me").status_code == 401
    assert get(client, "/me", token="not-a-real-token").status_code == 401
    token = sign_in(client, "driver@example.com")
    assert get(client, "/me", token).get_json()["user"]["email"] == "driver@example.com"


def test_logout_revokes_the_token(client, world):
    token = sign_in(client, "driver@example.com")
    assert post(client, "/auth/logout", token=token).status_code == 200
    assert get(client, "/me", token).status_code == 401


def test_deactivated_account_token_stops_working(client, world):
    token = sign_in(client, "driver@example.com")
    world["db"].query("UPDATE users SET is_active=0 WHERE id=%s", (world["driver"],))
    assert get(client, "/me", token).status_code == 401


def test_expired_token_is_rejected(client, world):
    token = sign_in(client, "driver@example.com")
    world["db"].query("UPDATE mobile_api_tokens SET expires_at=UTC_TIMESTAMP(3) - INTERVAL 1 MINUTE")
    assert get(client, "/me", token).status_code == 401


def test_browser_session_cookie_is_not_accepted_by_the_mobile_api(client, world):
    login(client, "driver@example.com")
    assert client.get("/driver-dashboard").status_code == 200           # signed in on the web
    assert get(client, "/me").status_code == 401                       # ...but not on the mobile API
    assert post(client, "/assistance/requests", {"lat": KIGALI[0], "lon": KIGALI[1]}).status_code == 401


def test_csrf_still_protects_the_web_and_bearer_header_does_not_bypass_it(world):
    from website import create_app
    app = create_app({"TESTING": True})                 # CSRF enabled, as in production
    client = app.test_client()
    token = sign_in(client, "driver@example.com")       # mobile API works without a CSRF token
    assert get(client, "/me", token).status_code == 200
    r = client.post("/assistance/requests", data=json.dumps({"lat": KIGALI[0], "lon": KIGALI[1]}),
                    content_type="application/json", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400 and "CSRF" in r.get_json()["error"]    # web endpoint: still CSRF-protected
    assert world["db"].query("SELECT COUNT(*) AS n FROM assistance_requests")[0]["n"] == 0


def test_roles_are_enforced(client, world):
    umu = sign_in(client, "umu@example.com")
    drv = sign_in(client, "driver@example.com")
    mgr = sign_in(client, "mgr-a@example.com")
    assert post(client, "/assistance/requests", {"lat": KIGALI[0], "lon": KIGALI[1]}, umu).status_code == 403
    assert get(client, "/assistance/umusare/status", drv).status_code == 403
    assert post(client, "/monitoring/phone/start", token=umu).status_code == 403     # no camera for Umusare
    assert post(client, "/monitoring/phone/start", token=mgr).status_code == 403     # ...or managers
    assert get(client, "/manager/console", drv).status_code == 403
    assert get(client, "/admin/overview", mgr).status_code == 403


# ================================================================== registration + email OTP
@pytest.fixture
def outbox():
    from website import mailer
    mailer.OUTBOX.clear()
    return mailer.OUTBOX


def register(client, coop, **over):
    body = {"username": "Aline Uwase", "email": "aline@example.com", "password": "Passw0rd!",
            "confirm_password": "Passw0rd!", "role": "driver", "cooperative_id": coop, "accept_terms": True,
            "vehicle_plate_number": "rab123a", **over}
    return post(client, "/auth/register", body)


def test_registration_options_list_only_approved_cooperatives(client, world):
    world["db"].cooperative("Draft", "DR", status="SUSPENDED")
    names = [c["name"] for c in get(client, "/registration-options").get_json()["cooperatives"]]
    assert names == ["Coop A", "Coop B"]


def test_register_verify_email_and_sign_in(client, world, outbox):
    r = register(client, world["a"])
    assert r.status_code == 201
    pending = r.get_json()["pending_token"]
    code = re.search(r"code is: (\d{6})", outbox[-1]["body"]).group(1)
    bad = post(client, "/auth/verify-email", {"pending_token": pending, "code": f"{(int(code) + 1) % 10**6:06d}"})
    assert bad.status_code == 400 and bad.get_json()["code"] == "INVALID_CODE"
    ok = post(client, "/auth/verify-email", {"pending_token": bad.get_json()["pending_token"], "code": code})
    body = ok.get_json()
    assert ok.status_code == 200 and body["token"] and body["user"]["email_verified"] is True
    assert body["user"]["verification"]["status"] == "PENDING"            # cooperative manager still decides
    assert get(client, "/me", body["token"]).status_code == 200


def test_registration_validation_errors(client, world):
    r = register(client, world["a"], accept_terms=False, password="short")
    errors = r.get_json()["errors"]
    assert r.status_code == 400 and any("Terms" in e for e in errors) and any("Password" in e for e in errors)
    assert register(client, world["a"], role="admin").status_code == 400          # no self-registered admins


def test_duplicate_email_registration_looks_the_same_and_never_verifies(client, world, outbox):
    new = register(client, world["a"], email="fresh@example.com").get_json()
    dup = register(client, world["a"], email="driver@example.com").get_json()
    assert set(new) == set(dup) and len(new["pending_token"]) == len(dup["pending_token"])
    mask = re.compile(r"\S+\*\*\*@\S+")
    assert mask.sub("MASKED", new["message"]) == mask.sub("MASKED", dup["message"])
    assert "already" in outbox[-1]["body"]                        # the owner was told; no code was sent
    r = post(client, "/auth/verify-email", {"pending_token": dup["pending_token"], "code": "123456"})
    assert r.status_code == 400 and "token" not in r.get_json()


# ================================================================== assistance workflow
def test_full_assistance_flow_driver_and_umusare(client, world):
    drv, umu = sign_in(client, "driver@example.com"), sign_in(client, "umu@example.com")
    r = post(client, "/assistance/umusare/availability",
             {"available": True, "lat": KIGALI[0] + 0.01, "lon": KIGALI[1], "accuracy": 10}, umu)
    assert r.status_code == 200 and r.get_json()["availability"] == "AVAILABLE"

    r = post(client, "/assistance/requests", {"lat": KIGALI[0], "lon": KIGALI[1], "accuracy": 8,
                                              "trigger": "AI_TRIGGERED"}, drv)
    req = r.get_json()["request"]
    assert r.status_code == 201 and req["status"] == "MATCHING"
    assert req["type"] == "DRIVER_INITIATED"           # AI claim without a monitoring session is not trusted

    incoming = get(client, "/assistance/umusare/status", umu).get_json()["incoming"]
    assert [i["id"] for i in incoming] == [req["id"]]
    assert "driver" not in incoming[0] and "pickup" not in incoming[0]     # no identity / exact spot before accepting
    assert post(client, f"/assistance/requests/{req['id']}/accept", token=umu).status_code == 200

    view = get(client, "/assistance/requests/current", drv).get_json()["request"]
    assert view["status"] == "ACCEPTED" and view["umusare"]["name"] == "umu"
    assert post(client, f"/assistance/requests/{req['id']}/location",
                {"lat": KIGALI[0], "lon": KIGALI[1] + 0.001, "accuracy": 5}, drv).status_code == 200
    assert post(client, f"/assistance/requests/{req['id']}/connect", token=umu).status_code == 200
    assert post(client, f"/assistance/requests/{req['id']}/complete", token=drv).status_code == 403   # Umusare only
    done = post(client, f"/assistance/requests/{req['id']}/complete", {"final_fare": 1}, umu).get_json()
    assert done["result"]["fare"] >= 0 and done["payments"][0]["status"] == "PAYMENT_PENDING"     # fare from the server
    view = get(client, "/assistance/requests/current", drv).get_json()["request"]
    assert view["status"] == "COMPLETED" and view["payment"]["status"] == "PAYMENT_PENDING"
    assert post(client, f"/assistance/requests/{req['id']}/payment-sent", token=drv).status_code == 200
    assert post(client, f"/assistance/requests/{req['id']}/payment-received", token=umu).status_code == 200
    rated = post(client, f"/assistance/requests/{req['id']}/rate", {"rating": 5}, drv).get_json()
    assert rated["request"]["summary"]["rating"] == 5


def test_other_drivers_cannot_touch_a_request(client, world):
    other = world["db"].user("other@example.com", role="driver", coop_id=world["a"])
    assert other
    drv, oth = sign_in(client, "driver@example.com"), sign_in(client, "other@example.com")
    req = post(client, "/assistance/requests", {"lat": KIGALI[0], "lon": KIGALI[1]}, drv).get_json()["request"]
    assert post(client, f"/assistance/requests/{req['id']}/cancel", token=oth).status_code == 404
    assert get(client, "/assistance/requests/current", oth).get_json()["request"] is None


def test_invalid_location_is_rejected(client, world):
    drv = sign_in(client, "driver@example.com")
    r = post(client, "/assistance/requests", {"lat": 200, "lon": "x"}, drv)
    assert r.status_code == 400 and r.get_json()["code"] == "INVALID_LOCATION"


# ================================================================== manager / admin scoping
def test_manager_console_is_scoped_to_own_cooperative_and_review_works(client, world):
    mgr_a, mgr_b = sign_in(client, "mgr-a@example.com"), sign_in(client, "mgr-b@example.com")
    console = get(client, "/manager/console", mgr_a).get_json()["console"]
    assert [d["id"] for d in console["drivers"]] == [world["driver"]] and console["umusare"] == []
    assert post(client, f"/manager/members/{world['driver']}/review", {"action": "VERIFY"}, mgr_b).status_code in (403, 404)
    r = post(client, f"/manager/members/{world['driver']}/review", {"action": "VERIFY"}, mgr_a)
    assert r.status_code == 200 and r.get_json()["verification_status"] == "VERIFIED"


def test_admin_overview(client, world):
    body = get(client, "/admin/overview", sign_in(client, "admin@example.com")).get_json()
    assert body["users"]["driver"] == 1 and body["users"]["manager"] == 2


# ================================================================== phone-camera monitoring
class Person:
    status, error = ModelStatus.READY, ""

    def load(self):
        return True

    def detect(self, image):
        return [PersonDetection(BoundingBox(60, 20, 420, 460), 0.9)]


class Face:
    status, error, landmark_detector = ModelStatus.READY, "", None

    def __init__(self, found=True):
        self.found = found

    def load(self):
        return True

    def detect(self, image, roi):
        return FaceDetection(bbox=BoundingBox(140, 100, 340, 340), score=0.95, backend="yunet") if self.found else None


class FixedModel(ImpairmentModel):
    def __init__(self, p):
        self.p = p
        self._info = ModelInfo(name="fixed", version="1", provider="test", schema_version=1,
                               classes=("non_alcoholic", "alcoholic"), input_type="face_image",
                               positive_class="alcoholic", development_only=True)

    @property
    def info(self):
        return self._info

    def _predict(self, face):
        return ("alcoholic" if self.p >= 0.5 else "non_alcoholic"), {"non_alcoholic": 1 - self.p, "alcoholic": self.p}


def jpeg():
    rng = np.random.default_rng(3)
    image = np.clip(120 + rng.integers(-50, 50, (480, 480, 3)), 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


@pytest.fixture
def phone(monkeypatch):
    monkeypatch.setattr(mobile_monitoring, "MIN_FRAME_INTERVAL_S", 0.0)

    def use(p=0.9, face=True):
        mobile_monitoring.set_components(Person(), Face(face), FixedModel(p))
    yield use
    mobile_monitoring.reset_components()


def send_frame(client, token, data=None):
    return client.post(API + "/monitoring/phone/frame", data=data if data is not None else jpeg(),
                       content_type="image/jpeg", headers={"Authorization": f"Bearer {token}"})


def test_phone_frames_run_through_the_pipeline_to_a_temporal_assessment(client, world, phone):
    phone(p=0.92)
    drv = sign_in(client, "driver@example.com")
    assert send_frame(client, drv).status_code == 409                       # not started yet
    assert post(client, "/monitoring/phone/start", token=drv).get_json()["active"] is True
    results = [send_frame(client, drv).get_json() for _ in range(6)]
    first, last = results[0], results[-1]
    assert first["driver_detected"] and first["face_detected"]
    assert first["impairment"]["prediction"] == "alcoholic" and first["impairment"]["development_only"] is True
    assert first["assessment"]["assessment"] == "ASSESSING"                 # one frame never decides
    assert first["min_quality"] == 0.5 and first["counted"] == (first["face_quality"]["score"] >= 0.5)
    assert last["assessment"]["valid_frames"] == sum(r["counted"] for r in results)
    assert last["assessment"]["assessment"] == "POTENTIALLY_NOT_SOBER"
    assert get(client, "/monitoring/phone/status", drv).get_json()["frames_processed"] == 6

    # a verified phone assessment makes the AI_TRIGGERED claim true; the label is never sent by the client
    req = post(client, "/assistance/requests", {"lat": KIGALI[0], "lon": KIGALI[1], "trigger": "AI_TRIGGERED"},
               drv).get_json()["request"]
    assert req["type"] == "AI_TRIGGERED"
    assert post(client, "/monitoring/phone/stop", token=drv).get_json()["active"] is False
    assert get(client, "/monitoring/phone/status", drv).get_json()["active"] is False


def test_sober_frames_give_sober(client, world, phone):
    phone(p=0.05)
    drv = sign_in(client, "driver@example.com")
    post(client, "/monitoring/phone/start", token=drv)
    results = [send_frame(client, drv).get_json() for _ in range(6)]
    assert results[-1]["assessment"]["assessment"] == "SOBER"


def test_no_face_is_never_scored(client, world, phone):
    phone(p=0.95, face=False)
    drv = sign_in(client, "driver@example.com")
    post(client, "/monitoring/phone/start", token=drv)
    results = [send_frame(client, drv).get_json() for _ in range(16)]
    assert all(r["impairment"] is None for r in results)
    assert results[-1]["assessment"]["assessment"] in ("ASSESSING", "UNCERTAIN")
    assert "No face" in results[-1]["message"]


def test_invalid_and_oversized_frames_are_rejected(client, world, phone, monkeypatch):
    phone()
    drv = sign_in(client, "driver@example.com")
    post(client, "/monitoring/phone/start", token=drv)
    assert send_frame(client, drv, b"not an image").get_json()["code"] == "INVALID_FRAME"
    assert send_frame(client, drv, b"").get_json()["code"] == "NO_FRAME"
    monkeypatch.setattr(mobile_monitoring, "MAX_FRAME_BYTES", 100)
    assert send_frame(client, drv).status_code == 413


def test_frame_rate_is_limited(client, world, phone, monkeypatch):
    phone()
    monkeypatch.setattr(mobile_monitoring, "MIN_FRAME_INTERVAL_S", 60.0)
    drv = sign_in(client, "driver@example.com")
    post(client, "/monitoring/phone/start", token=drv)
    assert send_frame(client, drv).status_code == 200
    assert send_frame(client, drv).status_code == 429


def test_phone_sessions_are_per_driver(client, world, phone):
    phone(p=0.92)
    world["db"].user("other@example.com", role="driver", coop_id=world["a"])
    drv, oth = sign_in(client, "driver@example.com"), sign_in(client, "other@example.com")
    post(client, "/monitoring/phone/start", token=drv)
    for _ in range(6):
        send_frame(client, drv)
    assert get(client, "/monitoring/phone/status", oth).get_json() == {"active": False, "frames_processed": 0,
                                                                       "assessment": None}
    req = post(client, "/assistance/requests", {"lat": KIGALI[0], "lon": KIGALI[1], "trigger": "AI_TRIGGERED"},
               oth).get_json()["request"]
    assert req["type"] == "DRIVER_INITIATED"            # another driver's alert does not count
