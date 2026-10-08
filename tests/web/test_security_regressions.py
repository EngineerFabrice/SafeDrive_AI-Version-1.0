"""Regression tests for the audit findings: email enumeration, nearby-location spoofing / scanning,
nearby privacy documentation, open redirects, resend limits after delivery failure, stale presence
and the development secret key."""
import json
import re

import pytest

from tests.web.conftest import login
from tests.web.test_email_terms import FORM, clock, code_from, coop, outbox, register  # noqa: F401 (fixtures)
from tests.web.test_nearby import CLOSE, FAR, HERE, Town, post

pytestmark = pytest.mark.db


# ================================================================== 1. email enumeration
def _session(app, client):
    cookie = client.get_cookie("session")
    return cookie.value, app.session_interface.get_signing_serializer(app).loads(cookie.value)


def _normalize(html):
    html = re.sub(r'name="csrf_token" value="[^"]+"', "", html)
    html = re.sub(r"[a-z0-9]{1,4}\*\*\*@example\.com", "MASKED", html)   # echoes the address the user typed
    return re.sub(r'data-wait="\d+"', 'data-wait="N"', html)


def test_duplicate_email_registration_is_indistinguishable(app, db, coop, outbox):
    db.user("taken@example.com", coop_id=coop)
    new, dup = app.test_client(), app.test_client()
    r_new = register(new, coop, email="fresh@example.com")
    r_dup = register(dup, coop, email="taken@example.com")
    assert (r_new.status_code, r_new.headers["Location"]) == (r_dup.status_code, r_dup.headers["Location"])

    page_new, page_dup = new.get("/verify-email").get_data(as_text=True), dup.get("/verify-email").get_data(as_text=True)
    wait_new = int(re.search(r'data-wait="(\d+)"', page_new).group(1))
    wait_dup = int(re.search(r'data-wait="(\d+)"', page_dup).group(1))
    assert 55 <= wait_new <= 60 and abs(wait_new - wait_dup) <= 1          # same resend countdown
    assert _normalize(page_new) == _normalize(page_dup)                     # same page, same flash message

    raw_new, s_new = _session(app, new)
    raw_dup, s_dup = _session(app, dup)
    assert set(s_new["verify"]) == set(s_dup["verify"]) == {"t", "masked"}  # no readable user id
    assert len(s_new["verify"]["t"]) == len(s_dup["verify"]["t"])           # encrypted, same length
    assert "uid" not in json.dumps(s_new) and "uid" not in json.dumps(s_dup)
    assert abs(len(raw_new) - len(raw_dup)) <= 4                            # cookie size does not reveal it either

    # Immediate resend: same cooldown answer in both flows
    msgs = [re.sub(r"\d+", "N", re.search(r'class="flash \w+" role="alert">([^<]+)<',
                                          c.post("/verify-email/resend", follow_redirects=True).get_data(as_text=True)).group(1))
            for c in (new, dup)]
    assert msgs[0] == msgs[1] and "Please wait" in msgs[0]

    # Wrong codes: same generic answer, then the same lock-out, in both flows
    finals = []
    for c in (new, dup):
        texts = [c.post("/verify-email", data={"code": "000000"}, follow_redirects=True).get_data(as_text=True)
                 for _ in range(5)]
        assert "incorrect or has expired" in texts[0]
        finals.append("Too many incorrect attempts" in texts[-1])
    assert finals == [True, True]
    assert db.query("SELECT COUNT(*) AS n FROM users WHERE email='taken@example.com'")[0]["n"] == 1


def test_duplicate_flow_never_signs_anyone_in_and_never_sends_a_code(app, db, coop, outbox):
    db.user("taken@example.com", coop_id=coop)
    c = app.test_client()
    register(c, coop, email="taken@example.com")
    assert all("code is:" not in m["body"] for m in outbox)
    c.post("/verify-email", data={"code": "123456"})
    assert c.get("/driver-dashboard").status_code in (302, 401)            # still anonymous


def test_tampered_pending_state_is_rejected(app, db, coop, outbox):
    c = app.test_client()
    register(c, coop)
    with c.session_transaction() as s:
        s["verify"] = {"t": s["verify"]["t"][:-6] + "AAAAAA", "masked": s["verify"]["masked"]}
    r = c.get("/verify-email")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


# ================================================================== 5. resend limits when delivery fails
@pytest.fixture
def failing_mail(monkeypatch):
    from website import mailer
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise mailer.MailError("smtp down")
    monkeypatch.setattr(mailer, "send", boom)
    return calls


def test_registration_resend_is_limited_when_delivery_fails(client, db, coop, failing_mail, clock):
    r = register(client, coop)
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "could not send the verification email" in page
    for _ in range(5):                                                     # hammering during the cooldown
        client.post("/verify-email/resend")
    assert len(failing_mail) == 1
    rows = db.query("SELECT invalidated_at, code_hash FROM email_otps")
    assert len(rows) == 1 and rows[0]["invalidated_at"] is not None       # undelivered code can never be used
    for _ in range(10):                                                    # one attempt per minute for 10 minutes
        db.query("UPDATE email_otps SET created_at = created_at - INTERVAL 61 SECOND")
        clock["offset"] += 61
        client.post("/verify-email/resend")
    assert len(failing_mail) == 5                                          # hourly cap still applies
    assert db.query("SELECT COUNT(*) AS n FROM email_otps WHERE invalidated_at IS NULL")[0]["n"] == 0
    assert "EMAIL_DELIVERY_FAILED" in [r["action"] for r in db.query("SELECT action FROM audit_logs")]


def test_signed_in_resend_is_limited_when_delivery_fails(client, db, coop, failing_mail):
    db.user("late@example.com", coop_id=coop, email_verified=False)
    login(client, "late@example.com")
    for _ in range(4):
        client.post("/verify-email/resend")
    assert len(failing_mail) == 1


# ================================================================== 2. nearby lookup cannot be spoofed
@pytest.fixture
def t(app, db):
    return Town(app, db)


def _cell(db, email="viewer@example.com"):
    r = db.query("SELECT p.approx_lat, p.approx_lon FROM driver_presence p JOIN users u ON u.id=p.user_id "
                 "WHERE u.email=%s", (email,))[0]
    return float(r["approx_lat"]), float(r["approx_lon"])


def _allow_next_update(db, seconds=30, email="viewer@example.com"):
    """Pretend the last update happened `seconds` ago (instead of sleeping in the test)."""
    db.query("UPDATE driver_presence p JOIN users u ON u.id=p.user_id SET p.updated_at = p.updated_at - INTERVAL %s SECOND, "
             "p.last_attempt_at = p.last_attempt_at - INTERVAL %s SECOND WHERE u.email=%s", (seconds, seconds, email))


def test_legitimate_lookup_uses_the_server_known_cell(t):
    t.driver("near@example.com")
    t.share("near@example.com", CLOSE)
    data = t.look()
    assert data["total"] == 1 and (data["you"]["approx_lat"], data["you"]["approx_lon"]) == (-1.94, 30.06)


def test_coordinates_sent_to_the_lookup_are_ignored(t):
    t.driver("far@example.com")
    t.share("far@example.com", FAR)
    t.share("viewer@example.com", HERE)
    r = post(t.c("viewer@example.com"), "/driver/nearby", {"lat": FAR[0], "lon": FAR[1]})
    data = r.get_json()
    assert r.status_code == 200 and data["total"] == 0                     # the far driver is not reachable
    assert (data["you"]["approx_lat"], data["you"]["approx_lon"]) == (-1.94, 30.06)


@pytest.mark.parametrize("forged", [(HERE[0] + 0.5, HERE[1]), (HERE[0], HERE[1] + 0.5), FAR],
                         ids=["forged-latitude", "forged-longitude", "arbitrary-place"])
def test_implausible_jumps_are_rejected_and_the_cell_is_kept(t, forged):
    t.share("viewer@example.com", HERE)
    _allow_next_update(t.db)
    r = t.share("viewer@example.com", forged)
    assert r.status_code == 409
    assert _cell(t.db) == (-1.94, 30.06)


def test_rapid_geographic_scanning_is_blocked(t):
    t.driver("target@example.com")
    t.share("target@example.com", FAR)
    t.share("viewer@example.com", HERE)
    found = 0
    for i in range(6):                                                      # try to hop around the country
        _allow_next_update(t.db, seconds=20)
        t.share("viewer@example.com", (FAR[0] + i * 0.01, FAR[1]))
        r = post(t.c("viewer@example.com"), "/driver/nearby", {})
        found += r.get_json().get("total", 0) if r.status_code == 200 else 0
    assert found == 0 and _cell(t.db) == (-1.94, 30.06)


def test_location_updates_have_a_minimum_interval(t):
    assert t.share("viewer@example.com", HERE).status_code == 200
    r = t.share("viewer@example.com", CLOSE)
    assert r.status_code == 429


def test_plausible_movement_is_accepted(t):
    t.share("viewer@example.com", HERE)
    _allow_next_update(t.db, seconds=60)
    assert t.share("viewer@example.com", CLOSE).status_code == 200
    assert _cell(t.db) == (-1.95, 30.07)


def test_lookups_are_rate_limited(t):
    t.share("viewer@example.com", HERE)
    codes = [post(t.c("viewer@example.com"), "/driver/nearby", {}).status_code for _ in range(16)]
    assert codes[:15] == [200] * 15 and codes[15] == 429


def test_lookup_needs_a_current_server_position(t):
    assert post(t.c("viewer@example.com"), "/driver/nearby", {}).status_code == 409   # nothing shared yet
    t.share("viewer@example.com", HERE)
    _allow_next_update(t.db, seconds=360)                                             # older than 5 minutes
    assert post(t.c("viewer@example.com"), "/driver/nearby", {}).status_code == 409


def test_imprecise_fixes_are_not_used(t):
    r = post(t.c("viewer@example.com"), "/driver/presence", {"lat": HERE[0], "lon": HERE[1], "accuracy": 5000})
    assert r.status_code == 422 and t.db.query("SELECT COUNT(*) AS n FROM driver_presence")[0]["n"] == 0


def test_unverified_and_other_roles_cannot_share_or_look(t):
    t.driver("pending@example.com", status="PENDING")
    assert t.share("pending@example.com", HERE).status_code == 403
    for role in ("manager", "umusare", "admin"):
        t.db.user(f"{role}@example.com", role=role, coop_id=t.coop if role != "admin" else None)
        c = t.c(f"{role}@example.com")
        assert post(c, "/driver/presence", {"lat": HERE[0], "lon": HERE[1]}).status_code == 403
        assert post(c, "/driver/nearby", {}).status_code == 403
    anon = t.app.test_client()
    assert post(anon, "/driver/nearby", {}).status_code in (302, 401)
    assert t.db.query("SELECT COUNT(*) AS n FROM driver_presence")[0]["n"] == 0


def test_no_exact_coordinates_or_contact_data_leak(t):
    t.driver("near@example.com")
    t.db.query("UPDATE users SET phone='+250788999000' WHERE email='near@example.com'")
    t.share("near@example.com", CLOSE)
    raw = json.dumps(t.look())
    for secret in ("1.951234", "30.068765", "1.944123", "30.061987", "+250788999000", "near@example.com", "RAB",
                   "DRV-", "driver_id"):
        assert secret not in raw, secret


# ================================================================== 6. stale presence
def test_presence_older_than_five_minutes_is_never_shown(t):
    fresh, stale = t.driver("fresh@example.com"), t.driver("stale@example.com")
    t.db.query("INSERT INTO driver_presence (user_id, approx_lat, approx_lon, updated_at) VALUES "
               "(%s,-1.95,30.07,UTC_TIMESTAMP(3) - INTERVAL 240 SECOND), "
               "(%s,-1.95,30.07,UTC_TIMESTAMP(3) - INTERVAL 301 SECOND)", (fresh, stale))
    assert t.names(t.look()) == {"fresh"}


# ================================================================== 3. privacy text matches the API
def test_nearby_fields_match_the_privacy_documentation(t, client):
    from website.nearby_service import DRIVER_FIELDS
    t.driver("near@example.com")
    t.share("near@example.com", CLOSE)
    d = t.look()["cells"][0]["drivers"][0]
    assert tuple(sorted(d)) == tuple(sorted(DRIVER_FIELDS))
    privacy = client.get("/privacy").get_data(as_text=True)
    terms = client.get("/terms").get_data(as_text=True)
    assert "(name, cooperative, group and approximate" in privacy
    assert "your name, cooperative and group" in terms and "never your Driver ID" in terms


# ================================================================== 4. open redirects
BAD_NEXT = ["//evil.com", "/\\evil.com", "\\\\evil.com", "/\\/evil.com", "https://evil.com", "http:evil.com",
            "/%5Cevil.com", "/%2F%2Fevil.com", "javascript:alert(1)", "/x\r\nLocation: https://evil.com", " /x", ""]


@pytest.mark.parametrize("target", BAD_NEXT)
def test_login_rejects_external_next(client, db, target):
    db.user("d@example.com", coop_id=db.cooperative())
    r = client.post("/login", query_string={"next": target}, data={"email": "d@example.com", "password": "Passw0rd!"})
    assert r.status_code == 302 and r.headers["Location"] == "/driver-dashboard"


@pytest.mark.parametrize("target", BAD_NEXT)
def test_terms_accept_rejects_external_next(client, db, target):
    db.user("d@example.com", coop_id=db.cooperative())
    login(client, "d@example.com")
    r = client.post("/terms/accept", data={"accept_terms": "1", "next": target})
    assert r.status_code == 302 and r.headers["Location"] == "/driver-dashboard"


def test_internal_next_still_works(client, db):
    db.user("d@example.com", coop_id=db.cooperative())
    r = client.post("/login", query_string={"next": "/chat/?c=abc#top"}, data={"email": "d@example.com", "password": "Passw0rd!"})
    assert r.headers["Location"] == "/chat/?c=abc#top"
    r = client.post("/terms/accept", data={"accept_terms": "1", "next": "/driver-dashboard#vehicle"})
    assert r.headers["Location"] == "/driver-dashboard#vehicle"


# ================================================================== 7. development secret
def test_development_secret_is_stable_and_production_still_requires_one(tmp_path, monkeypatch):
    from website import config
    from website.config import ConfigError, development_secret, flask_config
    key_file = tmp_path / "instance" / ".dev_secret_key"
    first, second = development_secret(str(key_file)), development_secret(str(key_file))
    assert first == second and len(first) >= 64
    monkeypatch.setattr(config, "DEV_SECRET_FILE", str(key_file))
    monkeypatch.delenv("SAFEDRIVE_SECRET_KEY", raising=False)
    monkeypatch.setenv("SAFEDRIVE_ENV", "development")
    assert flask_config()["SECRET_KEY"] == flask_config()["SECRET_KEY"] == first   # restarts keep pending OTPs valid
    monkeypatch.setenv("SAFEDRIVE_ENV", "testing")
    assert flask_config()["SECRET_KEY"] != flask_config()["SECRET_KEY"]             # tests: random per app
    monkeypatch.setenv("SAFEDRIVE_ENV", "production")
    monkeypatch.setenv("SAFEDRIVE_DB_PASSWORD", "x")
    with pytest.raises(ConfigError):
        flask_config()                                                             # production: never a fallback key
