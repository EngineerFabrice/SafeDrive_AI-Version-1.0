"""Email verification (OTP), Terms/Privacy acceptance and versioning, and verification-badge separation."""
import logging
import re
from datetime import timedelta

import pytest

from tests.web.conftest import login

pytestmark = pytest.mark.db

FORM = {"username": "Aline Uwase", "email": "aline@example.com", "password": "Passw0rd!", "confirm_password": "Passw0rd!",
        "role": "driver", "accept_terms": "1", "vehicle_plate_number": "rab123a"}


@pytest.fixture
def outbox():
    from website import mailer
    mailer.OUTBOX.clear()
    return mailer.OUTBOX


@pytest.fixture
def clock(monkeypatch):
    """Moves time.time() forward (the pending-verification cooldown lives in the encrypted session)."""
    import time as time_module
    real, state = time_module.time, {"offset": 0.0}
    monkeypatch.setattr(time_module, "time", lambda: real() + state["offset"])
    return state


@pytest.fixture
def coop(db):
    a = db.cooperative("Coop A", "CA")
    mgr = db.user("mgr@example.com", role="manager", coop_id=a)
    db.query("UPDATE cooperatives SET manager_user_id=%s WHERE id=%s", (mgr, a))
    return a


def register(client, coop_id, **over):
    return client.post("/register", data={**FORM, "cooperative_id": str(coop_id), **over})


def code_from(outbox):
    return re.search(r"code is: (\d{6})", outbox[-1]["body"]).group(1)


def user(db, email="aline@example.com"):
    return db.query("SELECT * FROM users WHERE email=%s", (email,))[0]


def wrong(code):
    return f"{(int(code) + 1) % 10 ** 6:06d}"


# ---------------------------------------------------------------- terms at registration
def test_registration_requires_explicit_terms_acceptance(client, db, coop):
    r = register(client, coop, accept_terms="")
    assert r.status_code == 200 and "accept the Terms" in r.get_data(as_text=True)
    assert db.query("SELECT COUNT(*) AS n FROM users WHERE role='driver'")[0]["n"] == 0


def test_terms_checkbox_is_not_preselected_and_links_documents(client, db, coop):
    html = client.get("/register").get_data(as_text=True)
    box = re.search(r'<input type="checkbox" name="accept_terms"[^>]*>', html).group(0)
    assert "checked" not in box and 'href="/terms"' in html and 'href="/privacy"' in html


def test_terms_and_privacy_versions_and_timestamps_are_stored(client, db, coop, outbox):
    register(client, coop)
    u = user(db)
    assert (u["terms_version"], u["privacy_version"]) == ("1.0", "1.0")
    assert u["terms_accepted_at"] is not None and u["privacy_accepted_at"] is not None
    rows = db.query("SELECT document, version FROM legal_acceptances WHERE user_id=%s ORDER BY document", (u["id"],))
    assert sorted((r["document"], r["version"]) for r in rows) == [("PRIVACY", "1.0"), ("TERMS", "1.0")]


def test_legal_pages_are_real_documents(client, db):
    terms = client.get("/terms").get_data(as_text=True)
    for text in ("does <b>not</b> measure blood alcohol concentration", "does <b>not</b> guarantee that a driver is safe",
                 "POTENTIALLY NOT SOBER", "Email verification", "Manager (cooperative) verification", "Vehicle information",
                 "not</b> a government certification", "SafeDrive does <b>not</b> process, hold or transfer",
                 "Limitation of liability", "Fabrice NDAYISABA", "fabricendayisaba16@gmail.com"):
        assert text in terms, text
    privacy = client.get("/privacy").get_data(as_text=True)
    assert "approximate area" in privacy and "does not keep a location history" in privacy


def test_updated_terms_require_reacceptance_without_overwriting_history(client, db, coop, monkeypatch):
    uid = db.user("d@example.com", coop_id=coop)
    db.query("UPDATE users SET terms_version='0.9', privacy_version='1.0' WHERE id=%s", (uid,))
    db.query("INSERT INTO legal_acceptances (user_id, document, version, accepted_at) VALUES (%s,'TERMS','0.9',UTC_TIMESTAMP(3))",
             (uid,))
    login(client, "d@example.com")
    assert 'id="terms-banner"' in client.get("/driver-dashboard").get_data(as_text=True)
    client.post("/terms/accept", data={"accept_terms": "1"})
    assert user(db, "d@example.com")["terms_version"] == "1.0"
    history = [r["version"] for r in db.query("SELECT version FROM legal_acceptances WHERE user_id=%s AND document='TERMS' "
                                              "ORDER BY id", (uid,))]
    assert history == ["0.9", "1.0"]
    assert 'id="terms-banner"' not in client.get("/driver-dashboard").get_data(as_text=True)


# ---------------------------------------------------------------- OTP
def test_registration_sends_otp_and_stores_only_a_hash(client, db, coop, outbox, caplog):
    caplog.set_level(logging.DEBUG)
    r = register(client, coop)
    assert r.headers["Location"].endswith("/verify-email")
    code = code_from(outbox)
    u = user(db)
    assert u["email_verified_at"] is None
    otp = db.query("SELECT * FROM email_otps WHERE user_id=%s", (u["id"],))[0]
    assert len(otp["code_hash"]) == 64 and code not in otp["code_hash"]
    assert (otp["expires_at"] - otp["created_at"]) == timedelta(minutes=10)
    page = client.get("/verify-email").get_data(as_text=True)
    assert "alin***@example.com" in page and code not in page
    audit = " ".join(str(r["details"]) for r in db.query("SELECT details FROM audit_logs"))
    assert code not in audit and code not in caplog.text


def test_correct_otp_verifies_email_signs_in_and_notifies_manager(client, db, coop, outbox):
    register(client, coop)
    r = client.post("/verify-email", data={"code": code_from(outbox)})
    assert r.headers["Location"].endswith("/driver-dashboard")
    u = user(db)
    assert u["email_verified_at"] is not None
    assert db.query("SELECT kind FROM notifications")[0]["kind"] == "VERIFICATION_REQUESTED"
    html = client.get("/driver-dashboard").get_data(as_text=True)
    assert "awaiting cooperative verification" in html and "mgr@example.com" in html
    assert db.query("SELECT verification_status FROM driver_profiles WHERE user_id=%s", (u["id"],))[0][
        "verification_status"] == "PENDING"          # email verified does NOT mean cooperative verified


def test_wrong_otp_is_refused_and_counted(client, db, coop, outbox):
    register(client, coop)
    r = client.post("/verify-email", data={"code": wrong(code_from(outbox))}, follow_redirects=True)
    assert "incorrect or has expired" in r.get_data(as_text=True)
    assert user(db)["email_verified_at"] is None
    assert db.query("SELECT attempts FROM email_otps")[0]["attempts"] == 1          # the failure is committed


def test_too_many_attempts_lock_the_code(client, db, coop, outbox):
    register(client, coop)
    code = code_from(outbox)
    for _ in range(5):
        client.post("/verify-email", data={"code": wrong(code)})
    r = client.post("/verify-email", data={"code": code}, follow_redirects=True)
    assert "Too many incorrect attempts" in r.get_data(as_text=True) and user(db)["email_verified_at"] is None


def test_expired_otp_is_refused(client, db, coop, outbox):
    register(client, coop)
    db.query("UPDATE email_otps SET expires_at = created_at - INTERVAL 1 SECOND")
    client.post("/verify-email", data={"code": code_from(outbox)})
    assert user(db)["email_verified_at"] is None


def test_new_code_invalidates_previous_and_resend_has_cooldown(client, db, coop, outbox, clock):
    register(client, coop)
    first = code_from(outbox)
    r = client.post("/verify-email/resend", follow_redirects=True)                  # immediately: cooldown
    assert "Please wait" in r.get_data(as_text=True) and len(outbox) == 1
    db.query("UPDATE email_otps SET created_at = created_at - INTERVAL 2 MINUTE, expires_at = expires_at - INTERVAL 2 MINUTE")
    clock["offset"] += 120
    client.post("/verify-email/resend")
    assert len(outbox) == 2
    second = code_from(outbox)
    if second != first:
        client.post("/verify-email", data={"code": first})
        assert user(db)["email_verified_at"] is None                               # old code no longer valid
    client.post("/verify-email", data={"code": second})
    assert user(db)["email_verified_at"] is not None


def test_otp_cannot_be_reused(client, db, coop, outbox):
    register(client, coop)
    code = code_from(outbox)
    client.post("/verify-email", data={"code": code})
    first = user(db)["email_verified_at"]
    r = client.post("/verify-email", data={"code": code}, follow_redirects=True)
    assert "already verified" in r.get_data(as_text=True) and user(db)["email_verified_at"] == first
    assert db.query("SELECT consumed_at FROM email_otps")[0]["consumed_at"] is not None


def test_hourly_code_limit(client, db, coop, outbox, clock):
    register(client, coop)
    for _ in range(6):
        db.query("UPDATE email_otps SET created_at = created_at - INTERVAL 61 SECOND")
        clock["offset"] += 61
        client.post("/verify-email/resend")
    assert db.query("SELECT COUNT(*) AS n FROM email_otps")[0]["n"] == 5


def test_duplicate_email_registration_reveals_nothing(client, db, coop, outbox):
    existing = db.user("aline@example.com", coop_id=coop)
    before = db.query("SELECT password_hash, email_verified_at FROM users WHERE id=%s", (existing,))[0]
    r = register(client, coop)
    assert r.headers["Location"].endswith("/verify-email")                         # same next step as a new account
    page = client.get("/verify-email").get_data(as_text=True)
    assert "alin***@example.com" in page and "already exists" not in page
    assert db.query("SELECT COUNT(*) AS n FROM users WHERE email='aline@example.com'")[0]["n"] == 1
    assert "code is:" not in outbox[-1]["body"] and "already have an account" in outbox[-1]["body"]
    r = client.post("/verify-email", data={"code": "123456"}, follow_redirects=True)
    assert "incorrect or has expired" in r.get_data(as_text=True)
    assert db.query("SELECT password_hash, email_verified_at FROM users WHERE id=%s", (existing,))[0] == before
    assert db.query("SELECT COUNT(*) AS n FROM email_otps")[0]["n"] == 0


def test_registration_rate_limit_per_network(client, db, coop, outbox, monkeypatch):
    monkeypatch.setenv("REGISTRATION_LIMIT_PER_HOUR", "2")
    for i in range(3):
        c = client.application.test_client()
        register(c, coop, email=f"user{i}@example.com")
    assert db.query("SELECT COUNT(*) AS n FROM users WHERE role='driver'")[0]["n"] == 2


def test_verify_page_requires_a_pending_or_signed_in_account(client, db):
    r = client.get("/verify-email")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_otp_endpoints_are_csrf_protected(db, coop, outbox):
    from website import create_app
    app = create_app({"WTF_CSRF_ENABLED": True, "BCRYPT_LOG_ROUNDS": 4, "TESTING": True})
    c = app.test_client()
    page = c.get("/register").get_data(as_text=True)
    token = page.split('name="csrf_token" value="')[1].split('"')[0]
    c.post("/register", data={**FORM, "cooperative_id": str(coop), "csrf_token": token})
    assert c.post("/verify-email", data={"code": code_from(outbox)}).status_code == 400
    assert c.post("/verify-email/resend").status_code == 400
    assert user(db)["email_verified_at"] is None


def test_unverified_email_banner_and_signed_in_verification(client, db, coop, outbox):
    db.user("late@example.com", coop_id=coop, email_verified=False)
    login(client, "late@example.com")
    html = client.get("/driver-dashboard").get_data(as_text=True)
    assert "EMAIL VERIFICATION REQUIRED" in html and "Please verify your email address to continue." in html
    client.post("/verify-email/resend")
    client.post("/verify-email", data={"code": code_from(outbox)})
    assert user(db, "late@example.com")["email_verified_at"] is not None


# ---------------------------------------------------------------- verification separation / blue badge
def _verified_driver(db, coop, email="v@example.com"):
    uid = db.user(email, coop_id=coop)
    db.query("UPDATE users SET terms_version='1.0', privacy_version='1.0' WHERE id=%s", (uid,))
    db.query("INSERT INTO driver_profiles (user_id, verification_status, vehicle_plate_number) "
             "VALUES (%s,'VERIFIED','RAB 123 A')", (uid,))
    return uid


def badge(uid):
    from website.badges import for_user
    return for_user(uid)


def test_blue_badge_only_when_every_condition_holds(app, db, coop):
    uid = _verified_driver(db, coop)
    with app.app_context():
        assert badge(uid)["verified"]
        db.query("UPDATE users SET email_verified_at=NULL WHERE id=%s", (uid,))
        assert not badge(uid)["verified"]                                          # cooperative verified, email not
        db.query("UPDATE users SET email_verified_at=UTC_TIMESTAMP(3) WHERE id=%s", (uid,))
        db.query("UPDATE driver_profiles SET verification_status='PENDING' WHERE user_id=%s", (uid,))
        assert not badge(uid)["verified"]                                          # email verified, manager pending
        db.query("UPDATE driver_profiles SET verification_status='SUSPENDED' WHERE user_id=%s", (uid,))
        assert not badge(uid)["verified"]
        db.query("UPDATE driver_profiles SET verification_status='VERIFIED', vehicle_plate_number=NULL WHERE user_id=%s", (uid,))
        assert not badge(uid)["verified"]                                          # plate missing
        db.query("UPDATE driver_profiles SET vehicle_plate_number='RAB 123 A' WHERE user_id=%s", (uid,))
        db.query("UPDATE users SET is_active=0 WHERE id=%s", (uid,))
        assert not badge(uid)["verified"]


def test_badge_states_are_kept_separate(app, db, coop):
    uid = _verified_driver(db, coop)
    db.query("UPDATE driver_profiles SET verification_status='PENDING' WHERE user_id=%s", (uid,))
    with app.app_context():
        st = badge(uid)["states"]
    assert st == {"email": "VERIFIED", "terms": "ACCEPTED", "cooperative": "MEMBER", "account": "ACTIVE", "manager": "PENDING"}


def test_dashboard_shows_badge_with_basis_and_no_safety_claim(client, db, coop):
    _verified_driver(db, coop)
    login(client, "v@example.com")
    html = client.get("/driver-dashboard").get_data(as_text=True)
    assert "Verified by SafeDrive" in html and "Verified by Cooperative Manager" in html
    assert "not a guarantee of driving safety" in html
    assert "safe and trusted" not in html.lower()


def test_pending_driver_has_no_blue_badge(client, db, coop):
    uid = db.user("p@example.com", coop_id=coop)
    db.query("INSERT INTO driver_profiles (user_id, vehicle_plate_number) VALUES (%s, 'RAB 123 A')", (uid,))
    login(client, "p@example.com")
    html = client.get("/driver-dashboard").get_data(as_text=True)
    assert "NOT YET VERIFIED" in html and "Verified by SafeDrive" not in html


def test_manager_cannot_verify_member_whose_email_is_unverified(client, db, coop):
    uid = db.user("noemail@example.com", coop_id=coop, membership="PENDING", email_verified=False)
    db.query("INSERT INTO driver_profiles (user_id, vehicle_plate_number) VALUES (%s, 'RAB 123 A')", (uid,))
    login(client, "mgr@example.com")
    r = client.post(f"/manager/members/{uid}/review", data={"action": "VERIFY"}, follow_redirects=True)
    assert "not verified their email" in r.get_data(as_text=True)
    assert db.query("SELECT verification_status FROM driver_profiles WHERE user_id=%s", (uid,))[0][
        "verification_status"] == "PENDING"
